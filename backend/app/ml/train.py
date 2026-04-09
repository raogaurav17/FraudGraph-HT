"""
Training script for HTGNN fraud detection model.
Supports multiple datasets, early stopping, checkpointing.

Usage:
    uv run python -m app.ml.train --dataset ieee_cis --epochs 5
    uv run python -m app.ml.train --dataset ieee_cis --loss weighted_bce --pos-weight 8.0
    uv run python -m app.ml.train --dataset ieee_cis --loss focal --focal-alpha 0.5 --focal-gamma 1.5
    uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --save-path ../models/custom_name.pt
    uv run python -m app.ml.train --dataset ieee_cis --epochs 5 --logs-dir ../logs
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)
from scipy.special import expit

from app.ml.model import HTGNN, build_model
from app.ml.losses import AVAILABLE_LOSSES, LossFn, build_loss_fn
from app.ml.pipeline import load_all_datasets
from app.ml.visualization import create_training_artifacts
from app.core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _format_float_token(value: float) -> str:
    """Format float hyperparameters into filename-safe short tokens."""
    text = f"{value:.6g}"  # compact, stable precision
    return text.replace("-", "m").replace(".", "p")


def build_checkpoint_filename(
    dataset_name: str,
    loss_name: str,
    hidden: int,
    heads: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    version: str,
) -> str:
    """Create descriptive checkpoint filename for multi-run experimentation."""
    return (
        f"htgnn_{dataset_name}"
        f"_loss-{loss_name}"
        f"_h{hidden}"
        f"_heads{heads}"
        f"_do{_format_float_token(dropout)}"
        f"_lr{_format_float_token(lr)}"
        f"_bs{batch_size}"
        f"_ep{epochs}"
        f"_{version}.pt"
    )


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stopped = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


def evaluate(
    model: HTGNN,
    data,
    mask_key: str,
    device: torch.device,
    node_type: str = "txn",
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    y_true, y_score = predict_scores(model, data, mask_key, device, node_type)
    if not np.isfinite(y_score).all():
        bad = int((~np.isfinite(y_score)).sum())
        log.warning("Non-finite y_score detected during %s evaluation (%d values); applying nan_to_num.", mask_key, bad)
        y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)

    score_threshold = settings.model_threshold if threshold is None else threshold
    y_pred = (y_score >= score_threshold).astype(int)

    metrics = {
        "auprc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "precision": float((y_pred * y_true).sum() / (y_pred.sum() + 1e-8)),
        "recall": float((y_pred * y_true).sum() / (y_true.sum() + 1e-8)),
    }
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
        metrics["precision"] + metrics["recall"] + 1e-8
    )
    return metrics


def predict_scores(
    model: HTGNN,
    data,
    mask_key: str,
    device: torch.device,
    node_type: str = "txn",
) -> Tuple[np.ndarray, np.ndarray]:
    y_true, logits = predict_logits(model, data, mask_key, device, node_type)
    logits = np.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    return y_true, expit(logits).astype(float)


def predict_logits(
    model: HTGNN,
    data,
    mask_key: str,
    device: torch.device,
    node_type: str = "txn",
    num_neighbors: Optional[Dict] = None,
    eval_batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    mask = getattr(data[node_type], mask_key)
    neighbors = num_neighbors or {rel: [8, 4] for rel in data.metadata()[1]}
    eval_loader = NeighborLoader(
        data,
        num_neighbors=neighbors,
        batch_size=eval_batch_size,
        input_nodes=(node_type, mask),
        shuffle=False,
    )

    full_txn_x = data[node_type].x
    full_seq_index = getattr(data[node_type], "seq_index", None)
    full_delta_t = getattr(data[node_type], "delta_t", None)
    full_seq_mask = getattr(data[node_type], "seq_mask", None)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            n_seed = batch[node_type].batch_size
            txn_seq, delta_t, seq_mask = build_batch_sequence_inputs(
                batch=batch,
                node_type=node_type,
                full_txn_x=full_txn_x,
                full_seq_index=full_seq_index,
                full_delta_t=full_delta_t,
                full_seq_mask=full_seq_mask,
                device=device,
            )
            logits = model(
                batch.x_dict,
                batch.edge_index_dict,
                txn_seq=txn_seq,
                delta_t=delta_t,
                seq_mask=seq_mask,
            )[:n_seed]
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            labels = batch[node_type].y[:n_seed]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            # Release batch-scoped tensors early to lower peak memory.
            del batch, txn_seq, delta_t, seq_mask, logits, labels
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    y_logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    return y_true.astype(int), y_logits.astype(float)


def apply_platt_scaling(logits: np.ndarray, calibration: Optional[Dict[str, float]]) -> np.ndarray:
    if not calibration:
        return expit(logits)
    scale = float(calibration.get("scale", 1.0))
    bias = float(calibration.get("bias", 0.0))
    return expit(scale * logits + bias)


def compute_metrics_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    precision = float((y_pred * y_true).sum() / (y_pred.sum() + 1e-8))
    recall = float((y_pred * y_true).sum() / (y_true.sum() + 1e-8))
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
    }


def fit_platt_scaler(logits: np.ndarray, labels: np.ndarray) -> Optional[Dict[str, float]]:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return None

    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(np.asarray(logits).reshape(-1, 1), labels)
    return {
        "method": "platt",
        "scale": float(clf.coef_[0][0]),
        "bias": float(clf.intercept_[0]),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, Dict[str, float], list]:
    best_threshold = float(thresholds[0])
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    scan_rows = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        scan_rows.append(row)
        if f1 > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = row

    return best_threshold, best_metrics, scan_rows


def train_epoch(
    model: HTGNN,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    node_type: str = "txn",
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    full_txn_x: Optional[torch.Tensor] = None,
    full_seq_index: Optional[torch.Tensor] = None,
    full_delta_t: Optional[torch.Tensor] = None,
    full_seq_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    skipped_batches = 0

    for batch in loader:
        batch = batch.to(device)
        n_seed = batch[node_type].batch_size
        txn_seq, delta_t, seq_mask = build_batch_sequence_inputs(
            batch=batch,
            node_type=node_type,
            full_txn_x=full_txn_x,
            full_seq_index=full_seq_index,
            full_delta_t=full_delta_t,
            full_seq_mask=full_seq_mask,
            device=device,
        )
        logits = model(
            batch.x_dict,
            batch.edge_index_dict,
            txn_seq=txn_seq,
            delta_t=delta_t,
            seq_mask=seq_mask,
        )[:n_seed]
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        labels = batch[node_type].y[:n_seed].float()

        if not torch.isfinite(logits).all():
            finite_rows = torch.isfinite(logits)
            if not finite_rows.any():
                skipped_batches += 1
                log.warning("Skipping batch with all non-finite logits")
                del batch, txn_seq, delta_t, seq_mask, logits, labels
                continue

            bad_count = int((~finite_rows).sum().item())
            log.warning("Dropping %d non-finite logits from batch", bad_count)
            logits = logits[finite_rows]
            labels = labels[finite_rows]

        loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            skipped_batches += 1
            log.warning("Skipping batch with non-finite loss")
            del batch, txn_seq, delta_t, seq_mask, logits, labels, loss
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Release batch-scoped tensors early to lower peak memory.
        del batch, txn_seq, delta_t, seq_mask, logits, labels, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(n_batches, 1)
    if skipped_batches:
        log.warning("Skipped %d training batches due to non-finite values", skipped_batches)
    return {"loss": avg_loss}


def build_batch_sequence_inputs(
    batch,
    node_type: str,
    full_txn_x: Optional[torch.Tensor],
    full_seq_index: Optional[torch.Tensor],
    full_delta_t: Optional[torch.Tensor],
    full_seq_mask: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Materialize [B, K, D] sequence tensors only for seed nodes in the current batch."""
    if (
        full_txn_x is None
        or full_seq_index is None
        or full_delta_t is None
        or full_seq_mask is None
        or not hasattr(batch[node_type], "n_id")
    ):
        return None, None, None

    n_seed = int(batch[node_type].batch_size)
    global_seed_ids = batch[node_type].n_id[:n_seed].detach().cpu()

    seq_index = full_seq_index.index_select(0, global_seed_ids)
    delta_t = full_delta_t.index_select(0, global_seed_ids)
    seq_mask = full_seq_mask.index_select(0, global_seed_ids)

    seq_len = seq_index.size(1)
    safe_index = seq_index.clamp(min=0)
    flat_index = safe_index.reshape(-1)
    txn_seq = full_txn_x.index_select(0, flat_index).reshape(n_seed, seq_len, -1)

    # Zero-out padded positions so clamped index values do not leak signal.
    txn_seq = txn_seq * (~seq_mask).unsqueeze(-1).float()
    txn_seq = torch.nan_to_num(txn_seq, nan=0.0, posinf=0.0, neginf=0.0)
    delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)

    return txn_seq.to(device), delta_t.to(device), seq_mask.to(device)


def train(
    dataset_name: str = "ieee_cis",
    epochs: int = 50,
    hidden: int = 128,
    heads: int = 4,
    dropout: float = 0.3,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    patience: int = 8,
    loss_name: str = "weighted_focal_smooth",
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    smoothing: float = 0.05,
    pos_weight: Optional[float] = None,
    max_rows: Optional[int] = None,
    max_v_features: int = 120,
    max_cards_per_device: int = 20,
    neighbor_hop1: int = 4,
    neighbor_hop2: int = 2,
    feature_mode: str = "original",
    save_path: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
) -> Dict:

    device = torch.device(settings.device)
    log.info(f"Training on device: {device}")

    # ── Load data ──
    log.info(f"Loading dataset: {dataset_name}")
    
    # Support 'all' to load all available datasets
    if dataset_name == "all":
        dataset_list = ["ieee_cis", "paysim", "elliptic"]
    else:
        dataset_list = [dataset_name]
    
    datasets = load_all_datasets(
        settings.data_dir,
        dataset_list,
        ieee_max_rows=max_rows if dataset_name == "ieee_cis" else None,
        ieee_max_v_features=max_v_features,
        ieee_max_cards_per_device=max_cards_per_device,
        ieee_feature_mode=feature_mode,
    )

    if not datasets:
        raise RuntimeError(f"No datasets loaded. Check data directory: {settings.data_dir}")

    # Handle 'all' by using the first loaded dataset (prefer ieee_cis)
    if dataset_name == "all":
        primary_dataset = "ieee_cis" if "ieee_cis" in datasets else list(datasets.keys())[0]
        log.info(f"Training on primary dataset: {primary_dataset}")
    else:
        primary_dataset = dataset_name
    
    split = datasets[primary_dataset]
    data = split.train

    # Use a stable run version for model/artifacts/log naming.
    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save_path:
        resolved_save_path = Path(save_path)
    else:
        models_dir = PROJECT_ROOT / "models"
        filename = build_checkpoint_filename(
            dataset_name=primary_dataset,
            loss_name=loss_name,
            hidden=hidden,
            heads=heads,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            version=version,
        )
        resolved_save_path = models_dir / filename
    save_path = str(resolved_save_path)
    run_name = Path(save_path).stem

    # Configure per-run file logging under logs/.
    logs_root = Path(logs_dir) if logs_dir else PROJECT_ROOT / "logs" / "training"
    logs_root.mkdir(parents=True, exist_ok=True)
    run_log_path = logs_root / f"{run_name}.log"
    root_logger = logging.getLogger()
    if not any(
        isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == run_log_path
        for h in root_logger.handlers
    ):
        file_handler = logging.FileHandler(run_log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root_logger.addHandler(file_handler)
    log.info(f"Training logs will be written to {run_log_path}")

    # Detect primary node type
    node_type = "txn" if "txn" in data.node_types else data.node_types[0]
    log.info(f"Primary node type: {node_type}")

    # ── Build model ──
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = build_model(
        data.metadata(), in_channels,
        hidden=hidden, heads=heads, dropout=dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    # ── Oversample positives ──
    train_mask = data[node_type].train_mask
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_labels_idx = data[node_type].y[train_idx]
    
    pos_idx = train_idx[train_labels_idx == 1]
    neg_idx = train_idx[train_labels_idx == 0]
    
    # Duplicate positive samples to balance the batch a bit more, say 4x
    if len(pos_idx) > 0 and len(pos_idx) < len(neg_idx):
        dup_factor = min(4, len(neg_idx) // len(pos_idx))
        balanced_train_idx = torch.cat([neg_idx] + [pos_idx] * dup_factor)
        # shuffle
        balanced_train_idx = balanced_train_idx[torch.randperm(len(balanced_train_idx))]
    else:
        balanced_train_idx = train_idx
    
    neighbor_sizes = {
        rel: [max(1, int(neighbor_hop1)), max(1, int(neighbor_hop2))]
        for rel in data.metadata()[1]
    }
    loader = NeighborLoader(
        data,
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        input_nodes=(node_type, balanced_train_idx),
        shuffle=True,
    )

    full_txn_x = data[node_type].x
    full_seq_index = getattr(data[node_type], "seq_index", None)
    full_delta_t = getattr(data[node_type], "delta_t", None)
    full_seq_mask = getattr(data[node_type], "seq_mask", None)
    train_labels = data[node_type].y[train_mask].float()
    effective_pos_weight = pos_weight
    if loss_name == "weighted_bce" and effective_pos_weight is None:
        positives = float(train_labels.sum().item())
        negatives = float(train_labels.numel() - positives)
        effective_pos_weight = negatives / max(positives, 1.0)
        log.info(f"Auto-computed pos_weight={effective_pos_weight:.4f} from train split")

    effective_focal_alpha = focal_alpha
    if loss_name == "focal" and focal_alpha == 0.25:
        fraud_rate = float(train_labels.mean().item())
        effective_focal_alpha = 1.0 - fraud_rate
        log.info(f"Auto-computed focal_alpha={effective_focal_alpha:.4f} from train split")


    loss_fn = build_loss_fn(
        loss_name=loss_name,
        focal_alpha=effective_focal_alpha,
        focal_gamma=focal_gamma,
        pos_weight=effective_pos_weight,
        smoothing=smoothing,
    )
    log.info(f"Using loss function: {loss_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    early_stop = EarlyStopping(patience=patience)

    best_val_auprc = 0.0
    best_state = None
    history = []

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            device,
            node_type,
            scheduler=scheduler,
            full_txn_x=full_txn_x,
            full_seq_index=full_seq_index,
            full_delta_t=full_delta_t,
            full_seq_mask=full_seq_mask,
        )
        val_metrics = evaluate(model, data, "val_mask", device, node_type)

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        log.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if scheduler is not None:
            scheduler.step(val_metrics["auprc"])

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if early_stop(val_metrics["auprc"]):
            log.info(f"Early stopping at epoch {epoch}")
            break

    # ── Test evaluation ──
    if best_state:
        model.load_state_dict(best_state)

    val_y_true, val_logits = predict_logits(
        model,
        data,
        "val_mask",
        device,
        node_type,
        num_neighbors=neighbor_sizes,
    )
    calibration = fit_platt_scaler(val_logits, val_y_true)
    val_y_score = apply_platt_scaling(val_logits, calibration)

    thresholds = np.linspace(0.05, 0.5, 50)
    best_threshold, best_val_threshold_metrics, threshold_scan = sweep_thresholds(
        val_y_true,
        val_y_score,
        thresholds,
    )
    log.info(
        "Selected threshold %.3f on validation set | precision=%.4f | recall=%.4f | f1=%.4f",
        best_threshold,
        best_val_threshold_metrics["precision"],
        best_val_threshold_metrics["recall"],
        best_val_threshold_metrics["f1"],
    )

    test_y_true, test_logits = predict_logits(
        model,
        data,
        "test_mask",
        device,
        node_type,
        num_neighbors=neighbor_sizes,
    )
    test_y_score = apply_platt_scaling(test_logits, calibration)
    test_metrics = compute_metrics_from_scores(test_y_true, test_y_score, best_threshold)
    log.info(f"Test metrics: {test_metrics}")

    # ── Save checkpoint ──
    checkpoint = {
        "version": version,
        "model_state": best_state or model.state_dict(),
        "metadata": data.metadata(),
        "in_channels": in_channels,
        "node_type": node_type,
        "hyperparams": {
            "hidden": hidden,
            "heads": heads,
            "dropout": dropout,
            "loss_name": loss_name,
            "focal_alpha": effective_focal_alpha,
            "focal_gamma": focal_gamma,
            "pos_weight": effective_pos_weight,
            "feature_mode": feature_mode,
            "max_v_features": max_v_features,
            "neighbor_hop1": neighbor_hop1,
            "neighbor_hop2": neighbor_hop2,
        },
        "decision_threshold": best_threshold,
        "calibration": calibration,
        "train_metrics": history[-1] if history else {},
        "val_threshold_metrics": best_val_threshold_metrics,
        "threshold_scan": threshold_scan,
        "test_metrics": test_metrics,
        "dataset": primary_dataset,
        "n_params": n_params,
    }
    
    # Ensure directory exists and is writable
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    
    try:
        torch.save(checkpoint, save_path)
        # Ensure file is readable/writable
        os.chmod(save_path, 0o644)
        log.info(f"Model saved to {save_path}")
        print("Threshold for fraud classification:")
        print(f"  High: {settings.fraud_threshold_high}")
        print(f"  Medium: {settings.fraud_threshold_medium}")
        print(f"  Low: {settings.fraud_threshold_low}")
    except Exception as e:
        log.error(f"Failed to save model to {save_path}: {e}")
        raise

    # ── Save training/evaluation artifacts ──
    artifacts_root = Path(artifacts_dir) if artifacts_dir else PROJECT_ROOT / "artifacts"
    run_artifacts_dir = artifacts_root / "training" / run_name
    artifact_paths = create_training_artifacts(
        history=history,
        test_y_true=test_y_true,
        test_y_score=test_y_score,
        test_metrics=test_metrics,
        threshold=best_threshold,
        output_dir=run_artifacts_dir,
        run_summary={
            "version": version,
            "run_name": run_name,
            "dataset": primary_dataset,
            "loss_name": loss_name,
            "model_path": str(save_path),
            "log_path": str(run_log_path),
            "decision_threshold": best_threshold,
            "calibration": calibration,
            "n_params": int(n_params),
        },
        threshold_scan=threshold_scan,
    )
    log.info(f"Artifacts saved to {run_artifacts_dir}")

    return {
        "version": version,
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "decision_threshold": best_threshold,
        "calibration": calibration,
        "n_params": n_params,
        "history": history,
        "log_path": str(run_log_path),
        "artifacts": artifact_paths,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ieee_cis")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loss", choices=list(AVAILABLE_LOSSES), default="weighted_focal_smooth")
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-v-features", type=int, default=120)
    parser.add_argument("--max-cards-per-device", type=int, default=20)
    parser.add_argument("--neighbor-hop1", type=int, default=4)
    parser.add_argument("--neighbor-hop2", type=int, default=2)
    parser.add_argument("--feature-mode", choices=["original", "enhanced"], default="original")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--artifacts-dir", default='artifacts')
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()

    result = train(
        dataset_name=args.dataset,
        epochs=args.epochs,
        hidden=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        loss_name=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        smoothing=args.smoothing,
        pos_weight=args.pos_weight,
        max_rows=args.max_rows,
        max_v_features=args.max_v_features,
        max_cards_per_device=args.max_cards_per_device,
        neighbor_hop1=args.neighbor_hop1,
        neighbor_hop2=args.neighbor_hop2,
        feature_mode=args.feature_mode,
        save_path=args.save_path,
        artifacts_dir=args.artifacts_dir,
        logs_dir=args.logs_dir,
    )
    print(json.dumps(result, indent=2, default=str))
