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
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)

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
) -> Dict[str, float]:
    y_true, y_score = predict_scores(model, data, mask_key, device, node_type)
    y_pred = (y_score >= settings.model_threshold).astype(int)

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
    model.eval()
    with torch.no_grad():
        # For simplicity: full-graph inference (use NeighborLoader in production)
        logits = model(
            {k: v.to(device) for k, v in data.x_dict.items()},
            {k: v.to(device) for k, v in data.edge_index_dict.items()},
        )
        mask = getattr(data[node_type], mask_key)
        y_true = data[node_type].y[mask].numpy()
        y_score = torch.sigmoid(logits[mask]).cpu().numpy()
    return y_true.astype(int), y_score.astype(float)


def train_epoch(
    model: HTGNN,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    node_type: str = "txn",
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x_dict, batch.edge_index_dict)
        labels = batch[node_type].y.float()
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def train(
    dataset_name: str = "ieee_cis",
    epochs: int = 50,
    hidden: int = 128,
    heads: int = 4,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    patience: int = 8,
    loss_name: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Optional[float] = None,
    max_rows: Optional[int] = None,
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

    # ── Data loader ──
    train_mask = data[node_type].train_mask
    loader = NeighborLoader(
        data,
        num_neighbors={rel: [16, 8] for rel in data.metadata()[1]},
        batch_size=batch_size,
        input_nodes=(node_type, train_mask),
        shuffle=True,
    )

    effective_pos_weight = pos_weight
    if loss_name == "weighted_bce" and effective_pos_weight is None:
        train_labels = data[node_type].y[train_mask].float()
        positives = float(train_labels.sum().item())
        negatives = float(train_labels.numel() - positives)
        effective_pos_weight = negatives / max(positives, 1.0)
        log.info(f"Auto-computed pos_weight={effective_pos_weight:.4f} from train split")

    loss_fn = build_loss_fn(
        loss_name=loss_name,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        pos_weight=effective_pos_weight,
    )
    log.info(f"Using loss function: {loss_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    early_stop = EarlyStopping(patience=patience)

    best_val_auprc = 0.0
    best_state = None
    history = []

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, loader, optimizer, loss_fn, device, node_type)
        val_metrics = evaluate(model, data, "val_mask", device, node_type)
        scheduler.step()

        epoch_time = time.time() - t0
        log.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if early_stop(val_metrics["auprc"]):
            log.info(f"Early stopping at epoch {epoch}")
            break

    # ── Test evaluation ──
    if best_state:
        model.load_state_dict(best_state)
    test_y_true, test_y_score = predict_scores(model, data, "test_mask", device, node_type)
    test_metrics = evaluate(model, data, "test_mask", device, node_type)
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
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "pos_weight": effective_pos_weight,
        },
        "train_metrics": history[-1] if history else {},
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
        threshold=settings.model_threshold,
        output_dir=run_artifacts_dir,
        run_summary={
            "version": version,
            "run_name": run_name,
            "dataset": primary_dataset,
            "loss_name": loss_name,
            "model_path": str(save_path),
            "log_path": str(run_log_path),
            "n_params": int(n_params),
        },
    )
    log.info(f"Artifacts saved to {run_artifacts_dir}")

    return {
        "version": version,
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--loss", choices=list(AVAILABLE_LOSSES), default="focal")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
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
        pos_weight=args.pos_weight,
        max_rows=args.max_rows,
        save_path=args.save_path,
        artifacts_dir=args.artifacts_dir,
        logs_dir=args.logs_dir,
    )
    print(json.dumps(result, indent=2, default=str))
