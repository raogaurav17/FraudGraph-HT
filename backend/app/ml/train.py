"""
Training script for HTGNN fraud detection model.
Supports multiple datasets, early stopping, checkpointing.

Usage:
    python -m app.ml.train --dataset ieee_cis --epochs 50
    python -m app.ml.train --dataset all --epochs 30
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report
)
import numpy as np

from app.ml.model import HTGNN, focal_loss, build_model
from app.ml.pipeline import load_all_datasets, DatasetSplit
from app.core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


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


def train_epoch(
    model: HTGNN,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
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
        loss = focal_loss(logits, labels)

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
    save_path: Optional[str] = None,
) -> Dict:

    device = torch.device(settings.device)
    log.info(f"Training on device: {device}")

    # ── Load data ──
    log.info(f"Loading dataset: {dataset_name}")
    datasets = load_all_datasets(settings.data_dir, [dataset_name])

    if not datasets:
        raise RuntimeError(f"No datasets loaded. Check data directory: {settings.data_dir}")

    split = datasets[dataset_name]
    data = split.train

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    early_stop = EarlyStopping(patience=patience)

    best_val_auprc = 0.0
    best_state = None
    history = []

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, loader, optimizer, device, node_type)
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
    test_metrics = evaluate(model, data, "test_mask", device, node_type)
    log.info(f"Test metrics: {test_metrics}")

    # ── Save checkpoint ──
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_path or settings.model_path
    checkpoint = {
        "version": version,
        "model_state": best_state or model.state_dict(),
        "metadata": data.metadata(),
        "in_channels": in_channels,
        "node_type": node_type,
        "hyperparams": {
            "hidden": hidden, "heads": heads, "dropout": dropout
        },
        "train_metrics": history[-1] if history else {},
        "test_metrics": test_metrics,
        "dataset": dataset_name,
        "n_params": n_params,
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    log.info(f"Model saved to {save_path}")

    return {
        "version": version,
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "n_params": n_params,
        "history": history,
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
    args = parser.parse_args()

    result = train(
        dataset_name=args.dataset,
        epochs=args.epochs,
        hidden=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, default=str))
