"""Visualization helpers for model training diagnostics and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _plot_training_curves(history: List[Dict], output_dir: Path) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    losses = [row.get("loss", np.nan) for row in history]
    val_auprc = [row.get("val_auprc", np.nan) for row in history]
    val_auroc = [row.get("val_auroc", np.nan) for row in history]
    val_precision = [row.get("val_precision", np.nan) for row in history]
    val_recall = [row.get("val_recall", np.nan) for row in history]
    val_f1 = [row.get("val_f1", np.nan) for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, losses, marker="o", color="#1f77b4")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, val_auprc, marker="o", color="#2ca02c", label="Val AUPRC")
    axes[0, 1].plot(epochs, val_auroc, marker="o", color="#ff7f0e", label="Val AUROC")
    axes[0, 1].set_title("Validation Ranking Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, val_precision, marker="o", label="Val Precision", color="#d62728")
    axes[1, 0].plot(epochs, val_recall, marker="o", label="Val Recall", color="#9467bd")
    axes[1, 0].plot(epochs, val_f1, marker="o", label="Val F1", color="#8c564b")
    axes[1, 0].set_title("Validation Thresholded Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    best_idx = int(np.nanargmax(val_auprc)) if len(val_auprc) > 0 else 0
    axes[1, 1].bar(
        ["Best AUPRC", "Best AUROC"],
        [float(np.nanmax(val_auprc)), float(np.nanmax(val_auroc))],
        color=["#2ca02c", "#ff7f0e"],
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title(f"Best Validation Summary (Epoch {epochs[best_idx]})")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_validation_curves.png", dpi=180)
    plt.close(fig)


def _plot_precision_recall(y_true: np.ndarray, y_score: np.ndarray, output_dir: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig = plt.figure(figsize=(7.5, 6))
    plt.plot(recall, precision, color="#2ca02c", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)


def _plot_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, output_dir: Path) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    precision_vals = []
    recall_vals = []
    f1_vals = []

    for threshold in thresholds:
        metrics = _binary_metrics_at_threshold(y_true, y_score, float(threshold))
        precision_vals.append(metrics["precision"])
        recall_vals.append(metrics["recall"])
        f1_vals.append(metrics["f1"])

    fig = plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_vals, label="Precision", color="#d62728", linewidth=2)
    plt.plot(thresholds, recall_vals, label="Recall", color="#9467bd", linewidth=2)
    plt.plot(thresholds, f1_vals, label="F1", color="#8c564b", linewidth=2)
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold vs Precision / Recall / F1")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_vs_precision_recall_f1.png", dpi=180)
    plt.close(fig)


def _plot_score_distribution(y_true: np.ndarray, y_score: np.ndarray, output_dir: Path) -> None:
    fraud_scores = y_score[y_true == 1]
    normal_scores = y_score[y_true == 0]

    fig = plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.65, density=True, label="Non-Fraud", color="#1f77b4")
    plt.hist(fraud_scores, bins=50, alpha=0.65, density=True, label="Fraud", color="#d62728")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Predicted Probability Distribution (Fraud vs Non-Fraud)")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution_fraud_vs_nonfraud.png", dpi=180)
    plt.close(fig)


def _plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = _safe_roc_auc(y_true, y_score)

    fig = plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.4f})", color="#ff7f0e", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)


def _plot_calibration_curve(y_true: np.ndarray, y_score: np.ndarray, output_dir: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=12, strategy="quantile")

    fig = plt.figure(figsize=(7.5, 6))
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="#17becf", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve_reliability_diagram.png", dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig = plt.figure(figsize=(6.8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix @ threshold={threshold:.2f}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred: Non-Fraud", "Pred: Fraud"], rotation=15)
    plt.yticks(tick_marks, ["True: Non-Fraud", "True: Fraud"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def _plot_test_metric_bars(test_metrics: Dict[str, float], output_dir: Path) -> None:
    keys = ["auprc", "auroc", "precision", "recall", "f1"]
    vals = [float(test_metrics.get(k, np.nan)) for k in keys]

    fig = plt.figure(figsize=(8.5, 5.5))
    plt.bar([k.upper() for k in keys], vals, color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"])
    plt.ylim(0.0, 1.0)
    plt.title("Test Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "test_metrics_bar_chart.png", dpi=180)
    plt.close(fig)


def create_training_artifacts(
    history: List[Dict],
    test_y_true: np.ndarray,
    test_y_score: np.ndarray,
    test_metrics: Dict[str, float],
    threshold: float,
    output_dir: Path,
    run_summary: Dict,
    threshold_scan: List[Dict] | None = None,
) -> Dict[str, str]:
    """Generate and save plots/metrics artifacts for one training run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_training_curves(history, output_dir)
    _plot_precision_recall(test_y_true, test_y_score, output_dir)
    _plot_threshold_metrics(test_y_true, test_y_score, output_dir)
    _plot_score_distribution(test_y_true, test_y_score, output_dir)
    _plot_roc_curve(test_y_true, test_y_score, output_dir)
    _plot_calibration_curve(test_y_true, test_y_score, output_dir)
    _plot_confusion_matrix(test_y_true, test_y_score, threshold, output_dir)
    _plot_test_metric_bars(test_metrics, output_dir)

    if threshold_scan is None:
        threshold_scan = []
        for threshold_value in np.linspace(0.05, 0.5, 50):
            threshold_scan.append(
                _binary_metrics_at_threshold(test_y_true, test_y_score, float(threshold_value))
            )

    summary = {
        **run_summary,
        "threshold": float(threshold),
        "test_metrics": test_metrics,
        "roc_auc": _safe_roc_auc(test_y_true, test_y_score),
        "n_test": int(test_y_true.shape[0]),
        "n_test_fraud": int((test_y_true == 1).sum()),
        "n_test_non_fraud": int((test_y_true == 0).sum()),
    }

    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "threshold_scan_metrics.json").write_text(json.dumps(threshold_scan, indent=2), encoding="utf-8")

    return {
        "artifacts_dir": str(output_dir),
        "training_curves": str(output_dir / "training_validation_curves.png"),
        "pr_curve": str(output_dir / "precision_recall_curve.png"),
        "threshold_curve": str(output_dir / "threshold_vs_precision_recall_f1.png"),
        "score_distribution": str(output_dir / "score_distribution_fraud_vs_nonfraud.png"),
        "roc_curve": str(output_dir / "roc_curve.png"),
        "calibration_curve": str(output_dir / "calibration_curve_reliability_diagram.png"),
        "confusion_matrix": str(output_dir / "confusion_matrix.png"),
        "test_metric_bars": str(output_dir / "test_metrics_bar_chart.png"),
        "summary_json": str(output_dir / "metrics_summary.json"),
        "threshold_scan_json": str(output_dir / "threshold_scan_metrics.json"),
    }