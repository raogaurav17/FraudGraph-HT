"""
Loss functions for fraud detection training.

Provides a small registry so training can switch losses from CLI.
"""

from typing import Callable

import torch
import torch.nn.functional as F


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

AVAILABLE_LOSSES = (
    "focal",
    "bce",
    "weighted_bce",
    "dice",
    "v2_focal",
    "weighted_focal_smooth",
)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for highly imbalanced binary classification."""
    labels = labels.float()
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    pt = labels * probs + (1.0 - labels) * (1.0 - probs)
    alpha_t = labels * alpha + (1.0 - labels) * (1.0 - alpha)
    loss = alpha_t * (1.0 - pt).pow(gamma) * bce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard binary cross-entropy with logits."""
    return F.binary_cross_entropy_with_logits(logits, labels.float(), reduction=reduction)


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross-entropy with positive-class weighting."""
    weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=weight,
        reduction=reduction,
    )


def dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss for binary labels to emphasize recall on rare positives."""
    labels = labels.float()
    probs = torch.sigmoid(logits)
    intersection = (probs * labels).sum()
    denom = probs.sum() + labels.sum()
    score = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - score


def weighted_focal_smooth_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.9,
    gamma: float = 2.0,
    smoothing: float = 0.05,
) -> torch.Tensor:
    """Weighted focal loss with label smoothing for noisy, imbalanced fraud labels."""
    labels = labels.float()
    labels_smooth = labels * (1.0 - smoothing) + 0.5 * smoothing
    bce = F.binary_cross_entropy_with_logits(logits, labels_smooth, reduction="none")
    pt = torch.exp(-bce)
    loss = alpha * (1.0 - pt).pow(gamma) * bce
    return loss.mean()


def build_loss_fn(
    loss_name: str,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: float | None = None,
    smoothing: float = 0.05,
) -> LossFn:
    """Build a callable loss function based on name and hyperparameters."""
    name = loss_name.lower().strip()

    if name == "focal":
        return lambda logits, labels: focal_loss(
            logits,
            labels,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

    if name == "bce":
        return bce_loss

    if name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("pos_weight must be set for weighted_bce")
        return lambda logits, labels: weighted_bce_loss(
            logits,
            labels,
            pos_weight=pos_weight,
        )

    if name == "dice":
        return dice_loss

    if name in ("v2_focal", "weighted_focal_smooth"):
        return lambda logits, labels: weighted_focal_smooth_loss(
            logits,
            labels,
            alpha=focal_alpha,
            gamma=focal_gamma,
            smoothing=smoothing,
        )

    available = ", ".join(AVAILABLE_LOSSES)
    raise ValueError(f"Unsupported loss function '{loss_name}'. Available: {available}")