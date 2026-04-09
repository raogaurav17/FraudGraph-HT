"""
Inference service — loads trained HTGNN checkpoint and scores
individual transactions in real-time with explanation.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from scipy.special import expit

from app.ml.model import HTGNN, build_model
from app.core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

_model_cache: Optional[Dict] = None


def _load_compatible_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> bool:
    """Load only parameters whose keys and shapes match the current model."""
    model_state = model.state_dict()
    compatible_state = {
        key: tensor
        for key, tensor in state_dict.items()
        if key in model_state and model_state[key].shape == tensor.shape
    }
    if not compatible_state:
        return False

    model.load_state_dict(compatible_state, strict=False)
    return True


def load_model(model_path: str = None) -> Dict:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    path = model_path or settings.model_path
    if not Path(path).exists():
        log.warning(f"No model found at {path}. Using random weights for demo.")
        return _create_demo_model()

    log.info(f"Loading model from {path}")
    checkpoint = torch.load(path, map_location=settings.device)

    model = build_model(
        checkpoint["metadata"],
        checkpoint["in_channels"],
        **checkpoint.get("hyperparams", {}),
    )
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        log.warning("Model checkpoint is not fully compatible with current architecture: %s", exc)
        loaded = _load_compatible_state_dict(model, checkpoint.get("model_state", {}))
        if not loaded:
            log.warning("Falling back to demo model because no compatible weights could be loaded.")
            demo_cache = _create_demo_model()
            _model_cache = demo_cache
            return demo_cache
    model.eval()

    _model_cache = {
        "model": model,
        "version": checkpoint.get("version", "unknown"),
        "node_type": checkpoint.get("node_type", "txn"),
        "metadata": checkpoint.get("metadata"),
        "in_channels": checkpoint.get("in_channels"),
        "test_metrics": checkpoint.get("test_metrics", {}),
        "decision_threshold": checkpoint.get("decision_threshold", settings.model_threshold),
        "calibration": checkpoint.get("calibration"),
    }
    log.info(f"Loaded model v{_model_cache['version']}, test AUPRC={_model_cache['test_metrics'].get('auprc', 'N/A')}")
    return _model_cache


def _create_demo_model() -> Dict:
    """Create a demo model with random weights for UI demonstration."""
    from torch_geometric.data import HeteroData
    import torch.nn as nn

    # Minimal metadata for demo
    node_types = ["txn", "card", "merchant"]
    edge_types = [
        ("card", "makes", "txn"),
        ("txn", "at", "merchant"),
    ]
    metadata = (node_types, edge_types)
    in_channels = {"txn": 32, "card": 4, "merchant": 3}

    model = build_model(metadata, in_channels, hidden=64, heads=2)
    model.eval()
    return {
        "model": model,
        "version": "demo_v0",
        "node_type": "txn",
        "metadata": metadata,
        "in_channels": in_channels,
        "test_metrics": {"auprc": 0.0, "auroc": 0.0},
        "decision_threshold": settings.model_threshold,
        "calibration": None,
        "is_demo": True,
    }


def score_transaction(
    txn_features: np.ndarray,
    card_features: Optional[np.ndarray] = None,
    merchant_features: Optional[np.ndarray] = None,
    device_features: Optional[np.ndarray] = None,
    card_sequence: Optional[np.ndarray] = None,
    delta_t: Optional[np.ndarray] = None,
) -> Dict:
    """
    Score a single transaction for fraud probability.

    Returns:
        {
          fraud_probability: float,
          risk_level: str,
          model_version: str,
          latency_ms: float,
          explanation: {...}
        }
    """
    t0 = time.perf_counter()
    cache = load_model()
    model: HTGNN = cache["model"]
    device = torch.device(settings.device)

    # Build minimal HeteroData-like dicts for inference
    # In production, this would pull graph context from the DB
    txn_dim = cache["in_channels"].get("txn", 32)
    card_dim = cache["in_channels"].get("card", 4)
    merch_dim = cache["in_channels"].get("merchant", 3)

    # Pad/truncate features to expected dimensions
    txn_feat = _pad_features(txn_features, txn_dim)
    card_feat = _pad_features(card_features or np.zeros(card_dim), card_dim)
    merch_feat = _pad_features(merchant_features or np.zeros(merch_dim), merch_dim)

    x_dict = {
        "txn": torch.tensor(txn_feat, dtype=torch.float32).unsqueeze(0).to(device),
        "card": torch.tensor(card_feat, dtype=torch.float32).unsqueeze(0).to(device),
        "merchant": torch.tensor(merch_feat, dtype=torch.float32).unsqueeze(0).to(device),
    }

    # Minimal edge index (self-loop for demo; real usage uses graph context)
    edge_index_dict = {
        ("card", "makes", "txn"): torch.tensor([[0], [0]], dtype=torch.long).to(device),
        ("txn", "at", "merchant"): torch.tensor([[0], [0]], dtype=torch.long).to(device),
    }

    with torch.no_grad():
        logits_tensor = model.forward(x_dict, edge_index_dict)
        logits = float(logits_tensor.reshape(-1)[0].item())
        calibration = cache.get("calibration")
        if calibration:
                prob = expit(calibration["scale"] * logits + calibration["bias"])
        else:
                prob = expit(logits)

    latency_ms = (time.perf_counter() - t0) * 1000

    risk_level = _get_risk_level(prob)
    explanation = _explain(txn_features, prob)

    return {
        "fraud_probability": round(prob, 4),
        "risk_level": risk_level,
        "model_version": cache["version"],
        "decision_threshold": cache.get("decision_threshold", settings.model_threshold),
        "latency_ms": round(latency_ms, 2),
        "explanation": explanation,
    }


def batch_score(transactions: List[Dict]) -> List[Dict]:
    """Score multiple transactions in one forward pass."""
    results = []
    for txn in transactions:
        result = score_transaction(
            txn_features=txn.get("features", np.zeros(32)),
            card_features=txn.get("card_features"),
            merchant_features=txn.get("merchant_features"),
        )
        result["transaction_id"] = txn.get("transaction_id")
        results.append(result)
    return results


def get_model_info() -> Dict:
    cache = load_model()
    return {
        "version": cache["version"],
        "test_metrics": cache["test_metrics"],
        "in_channels": cache["in_channels"],
        "decision_threshold": cache.get("decision_threshold", settings.model_threshold),
        "calibration": cache.get("calibration"),
        "is_demo": cache.get("is_demo", False),
    }


def _pad_features(features: np.ndarray, target_dim: int) -> np.ndarray:
    features = np.array(features, dtype=np.float32).flatten()
    if len(features) >= target_dim:
        return features[:target_dim]
    return np.pad(features, (0, target_dim - len(features)))


def _get_risk_level(prob: float) -> str:
    if prob >= settings.fraud_threshold_high:
        return "CRITICAL"
    elif prob >= settings.fraud_threshold_medium:
        return "HIGH"
    elif prob >= settings.fraud_threshold_low:
        return "MEDIUM"
    return "LOW"


def _explain(features: np.ndarray, prob: float) -> Dict:
    """Simple explanation based on feature magnitude."""
    feature_names = [
        "log_amount", "hour_sin", "hour_cos", "balance_diff",
        "velocity_1h", "velocity_24h", "cross_border",
    ]
    features = np.array(features).flatten()
    denom = float(np.abs(features[:7]).sum())
    denom = max(denom, 1.0)

    top_features = []
    for i, name in enumerate(feature_names):
        if i < len(features):
            contribution = float(abs(features[i]) * prob / denom)
            if not np.isfinite(contribution):
                contribution = 0.0
            top_features.append({
                "name": name,
                "value": round(float(features[i]), 4),
                "contribution": round(contribution, 4),
            })

    top_features.sort(key=lambda x: x["contribution"], reverse=True)

    return {
        "top_features": top_features[:5],
        "confidence": "high" if prob > 0.8 or prob < 0.2 else "medium",
    }
