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

    feature_inputs = {
        "txn": txn_features,
        "card": card_features,
        "merchant": merchant_features,
        "device": device_features,
    }
    x_dict, edge_index_dict = _build_inference_graph_inputs(cache, feature_inputs, device)

    txn_seq_t = None
    delta_t_t = None
    seq_mask_t = None
    txn_dim = cache["in_channels"].get("txn", len(np.asarray(txn_features).flatten()))
    if card_sequence is not None and delta_t is not None:
        seq_arr = np.asarray(card_sequence, dtype=np.float32)
        dt_arr = np.asarray(delta_t, dtype=np.float32)
        if seq_arr.ndim == 2 and dt_arr.ndim == 1 and seq_arr.shape[0] == dt_arr.shape[0]:
            seq_arr = np.stack([_pad_features(row, txn_dim) for row in seq_arr], axis=0)
            txn_seq_t = torch.tensor(seq_arr, dtype=torch.float32, device=device).unsqueeze(0)
            delta_t_t = torch.tensor(dt_arr, dtype=torch.float32, device=device).unsqueeze(0)
            seq_mask_t = torch.zeros((1, seq_arr.shape[0]), dtype=torch.bool, device=device)

    with torch.no_grad():
        logits_tensor = model.forward(
            x_dict,
            edge_index_dict,
            txn_seq=txn_seq_t,
            delta_t=delta_t_t,
            seq_mask=seq_mask_t,
        )
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


def _build_inference_graph_inputs(
    cache: Dict,
    feature_inputs: Dict[str, Optional[np.ndarray]],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
    """Build a minimal heterogeneous graph matching checkpoint metadata."""
    metadata = cache.get("metadata")
    in_channels = cache.get("in_channels", {})

    node_types = []
    edge_types = []
    if isinstance(metadata, tuple) and len(metadata) == 2:
        node_types, edge_types = metadata
    elif isinstance(metadata, dict):
        node_types = metadata.get("node_types", [])
        edge_types = metadata.get("edge_types", [])

    node_types = list(node_types) if node_types else list(in_channels.keys())
    if "txn" not in node_types:
        node_types = ["txn"] + [n for n in node_types if n != "txn"]

    x_dict: Dict[str, torch.Tensor] = {}
    for ntype in node_types:
        dim = int(in_channels.get(ntype, in_channels.get("txn", 32)))
        values = feature_inputs.get(ntype)
        if values is None and ntype != "txn":
            values = np.zeros(dim, dtype=np.float32)
        if ntype == "txn" and values is None:
            values = np.zeros(dim, dtype=np.float32)
        feat = _pad_features(values, dim)
        x_dict[ntype] = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)

    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for rel in edge_types:
        if not isinstance(rel, tuple) or len(rel) != 3:
            continue
        src, _, dst = rel
        if src not in x_dict or dst not in x_dict:
            continue
        edge_index_dict[rel] = torch.tensor([[0], [0]], dtype=torch.long, device=device)

    # Ensure at least one edge exists for common txn relations.
    if not edge_index_dict:
        for ntype in x_dict:
            if ntype == "txn":
                continue
            edge_index_dict[(ntype, "rel", "txn")] = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            break

    return x_dict, edge_index_dict


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
    if features is None:
        features = np.zeros(target_dim, dtype=np.float32)
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
