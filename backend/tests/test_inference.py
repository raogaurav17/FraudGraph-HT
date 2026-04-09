"""
Basic tests for inference and API endpoints.
Run: pytest tests/ -v
"""

import numpy as np
from unittest.mock import patch, MagicMock
from app.ml.inference import score_transaction, _get_risk_level, _pad_features


def test_risk_level_mapping():
    assert _get_risk_level(0.90) == "CRITICAL"
    assert _get_risk_level(0.65) == "HIGH"
    assert _get_risk_level(0.35) == "MEDIUM"
    assert _get_risk_level(0.10) == "LOW"


def test_pad_features_short():
    feat = np.array([1.0, 2.0])
    result = _pad_features(feat, 5)
    assert len(result) == 5
    assert result[0] == 1.0
    assert result[4] == 0.0


def test_pad_features_long():
    feat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = _pad_features(feat, 4)
    assert len(result) == 4


def test_score_transaction_returns_valid_structure():
    features = np.array([3.5, 0.5, 0.8, 0.0, 2.0, 5.0, 0.01, 1.2], dtype=np.float32)
    result = score_transaction(txn_features=features)

    assert "fraud_probability" in result
    assert "risk_level" in result
    assert "model_version" in result
    assert "latency_ms" in result
    assert "explanation" in result

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert result["latency_ms"] > 0


def test_score_transaction_probability_range():
    for _ in range(10):
        features = np.random.randn(32).astype(np.float32)
        result = score_transaction(txn_features=features)
        assert 0.0 <= result["fraud_probability"] <= 1.0


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
