"""
FastAPI API routers for FraudGraph.
Routes: /predict, /transactions, /model, /stats, /ws
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from enum import Enum
import uuid
import json
import numpy as np
import logging

from app.core.redis import cache_get, cache_set, get_redis
from app.ml.inference import score_transaction, get_model_info
from app.core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────

class TransactionIn(BaseModel):
    transaction_id: Optional[str] = None
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    card_id: Optional[str] = None
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None
    channel: Optional[str] = "online"
    product_category: Optional[str] = None
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)
    country_mismatch: Optional[bool] = False
    velocity_1h: Optional[float] = 0.0
    velocity_24h: Optional[float] = 0.0
    card_avg_amount_30d: Optional[float] = None
    merchant_fraud_rate: Optional[float] = 0.0
    dataset_source: Optional[str] = "live"

    @field_validator("transaction_id", mode="before")
    @classmethod
    def set_txn_id(cls, v):
        return v or str(uuid.uuid4())


class PredictionOut(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    model_version: str
    latency_ms: float
    explanation: Dict[str, Any]
    created_at: str


class BatchPredictIn(BaseModel):
    transactions: List[TransactionIn]


class TransactionListOut(BaseModel):
    items: List[Dict]
    total: int
    page: int
    page_size: int


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DatasetSource(str, Enum):
    IEEE_CIS = "ieee_cis"
    PAYSIM = "paysim"
    ELLIPTIC = "elliptic"
    LIVE = "live"


transactions_store: Dict[str, Dict[str, Any]] = {}
predictions_store: Dict[str, Dict[str, Any]] = {}


def _build_txn_features(txn_in: TransactionIn) -> np.ndarray:
    """Build a consistent feature vector for both single and batch inference."""
    import math

    log_amount = math.log1p(txn_in.amount)
    hour = txn_in.hour_of_day or 12
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    return np.array([
        log_amount,
        hour_sin,
        hour_cos,
        float(txn_in.country_mismatch or 0),
        txn_in.velocity_1h or 0,
        txn_in.velocity_24h or 0,
        txn_in.merchant_fraud_rate or 0,
        (txn_in.card_avg_amount_30d or txn_in.amount) / (txn_in.amount + 1),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionOut, tags=["Inference"])
async def predict(
    txn_in: TransactionIn,
):
    """Score a single transaction for fraud probability."""
    # Check cache
    cache_key = f"score:{txn_in.transaction_id}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    # Build feature vector
    txn_features = _build_txn_features(txn_in)

    # Score
    result = score_transaction(txn_features=txn_features)

    response = PredictionOut(
        transaction_id=txn_in.transaction_id,
        fraud_probability=result["fraud_probability"],
        risk_level=result["risk_level"],
        model_version=result["model_version"],
        latency_ms=result["latency_ms"],
        explanation=result["explanation"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    now = datetime.now(timezone.utc)
    ds_value = (txn_in.dataset_source or DatasetSource.LIVE.value).lower()
    try:
        ds = DatasetSource(ds_value)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid dataset_source: {ds_value}") from e

    transactions_store[txn_in.transaction_id] = {
        "id": str(uuid.uuid4()),
        "transaction_id": txn_in.transaction_id,
        "dataset_source": ds.value,
        "amount": txn_in.amount,
        "channel": txn_in.channel,
        "product_category": txn_in.product_category,
        "card_id": txn_in.card_id,
        "merchant_id": txn_in.merchant_id,
        "device_id": txn_in.device_id,
        "created_at": now,
        "is_fraud": None,
    }
    predictions_store[txn_in.transaction_id] = {
        "transaction_id": txn_in.transaction_id,
        "fraud_probability": result["fraud_probability"],
        "risk_level": result["risk_level"],
        "model_version": result["model_version"],
        "top_features": result["explanation"].get("top_features", []),
        "inference_latency_ms": result["latency_ms"],
        "created_at": now,
    }

    resp_dict = response.model_dump()

    try:
        await cache_set(cache_key, resp_dict, ttl=60)
    except Exception as e:
        log.warning(f"Failed to cache prediction response: {e}")

    try:
        r = await get_redis()
        await r.publish("fraud_scores", json.dumps({
            "type": "new_prediction",
            "data": resp_dict,
        }))
    except Exception as e:
        log.warning(f"Failed to publish prediction to Redis channel: {e}")

    return response


@router.post("/predict/batch", tags=["Inference"])
async def predict_batch(payload: BatchPredictIn):
    """Score multiple transactions in one request."""
    if len(payload.transactions) > 100:
        raise HTTPException(status_code=422, detail="Max 100 transactions per batch")

    results = []
    for txn in payload.transactions:
        features = _build_txn_features(txn)
        r = score_transaction(txn_features=features)
        r["transaction_id"] = txn.transaction_id
        results.append(r)

    return {"predictions": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────
# TRANSACTIONS
# ─────────────────────────────────────────────────────────────────

@router.get("/transactions", response_model=TransactionListOut, tags=["Data"])
async def list_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    risk_level: Optional[str] = None,
    dataset_source: Optional[str] = None,
):
    """List transactions with their fraud predictions."""
    cache_key = f"txns:{page}:{page_size}:{risk_level}:{dataset_source}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    merged_rows: List[Dict[str, Any]] = []
    for txn in transactions_store.values():
        pred = predictions_store.get(txn["transaction_id"])
        if risk_level and (not pred or str(pred.get("risk_level", "")).upper() != risk_level.upper()):
            continue
        if dataset_source and str(txn.get("dataset_source", "")).lower() != dataset_source.lower():
            continue
        merged_rows.append({"txn": txn, "pred": pred})

    merged_rows.sort(key=lambda row: row["txn"].get("created_at") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    paged_rows = merged_rows[(page - 1) * page_size: page * page_size]

    items: List[Dict[str, Any]] = []
    for row in paged_rows:
        txn = row["txn"]
        pred = row["pred"]
        item = {
            "id": txn["id"],
            "transaction_id": txn["transaction_id"],
            "amount": txn["amount"],
            "channel": txn["channel"],
            "card_id": txn["card_id"],
            "merchant_id": txn["merchant_id"],
            "dataset_source": txn["dataset_source"],
            "created_at": txn["created_at"].isoformat() if txn.get("created_at") else None,
            "is_fraud": txn.get("is_fraud"),
        }
        if pred:
            item.update({
                "fraud_probability": pred["fraud_probability"],
                "risk_level": pred["risk_level"],
                "model_version": pred["model_version"],
                "latency_ms": pred["inference_latency_ms"],
            })
        items.append(item)

    total = len(transactions_store)

    response = {"items": items, "total": total, "page": page, "page_size": page_size}
    await cache_set(cache_key, response, ttl=30)
    return response


@router.get("/transactions/{transaction_id}", tags=["Data"])
async def get_transaction(transaction_id: str):
    txn = transactions_store.get(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    pred = predictions_store.get(transaction_id)
    return {
        "transaction": {
            **txn,
            "created_at": txn["created_at"].isoformat() if txn.get("created_at") else None,
        },
        "prediction": {
            **pred,
            "created_at": pred["created_at"].isoformat() if pred.get("created_at") else None,
        } if pred else None,
    }


# ─────────────────────────────────────────────────────────────────
# MODEL & STATS
# ─────────────────────────────────────────────────────────────────

@router.get("/model/info", tags=["Model"])
async def model_info():
    """Get current model version and metrics."""
    cached = await cache_get("model:info")
    if cached:
        return cached
    info = get_model_info()
    await cache_set("model:info", info, ttl=300)
    return info


@router.get("/stats/overview", tags=["Analytics"])
async def stats_overview():
    """Dashboard overview statistics."""
    cached = await cache_get("stats:overview")
    if cached:
        return cached

    total = len(transactions_store)

    fraud_flagged = sum(
        1
        for pred in predictions_store.values()
        if pred.get("risk_level") in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value)
    )

    latencies = [float(pred.get("inference_latency_ms") or 0) for pred in predictions_store.values()]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    risk_dist: Dict[str, int] = {}
    for pred in predictions_store.values():
        rl = str(pred.get("risk_level"))
        if not rl:
            continue
        risk_dist[rl] = risk_dist.get(rl, 0) + 1

    dataset_dist: Dict[str, int] = {}
    for txn in transactions_store.values():
        ds = str(txn.get("dataset_source"))
        if not ds:
            continue
        dataset_dist[ds] = dataset_dist.get(ds, 0) + 1

    response = {
        "total_transactions": total,
        "fraud_flagged": fraud_flagged,
        "fraud_rate": round(fraud_flagged / max(total, 1), 4),
        "avg_latency_ms": avg_latency,
        "risk_distribution": risk_dist,
        "dataset_distribution": dataset_dist,
    }
    await cache_set("stats:overview", response, ttl=60)
    return response


@router.get("/stats/timeseries", tags=["Analytics"])
async def stats_timeseries(
    hours: int = Query(24, ge=1, le=168),
):
    """Hourly transaction and fraud counts for charts."""
    cached = await cache_get(f"stats:ts:{hours}")
    if cached:
        return cached

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    buckets: Dict[datetime, Dict[str, int]] = {}
    for txn_id, txn in transactions_store.items():
        created_at = txn.get("created_at")
        if not isinstance(created_at, datetime) or created_at < cutoff:
            continue
        hour_bucket = created_at.replace(minute=0, second=0, microsecond=0)
        if hour_bucket not in buckets:
            buckets[hour_bucket] = {"total": 0, "fraud": 0}
        buckets[hour_bucket]["total"] += 1

        pred = predictions_store.get(txn_id)
        if pred and pred.get("risk_level") in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value):
            buckets[hour_bucket]["fraud"] += 1

    series = [
        {"hour": hour.isoformat(), "total": vals["total"], "fraud": vals["fraud"]}
        for hour, vals in sorted(buckets.items(), key=lambda item: item[0])
    ]

    await cache_set(f"stats:ts:{hours}", series, ttl=120)
    return series


# ─────────────────────────────────────────────────────────────────
# WEBSOCKET — real-time fraud score stream
# ─────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        try:
            self.connections.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, message: str):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@router.websocket("/ws/scores")
async def websocket_scores(websocket: WebSocket):
    """Real-time WebSocket stream of fraud scores."""
    await manager.connect(websocket)
    pubsub = None

    try:
        r = await get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe("fraud_scores")

        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"])
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning(f"WebSocket Redis stream error: {e}")
    finally:
        manager.disconnect(websocket)
        if pubsub is not None:
            await pubsub.unsubscribe("fraud_scores")
