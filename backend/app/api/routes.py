"""
FastAPI API routers for FraudGraph.
Routes: /predict, /transactions, /model, /stats, /ws
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, case
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import uuid
import json
import numpy as np
import logging

from app.core.database import get_db
from app.core.redis import cache_get, cache_set, get_redis
from app.models.db import Transaction, FraudPrediction, RiskLevel, DatasetSource
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
    db: AsyncSession = Depends(get_db),
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

    # Persist prediction
    db_write_ok = False
    try:
        txn_db = Transaction(
            transaction_id=txn_in.transaction_id,
            amount=txn_in.amount,
            channel=txn_in.channel,
            card_id=txn_in.card_id,
            merchant_id=txn_in.merchant_id,
            device_id=txn_in.device_id,
            dataset_source=DatasetSource(txn_in.dataset_source or "live"),
        )
        db.add(txn_db)

        pred_db = FraudPrediction(
            transaction_id=txn_in.transaction_id,
            fraud_probability=result["fraud_probability"],
            risk_level=RiskLevel(result["risk_level"]),
            model_version=result["model_version"],
            top_features=result["explanation"].get("top_features", []),
            inference_latency_ms=result["latency_ms"],
        )
        db.add(pred_db)
        await db.commit()
        db_write_ok = True
    except Exception as e:
        await db.rollback()
        log.error(f"Failed to persist prediction: {e}")

    resp_dict = response.model_dump()

    # Only emit/cache from fresh computations when persistence succeeded.
    if db_write_ok:
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
    else:
        raise HTTPException(status_code=500, detail="Prediction could not be persisted")

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
    db: AsyncSession = Depends(get_db),
):
    """List transactions with their fraud predictions."""
    cache_key = f"txns:{page}:{page_size}:{risk_level}:{dataset_source}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    stmt = (
        select(Transaction, FraudPrediction)
        .outerjoin(FraudPrediction, Transaction.transaction_id == FraudPrediction.transaction_id)
        .order_by(desc(Transaction.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    if risk_level:
        stmt = stmt.where(FraudPrediction.risk_level == risk_level)
    if dataset_source:
        stmt = stmt.where(Transaction.dataset_source == dataset_source)

    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for txn, pred in rows:
        item = {
            "id": str(txn.id),
            "transaction_id": txn.transaction_id,
            "amount": txn.amount,
            "channel": txn.channel,
            "card_id": txn.card_id,
            "merchant_id": txn.merchant_id,
            "dataset_source": txn.dataset_source,
            "created_at": txn.created_at.isoformat() if txn.created_at else None,
            "is_fraud": txn.is_fraud,
        }
        if pred:
            item.update({
                "fraud_probability": pred.fraud_probability,
                "risk_level": pred.risk_level,
                "model_version": pred.model_version,
                "latency_ms": pred.inference_latency_ms,
            })
        items.append(item)

    # Count total
    count_result = await db.execute(select(func.count(Transaction.id)))
    total = count_result.scalar() or 0

    response = {"items": items, "total": total, "page": page, "page_size": page_size}
    await cache_set(cache_key, response, ttl=30)
    return response


@router.get("/transactions/{transaction_id}", tags=["Data"])
async def get_transaction(transaction_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Transaction, FraudPrediction)
        .outerjoin(FraudPrediction, Transaction.transaction_id == FraudPrediction.transaction_id)
        .where(Transaction.transaction_id == transaction_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Transaction not found")

    txn, pred = row
    return {
        "transaction": {k: v for k, v in txn.__dict__.items() if not k.startswith("_")},
        "prediction": {k: v for k, v in pred.__dict__.items() if not k.startswith("_")} if pred else None,
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
async def stats_overview(db: AsyncSession = Depends(get_db)):
    """Dashboard overview statistics."""
    cached = await cache_get("stats:overview")
    if cached:
        return cached

    # Total transactions
    total_result = await db.execute(select(func.count(Transaction.id)))
    total = total_result.scalar() or 0

    # Fraud flagged
    fraud_result = await db.execute(
        select(func.count(FraudPrediction.id))
        .where(FraudPrediction.risk_level.in_([RiskLevel.HIGH, RiskLevel.CRITICAL]))
    )
    fraud_flagged = fraud_result.scalar() or 0

    # Avg latency
    lat_result = await db.execute(select(func.avg(FraudPrediction.inference_latency_ms)))
    avg_latency = round(float(lat_result.scalar() or 0), 2)

    # Risk distribution
    risk_result = await db.execute(
        select(FraudPrediction.risk_level, func.count(FraudPrediction.id))
        .group_by(FraudPrediction.risk_level)
    )
    risk_dist = {
        (row[0].value if hasattr(row[0], "value") else str(row[0])): row[1]
        for row in risk_result.all()
        if row[0]
    }

    # Dataset breakdown
    ds_result = await db.execute(
        select(Transaction.dataset_source, func.count(Transaction.id))
        .group_by(Transaction.dataset_source)
    )
    dataset_dist = {str(row[0]): row[1] for row in ds_result.all() if row[0]}

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
    db: AsyncSession = Depends(get_db),
):
    """Hourly transaction and fraud counts for charts."""
    cached = await cache_get(f"stats:ts:{hours}")
    if cached:
        return cached

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Aggregate by hour using raw SQL via DuckDB-style grouping
    result = await db.execute(
        select(
            func.date_trunc("hour", Transaction.created_at).label("hour"),
            func.count(Transaction.id).label("total"),
            func.sum(
                case(
                    (FraudPrediction.risk_level.in_([RiskLevel.HIGH, RiskLevel.CRITICAL]), 1),
                    else_=0,
                )
            ).label("fraud"),
        )
        .outerjoin(FraudPrediction, Transaction.transaction_id == FraudPrediction.transaction_id)
        .where(Transaction.created_at >= cutoff)
        .group_by(func.date_trunc("hour", Transaction.created_at))
        .order_by(func.date_trunc("hour", Transaction.created_at))
    )

    rows = result.all()
    series = [{"hour": str(r[0]), "total": r[1], "fraud": int(r[2] or 0)} for r in rows]

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
