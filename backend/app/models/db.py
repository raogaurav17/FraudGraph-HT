from sqlalchemy import (
    Column, String, Float, Boolean, Integer,
    DateTime, JSON, Enum, ForeignKey, Index, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
import enum
from app.core.database import Base


class RiskLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DatasetSource(str, enum.Enum):
    IEEE_CIS = "ieee_cis"
    PAYSIM = "paysim"
    ELLIPTIC = "elliptic"
    LIVE = "live"


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    dataset_source = Column(Enum(DatasetSource), nullable=False, default=DatasetSource.LIVE)

    # Transaction features
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    channel = Column(String(20))          # online, pos, atm, mobile
    product_category = Column(String(50))

    # Entity references (graph nodes)
    card_id = Column(String(200), index=True)
    merchant_id = Column(String(200), index=True)
    device_id = Column(String(200), index=True)
    ip_address = Column(String(45))

    # Temporal
    transaction_dt = Column(Integer)      # timedelta from reference
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Ground truth (available after investigation)
    is_fraud = Column(Boolean, nullable=True)

    # Computed graph features (cached after extraction)
    card_txn_count_7d = Column(Float)
    card_avg_amount_30d = Column(Float)
    card_unique_merchants_7d = Column(Integer)
    merchant_fraud_rate_90d = Column(Float)

    # Raw metadata
    raw_features = Column(JSON)

    # Relationship
    prediction = relationship("FraudPrediction", back_populates="transaction", uselist=False)

    __table_args__ = (
        Index("ix_transactions_card_dt", "card_id", "transaction_dt"),
        Index("ix_transactions_created_fraud", "created_at", "is_fraud"),
    )


class FraudPrediction(Base):
    __tablename__ = "fraud_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String(100), ForeignKey("transactions.transaction_id"), nullable=False, index=True)

    # Model output
    fraud_probability = Column(Float, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    model_version = Column(String(50), nullable=False)

    # Explanation
    top_features = Column(JSON)           # [{name, contribution, value}, ...]
    graph_context = Column(JSON)          # {card_fraud_neighbors, shared_device_cards, ...}

    # Latency
    inference_latency_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    transaction = relationship("Transaction", back_populates="prediction")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(50), unique=True, nullable=False)
    description = Column(Text)

    # Training metadata
    datasets_used = Column(JSON)         # list of DatasetSource
    training_samples = Column(Integer)
    fraud_rate = Column(Float)
    train_auprc = Column(Float)
    val_auprc = Column(Float)
    test_auprc = Column(Float)
    train_auroc = Column(Float)
    val_auroc = Column(Float)

    # Architecture
    hidden_dim = Column(Integer)
    num_layers = Column(Integer)
    num_heads = Column(Integer)
    parameters = Column(Integer)

    # Status
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    metrics = relationship("ModelMetricSnapshot", back_populates="model_version")


class ModelMetricSnapshot(Base):
    __tablename__ = "model_metric_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)

    # Rolling window metrics (computed hourly)
    window_hours = Column(Integer, default=24)
    predictions_count = Column(Integer)
    fraud_flagged = Column(Integer)
    true_positives = Column(Integer)
    false_positives = Column(Integer)
    precision = Column(Float)
    recall = Column(Float)
    auprc = Column(Float)
    avg_latency_ms = Column(Float)

    snapshot_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    model_version = relationship("ModelVersion", back_populates="metrics")
