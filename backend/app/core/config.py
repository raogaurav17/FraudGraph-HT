from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "FraudGraph API"
    env: str = "development"
    debug: bool = True

    # Database
    database_url: str = "postgresql+asyncpg://fraud:fraud_secret@localhost:5432/fraudgraph"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 300  # seconds

    # ML Model
    model_path: str = "models/htgnn_latest.pt"
    model_threshold: float = 0.5
    batch_size: int = 256
    device: str = "cpu"  # "cuda" if GPU available

    # Fraud thresholds
    fraud_threshold_high: float = 0.80
    fraud_threshold_medium: float = 0.50
    fraud_threshold_low: float = 0.25

    # Datasets
    data_dir: str = "/data"
    ieee_cis_path: str = "/data/ieee_cis"
    paysim_path: str = "/data/paysim"
    elliptic_path: str = "/data/elliptic"

    # WebSocket
    ws_ping_interval: int = 30

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
