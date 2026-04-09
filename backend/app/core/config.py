from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data")
DEFAULT_IEEE_CIS_PATH = str(PROJECT_ROOT / "data" / "ieee_cis")
DEFAULT_PAYSIM_PATH = str(PROJECT_ROOT / "data" / "paysim")
DEFAULT_ELLIPTIC_PATH = str(PROJECT_ROOT / "data" / "elliptic")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "htgnn_latest.pt")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

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
    model_path: str = DEFAULT_MODEL_PATH
    model_threshold: float = 0.5
    batch_size: int = 256
    device: str = "cpu"  # "cuda" if GPU available

    # Fraud thresholds
    fraud_threshold_high: float = 0.80
    fraud_threshold_medium: float = 0.50
    fraud_threshold_low: float = 0.25

    # Datasets
    data_dir: str = DEFAULT_DATA_DIR
    ieee_cis_path: str = DEFAULT_IEEE_CIS_PATH
    paysim_path: str = DEFAULT_PAYSIM_PATH
    elliptic_path: str = DEFAULT_ELLIPTIC_PATH

    # WebSocket
    ws_ping_interval: int = 30

@lru_cache()
def get_settings() -> Settings:
    return Settings()
