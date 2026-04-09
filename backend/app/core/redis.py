import json
import redis.asyncio as aioredis
from typing import Any, Optional
import logging
from app.core.config import get_settings

settings = get_settings()
log = logging.getLogger(__name__)

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def cache_set(key: str, value: Any, ttl: int = None) -> None:
    try:
        r = await get_redis()
        await r.set(key, json.dumps(value), ex=ttl or settings.cache_ttl)
    except Exception as e:
        log.warning(f"cache_set failed for key={key}: {e}")


async def cache_get(key: str) -> Optional[Any]:
    try:
        r = await get_redis()
        val = await r.get(key)
        return json.loads(val) if val else None
    except Exception as e:
        log.warning(f"cache_get failed for key={key}: {e}")
        return None


async def cache_delete(key: str) -> None:
    try:
        r = await get_redis()
        await r.delete(key)
    except Exception as e:
        log.warning(f"cache_delete failed for key={key}: {e}")


async def publish_score(channel: str, data: dict) -> None:
    """Publish fraud score to WebSocket subscribers."""
    try:
        r = await get_redis()
        await r.publish(channel, json.dumps(data))
    except Exception as e:
        log.warning(f"publish_score failed for channel={channel}: {e}")
