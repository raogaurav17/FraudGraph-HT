import json
import redis.asyncio as aioredis
from typing import Any, Optional
from app.core.config import get_settings

settings = get_settings()

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def cache_set(key: str, value: Any, ttl: int = None) -> None:
    r = await get_redis()
    await r.set(key, json.dumps(value), ex=ttl or settings.cache_ttl)


async def cache_get(key: str) -> Optional[Any]:
    r = await get_redis()
    val = await r.get(key)
    return json.loads(val) if val else None


async def cache_delete(key: str) -> None:
    r = await get_redis()
    await r.delete(key)


async def publish_score(channel: str, data: dict) -> None:
    """Publish fraud score to WebSocket subscribers."""
    r = await get_redis()
    await r.publish(channel, json.dumps(data))
