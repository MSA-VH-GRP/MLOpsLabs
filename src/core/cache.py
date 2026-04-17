"""Redis client wrapper for online feature store and general caching."""

import redis

from src.core.config import settings

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def ping() -> bool:
    try:
        return get_redis().ping()
    except redis.RedisError:
        return False
