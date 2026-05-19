import json
import logging
from typing import Optional
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis get error for {key}: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = 180) -> bool:
        try:
            await self.redis.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis set error for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error for {key}: {e}")
            return False

    async def close(self):
        await self.redis.close()
