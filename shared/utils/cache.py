"""
Caching utilities using Redis
"""
from typing import Any, Optional, TypeVar
import json
import pickle
from loguru import logger

from shared.models.base import ICacheProvider
from shared.config.settings import settings


T = TypeVar('T')


class RedisCache(ICacheProvider[str, Any]):
    """Redis-based cache implementation"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        prefix: str = "quantai:"
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix
        """
        import redis.asyncio as aioredis

        self.host = host or settings.redis.redis_host
        self.port = port or settings.redis.redis_port
        self.db = db or settings.redis.redis_db
        self.password = password or settings.redis.redis_password
        self.prefix = prefix

        self.redis = aioredis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=False  # We'll handle encoding
        )

        logger.info(f"Initialized Redis cache: {self.host}:{self.port}/{self.db}")

    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            redis_key = self._make_key(key)
            value = await self.redis.get(redis_key)

            if value:
                # Try to unpickle
                try:
                    return pickle.loads(value)
                except Exception:
                    # If unpickle fails, return as string
                    return value.decode('utf-8')

            return None

        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with optional TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        try:
            redis_key = self._make_key(key)

            # Pickle the value
            pickled_value = pickle.dumps(value)

            if ttl:
                await self.redis.setex(redis_key, ttl, pickled_value)
            else:
                await self.redis.set(redis_key, pickled_value)

            logger.debug(f"Cached key: {key} (TTL: {ttl})")

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")

    async def delete(self, key: str) -> None:
        """
        Delete from cache

        Args:
            key: Cache key
        """
        try:
            redis_key = self._make_key(key)
            await self.redis.delete(redis_key)
            logger.debug(f"Deleted cache key: {key}")

        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")

    async def clear(self) -> None:
        """Clear entire cache (keys with prefix)"""
        try:
            pattern = f"{self.prefix}*"
            cursor = 0

            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis.delete(*keys)

                if cursor == 0:
                    break

            logger.warning(f"Cleared all cache keys with prefix: {self.prefix}")

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        try:
            redis_key = self._make_key(key)
            return bool(await self.redis.exists(redis_key))

        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter

        Args:
            key: Counter key
            amount: Amount to increment

        Returns:
            New value
        """
        try:
            redis_key = self._make_key(key)
            return await self.redis.incrby(redis_key, amount)

        except Exception as e:
            logger.error(f"Cache increment failed for key {key}: {e}")
            raise

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values at once

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        try:
            redis_keys = [self._make_key(k) for k in keys]
            values = await self.redis.mget(redis_keys)

            result = {}
            for i, key in enumerate(keys):
                if values[i]:
                    try:
                        result[key] = pickle.loads(values[i])
                    except Exception:
                        result[key] = values[i].decode('utf-8')

            return result

        except Exception as e:
            logger.error(f"Cache get_many failed: {e}")
            return {}

    async def set_many(self, mapping: dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Set multiple values at once

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live in seconds
        """
        try:
            pipe = self.redis.pipeline()

            for key, value in mapping.items():
                redis_key = self._make_key(key)
                pickled_value = pickle.dumps(value)

                if ttl:
                    pipe.setex(redis_key, ttl, pickled_value)
                else:
                    pipe.set(redis_key, pickled_value)

            await pipe.execute()

            logger.debug(f"Cached {len(mapping)} keys")

        except Exception as e:
            logger.error(f"Cache set_many failed: {e}")

    async def close(self) -> None:
        """Close Redis connection"""
        await self.redis.close()
        logger.info("Closed Redis connection")


class InMemoryCache(ICacheProvider[str, Any]):
    """Simple in-memory cache for testing"""

    def __init__(self):
        """Initialize in-memory cache"""
        self.cache: dict[str, Any] = {}
        logger.info("Initialized in-memory cache")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (TTL ignored for in-memory)"""
        self.cache[key] = value

    async def delete(self, key: str) -> None:
        """Delete from cache"""
        self.cache.pop(key, None)

    async def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()


class CachedFunction:
    """Decorator for caching function results"""

    def __init__(
        self,
        cache: ICacheProvider[str, Any],
        ttl: Optional[int] = 3600,
        key_prefix: str = "func:"
    ):
        """
        Initialize cached function decorator

        Args:
            cache: Cache provider
            ttl: Time-to-live in seconds
            key_prefix: Key prefix for cache keys
        """
        self.cache = cache
        self.ttl = ttl
        self.key_prefix = key_prefix

    def __call__(self, func):
        """Decorate function"""
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [self.key_prefix, func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_value = await self.cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            # Call function
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)

            # Store in cache
            await self.cache.set(cache_key, result, self.ttl)

            return result

        return wrapper


# Global cache instance
_global_cache: Optional[ICacheProvider[str, Any]] = None


def get_cache() -> ICacheProvider[str, Any]:
    """Get global cache instance"""
    global _global_cache

    if _global_cache is None:
        try:
            _global_cache = RedisCache()
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache, using in-memory: {e}")
            _global_cache = InMemoryCache()

    return _global_cache
