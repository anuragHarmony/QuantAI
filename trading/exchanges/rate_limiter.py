"""
Rate limiter implementations

Token bucket algorithm for rate limiting API requests.
Prevents exceeding exchange rate limits.
"""
import asyncio
import time
from typing import Optional, Dict
from dataclasses import dataclass
from loguru import logger

from .base import IRateLimiter


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_period: int  # Max requests
    period_seconds: float  # Time period
    weight_per_request: int = 1  # Default weight


class TokenBucket:
    """
    Token bucket for single endpoint

    Tokens are added at constant rate.
    Each request consumes tokens.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
    ):
        """
        Initialize token bucket

        Args:
            capacity: Maximum tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, weight: int = 1) -> None:
        """
        Consume tokens (blocks if insufficient)

        Args:
            weight: Number of tokens to consume
        """
        async with self._lock:
            # Refill tokens based on time passed
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + (time_passed * self.refill_rate)
            )
            self.last_update = now

            # If not enough tokens, wait
            if self.tokens < weight:
                wait_time = (weight - self.tokens) / self.refill_rate
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {weight} tokens")
                await asyncio.sleep(wait_time)

                # Refill again after waiting
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.capacity,
                    self.tokens + (time_passed * self.refill_rate)
                )
                self.last_update = now

            # Consume tokens
            self.tokens -= weight

    def get_remaining(self) -> int:
        """Get remaining tokens"""
        # Update tokens first
        now = time.time()
        time_passed = now - self.last_update
        current_tokens = min(
            self.capacity,
            self.tokens + (time_passed * self.refill_rate)
        )
        return int(current_tokens)


class TokenBucketRateLimiter(IRateLimiter):
    """
    Token bucket rate limiter

    Supports multiple endpoints with different limits.
    """

    def __init__(
        self,
        limits: Optional[Dict[str, RateLimit]] = None
    ):
        """
        Initialize rate limiter

        Args:
            limits: Dict mapping endpoint name to RateLimit
        """
        self.limits = limits or {}
        self.buckets: Dict[str, TokenBucket] = {}

        # Create buckets for each limit
        for endpoint, limit in self.limits.items():
            refill_rate = limit.requests_per_period / limit.period_seconds
            self.buckets[endpoint] = TokenBucket(
                capacity=limit.requests_per_period,
                refill_rate=refill_rate
            )

        logger.info(f"Initialized rate limiter with {len(self.limits)} endpoints")

    async def acquire(self, endpoint: str, weight: int = 1) -> None:
        """
        Acquire permission to make request

        Args:
            endpoint: Endpoint identifier
            weight: Request weight
        """
        if endpoint not in self.buckets:
            # No limit configured for this endpoint
            return

        await self.buckets[endpoint].consume(weight)

    def get_remaining(self, endpoint: str) -> int:
        """Get remaining requests for endpoint"""
        if endpoint not in self.buckets:
            return 999999  # Unlimited

        return self.buckets[endpoint].get_remaining()

    def reset_limits(self) -> None:
        """Reset all rate limits (for testing)"""
        for endpoint, limit in self.limits.items():
            refill_rate = limit.requests_per_period / limit.period_seconds
            self.buckets[endpoint] = TokenBucket(
                capacity=limit.requests_per_period,
                refill_rate=refill_rate
            )

        logger.info("Reset all rate limits")

    def get_statistics(self) -> Dict[str, Dict[str, any]]:
        """Get rate limit statistics"""
        stats = {}
        for endpoint, bucket in self.buckets.items():
            stats[endpoint] = {
                "remaining": bucket.get_remaining(),
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate
            }
        return stats


# Predefined limits for common exchanges

BINANCE_LIMITS = {
    "orders": RateLimit(10, 1.0),  # 10 orders per second
    "general": RateLimit(1200, 60.0),  # 1200 requests per minute (weight-based)
}

OKEX_LIMITS = {
    "orders": RateLimit(20, 2.0),  # 20 requests per 2 seconds
    "general": RateLimit(20, 2.0),
}

COINBASE_LIMITS = {
    "private": RateLimit(15, 1.0),  # 15 requests per second for private endpoints
    "public": RateLimit(10, 1.0),  # 10 requests per second for public
}

KRAKEN_LIMITS = {
    "private": RateLimit(15, 1.0),  # Kraken uses tier system, this is simplified
    "public": RateLimit(1, 1.0),  # 1 request per second for public
}


def create_rate_limiter(exchange: str) -> TokenBucketRateLimiter:
    """
    Factory function to create rate limiter for exchange

    Args:
        exchange: Exchange name

    Returns:
        Configured rate limiter
    """
    limits_map = {
        "binance": BINANCE_LIMITS,
        "okex": OKEX_LIMITS,
        "coinbase": COINBASE_LIMITS,
        "kraken": KRAKEN_LIMITS,
    }

    limits = limits_map.get(exchange.lower(), {})
    return TokenBucketRateLimiter(limits)
