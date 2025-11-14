"""
Event Bus implementations

Pub/sub event bus for distributing events to subscribers.

Implementations:
- InMemoryEventBus: Fast, in-process event bus
- RedisEventBus: Multi-process event bus using Redis pub/sub

Topic Pattern:
events.{event_type}.{exchange}.{symbol}

Examples:
- events.tick.binance.BTC/USDT
- events.order_filled.okex.*
- events.*.binance.*
"""
from abc import ABC, abstractmethod
from typing import Callable, Awaitable, Optional, Any
from collections import defaultdict
import asyncio
import fnmatch
from loguru import logger

from .base import BaseEvent


class IEventBus(ABC):
    """Abstract interface for event bus"""

    @abstractmethod
    async def publish(self, event: BaseEvent) -> None:
        """
        Publish an event to all subscribers

        Args:
            event: Event to publish
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[BaseEvent], Awaitable[None]],
        filter_func: Optional[Callable[[BaseEvent], bool]] = None
    ) -> str:
        """
        Subscribe to events

        Args:
            event_type: Type of event to subscribe to (supports wildcards)
            handler: Async function to handle events
            filter_func: Optional filter function

        Returns:
            Subscription ID (for unsubscribing)
        """
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from events

        Args:
            subscription_id: ID returned from subscribe()
        """
        pass

    @abstractmethod
    async def publish_batch(self, events: list[BaseEvent]) -> None:
        """
        Publish multiple events efficiently

        Args:
            events: List of events to publish
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the event bus and cleanup resources"""
        pass


class Subscription:
    """Represents a subscription to events"""

    def __init__(
        self,
        subscription_id: str,
        event_type_pattern: str,
        handler: Callable[[BaseEvent], Awaitable[None]],
        filter_func: Optional[Callable[[BaseEvent], bool]] = None
    ):
        self.subscription_id = subscription_id
        self.event_type_pattern = event_type_pattern
        self.handler = handler
        self.filter_func = filter_func

    def matches(self, event: BaseEvent) -> bool:
        """Check if event matches this subscription"""
        # Check event type pattern
        if not fnmatch.fnmatch(event.event_type, self.event_type_pattern):
            return False

        # Apply additional filter if provided
        if self.filter_func and not self.filter_func(event):
            return False

        return True


class InMemoryEventBus(IEventBus):
    """
    In-memory event bus for single-process applications

    Fast and simple, but only works within a single process.
    Ideal for development, testing, and single-process production deployments.
    """

    def __init__(self):
        self.subscriptions: dict[str, Subscription] = {}
        self._subscription_counter = 0
        self._closed = False

        logger.info("Initialized InMemoryEventBus")

    async def publish(self, event: BaseEvent) -> None:
        """Publish event to all matching subscribers"""
        if self._closed:
            logger.warning("Attempted to publish to closed event bus")
            return

        # Find matching subscriptions
        matching_subs = [
            sub for sub in self.subscriptions.values()
            if sub.matches(event)
        ]

        if not matching_subs:
            logger.debug(f"No subscribers for event type: {event.event_type}")
            return

        # Call all handlers concurrently
        tasks = []
        for sub in matching_subs:
            task = asyncio.create_task(self._call_handler(sub, event))
            tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _call_handler(self, sub: Subscription, event: BaseEvent) -> None:
        """Call a handler with error handling"""
        try:
            await sub.handler(event)
        except Exception as e:
            logger.error(
                f"Error in event handler for {event.event_type}: {e}",
                exc_info=True
            )
            # Don't re-raise - we don't want one bad handler to break the bus

    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[BaseEvent], Awaitable[None]],
        filter_func: Optional[Callable[[BaseEvent], bool]] = None
    ) -> str:
        """Subscribe to events"""
        self._subscription_counter += 1
        subscription_id = f"sub_{self._subscription_counter}"

        subscription = Subscription(
            subscription_id=subscription_id,
            event_type_pattern=event_type,
            handler=handler,
            filter_func=filter_func
        )

        self.subscriptions[subscription_id] = subscription

        logger.debug(f"Added subscription {subscription_id} for pattern: {event_type}")

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            logger.debug(f"Removed subscription {subscription_id}")
        else:
            logger.warning(f"Subscription {subscription_id} not found")

    async def publish_batch(self, events: list[BaseEvent]) -> None:
        """Publish multiple events"""
        tasks = [self.publish(event) for event in events]
        await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the event bus"""
        self._closed = True
        self.subscriptions.clear()
        logger.info("Closed InMemoryEventBus")

    def get_statistics(self) -> dict[str, Any]:
        """Get event bus statistics"""
        return {
            "num_subscriptions": len(self.subscriptions),
            "closed": self._closed
        }


class RedisEventBus(IEventBus):
    """
    Redis-based event bus for multi-process applications

    Uses Redis pub/sub for event distribution across processes.
    Slower than in-memory bus but supports multiple processes/servers.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "events"
    ):
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.subscriptions: dict[str, Subscription] = {}
        self._subscription_counter = 0
        self._closed = False
        self._redis = None
        self._pubsub = None
        self._listen_task: Optional[asyncio.Task] = None

        logger.info(f"Initialized RedisEventBus: {redis_url}")

    async def connect(self) -> None:
        """Connect to Redis"""
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(
            self.redis_url,
            decode_responses=False  # We'll handle serialization
        )

        self._pubsub = self._redis.pubsub()

        # Start listening task
        self._listen_task = asyncio.create_task(self._listen_loop())

        logger.info("Connected to Redis for event bus")

    async def _listen_loop(self) -> None:
        """Listen for messages from Redis"""
        while not self._closed:
            try:
                async for message in self._pubsub.listen():
                    if message["type"] == "message":
                        await self._handle_redis_message(message)
            except Exception as e:
                if not self._closed:
                    logger.error(f"Error in Redis listen loop: {e}")
                    await asyncio.sleep(1)  # Backoff before retry

    async def _handle_redis_message(self, message: dict) -> None:
        """Handle message received from Redis"""
        try:
            import msgpack

            # Deserialize event
            event_data = msgpack.unpackb(message["data"], raw=False)
            event = self._deserialize_event(event_data)

            # Find matching subscriptions
            matching_subs = [
                sub for sub in self.subscriptions.values()
                if sub.matches(event)
            ]

            # Call handlers
            for sub in matching_subs:
                asyncio.create_task(self._call_handler(sub, event))

        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")

    def _deserialize_event(self, event_data: dict) -> BaseEvent:
        """Deserialize event from dict"""
        from . import (
            TickEvent, TradeEvent, OrderBookEvent, BarEvent,
            OrderSubmittedEvent, OrderFilledEvent, # etc...
        )

        # Map event types to classes
        event_type = event_data.get("event_type")
        # TODO: Complete mapping for all event types
        # For now, return base event
        return BaseEvent(**event_data)

    async def _call_handler(self, sub: Subscription, event: BaseEvent) -> None:
        """Call handler with error handling"""
        try:
            await sub.handler(event)
        except Exception as e:
            logger.error(
                f"Error in event handler for {event.event_type}: {e}",
                exc_info=True
            )

    async def publish(self, event: BaseEvent) -> None:
        """Publish event to Redis"""
        if self._closed or not self._redis:
            logger.warning("Attempted to publish to closed/unconnected event bus")
            return

        try:
            import msgpack

            # Serialize event
            event_dict = event.dict()
            event_bytes = msgpack.packb(event_dict, use_bin_type=True)

            # Publish to Redis channel
            channel = f"{self.channel_prefix}.{event.event_type}"
            await self._redis.publish(channel, event_bytes)

        except Exception as e:
            logger.error(f"Error publishing event to Redis: {e}")

    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[BaseEvent], Awaitable[None]],
        filter_func: Optional[Callable[[BaseEvent], bool]] = None
    ) -> str:
        """Subscribe to events"""
        self._subscription_counter += 1
        subscription_id = f"sub_{self._subscription_counter}"

        subscription = Subscription(
            subscription_id=subscription_id,
            event_type_pattern=event_type,
            handler=handler,
            filter_func=filter_func
        )

        self.subscriptions[subscription_id] = subscription

        # Subscribe to Redis channel pattern
        if self._pubsub:
            channel_pattern = f"{self.channel_prefix}.{event_type}"
            if "*" in event_type:
                # Use pattern subscription for wildcards
                await self._pubsub.psubscribe(channel_pattern)
            else:
                await self._pubsub.subscribe(channel_pattern)

        logger.debug(f"Added subscription {subscription_id} for pattern: {event_type}")

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""
        if subscription_id in self.subscriptions:
            # TODO: Unsubscribe from Redis channel
            del self.subscriptions[subscription_id]
            logger.debug(f"Removed subscription {subscription_id}")

    async def publish_batch(self, events: list[BaseEvent]) -> None:
        """Publish multiple events"""
        tasks = [self.publish(event) for event in events]
        await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the event bus"""
        self._closed = True

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        logger.info("Closed RedisEventBus")


# Factory function
def create_event_bus(bus_type: str = "memory", **kwargs: Any) -> IEventBus:
    """
    Factory function to create event bus

    Args:
        bus_type: 'memory' or 'redis'
        **kwargs: Additional arguments for bus

    Returns:
        Event bus instance
    """
    if bus_type == "memory":
        return InMemoryEventBus()
    elif bus_type == "redis":
        return RedisEventBus(**kwargs)
    else:
        raise ValueError(f"Unknown bus type: {bus_type}")
