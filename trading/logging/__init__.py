"""
Centralized Event Logging System

Comprehensive logging of all trading events for:
- Audit trails
- Debugging and analysis
- Compliance and reporting
- Performance monitoring
- Event replay

Logged Events:
- Tick events: Market data updates
- Trade events: Market trades
- Order events: Order placement
- Fill events: Order fills
- Cancel events: Order cancellations
- Touch events: Price level touches

Usage:
    from trading.logging import EventLogger, get_event_logger

    logger = get_event_logger()
    await logger.log_order(order)
    await logger.log_fill(fill)
"""

from .event_logger import (
    EventLogger,
    EventType,
    LoggedEvent,
    get_event_logger,
    set_event_logger,
)

from .storage import (
    LogStorage,
    FileLogStorage,
    DatabaseLogStorage,
    MemoryLogStorage,
)

__all__ = [
    "EventLogger",
    "EventType",
    "LoggedEvent",
    "get_event_logger",
    "set_event_logger",
    "LogStorage",
    "FileLogStorage",
    "DatabaseLogStorage",
    "MemoryLogStorage",
]
