"""
Centralized Event Logger

Logs all trading events to storage backends for audit, analysis, and replay.
"""
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger
import json

from ..events import (
    TickEvent,
    TradeEvent,
    FillEvent,
    OrderEvent,
    CancelEvent,
)


class EventType(Enum):
    """Types of logged events"""
    TICK = "tick"
    TRADE = "trade"
    ORDER = "order"
    FILL = "fill"
    CANCEL = "cancel"
    SENT_CANCEL = "sent_cancel"
    TOUCH = "touch"  # Price level touch
    SIGNAL = "signal"  # Strategy signal
    ERROR = "error"  # Error event


@dataclass
class LoggedEvent:
    """
    Logged event container

    Stores all event data with metadata for storage and retrieval.
    """
    # Event metadata
    event_id: str  # Unique event ID
    event_type: EventType
    timestamp: datetime
    strategy_id: Optional[str] = None

    # Event data
    exchange: str = ""
    symbol: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    # Additional context
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "data": self._serialize_data(self.data),
            "session_id": self.session_id,
            "tags": self.tags,
        }
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @staticmethod
    def _serialize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data, converting Decimal to float"""
        result = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = LoggedEvent._serialize_data(value)
            else:
                result[key] = value
        return result


class EventLogger:
    """
    Centralized event logger

    Logs all trading events to one or more storage backends.
    """

    def __init__(
        self,
        storage_backends: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize event logger

        Args:
            storage_backends: List of storage backends (FileLogStorage, DatabaseLogStorage, etc.)
            session_id: Optional session ID to tag all events
            enabled: Whether logging is enabled
        """
        self.storage_backends = storage_backends or []
        self.session_id = session_id or self._generate_session_id()
        self.enabled = enabled

        # Event counter for unique IDs
        self._event_counter = 0

        # Statistics
        self.events_logged = 0
        self.events_by_type: Dict[EventType, int] = {}

        if not enabled:
            logger.warning("Event logging is DISABLED")
        else:
            logger.info(
                f"Event logger initialized: "
                f"Session ID: {self.session_id}, "
                f"Backends: {len(self.storage_backends)}"
            )

    async def log_tick(self, tick: TickEvent, strategy_id: Optional[str] = None) -> None:
        """Log tick event"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.TICK,
            timestamp=tick.exchange_timestamp or datetime.now(),
            strategy_id=strategy_id,
            exchange=tick.exchange,
            symbol=tick.symbol,
            data={
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "raw_symbol": tick.raw_symbol,
            },
            session_id=self.session_id,
            tags=["market_data"]
        )

        await self._store_event(event)

    async def log_trade(self, trade: TradeEvent, strategy_id: Optional[str] = None) -> None:
        """Log trade event (market trade, not our trade)"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.TRADE,
            timestamp=trade.timestamp,
            strategy_id=strategy_id,
            exchange=trade.exchange,
            symbol=trade.symbol,
            data={
                "trade_id": trade.trade_id,
                "price": trade.price,
                "quantity": trade.quantity,
                "side": trade.side,
            },
            session_id=self.session_id,
            tags=["market_data", "trade"]
        )

        await self._store_event(event)

    async def log_order(
        self,
        order_id: str,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        strategy_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log order placement"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.ORDER,
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            exchange=exchange,
            symbol=symbol,
            data={
                "order_id": order_id,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                **kwargs
            },
            session_id=self.session_id,
            tags=["order", "execution"]
        )

        await self._store_event(event)

        logger.debug(f"Logged ORDER: {order_id} {side} {quantity} {symbol} @ {exchange}")

    async def log_fill(self, fill: FillEvent, strategy_id: Optional[str] = None) -> None:
        """Log order fill"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.FILL,
            timestamp=fill.timestamp,
            strategy_id=strategy_id,
            exchange=fill.exchange,
            symbol=fill.symbol,
            data={
                "order_id": fill.order_id,
                "fill_id": fill.fill_id,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "fee": fill.fee,
                "fee_currency": fill.fee_currency,
            },
            session_id=self.session_id,
            tags=["fill", "execution"]
        )

        await self._store_event(event)

        logger.debug(f"Logged FILL: {fill.fill_id} {fill.side} {fill.quantity} {fill.symbol} @ {fill.price}")

    async def log_cancel(
        self,
        order_id: str,
        exchange: str,
        symbol: str,
        reason: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> None:
        """Log order cancellation (confirmed)"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.CANCEL,
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            exchange=exchange,
            symbol=symbol,
            data={
                "order_id": order_id,
                "reason": reason,
            },
            session_id=self.session_id,
            tags=["cancel", "execution"]
        )

        await self._store_event(event)

        logger.debug(f"Logged CANCEL: {order_id} {symbol} @ {exchange}")

    async def log_sent_cancel(
        self,
        order_id: str,
        exchange: str,
        symbol: str,
        strategy_id: Optional[str] = None
    ) -> None:
        """Log cancel request sent (not yet confirmed)"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.SENT_CANCEL,
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            exchange=exchange,
            symbol=symbol,
            data={
                "order_id": order_id,
            },
            session_id=self.session_id,
            tags=["cancel", "sent", "execution"]
        )

        await self._store_event(event)

        logger.debug(f"Logged SENT_CANCEL: {order_id} {symbol} @ {exchange}")

    async def log_touch(
        self,
        exchange: str,
        symbol: str,
        level: Decimal,
        side: str,  # "bid" or "ask"
        strategy_id: Optional[str] = None
    ) -> None:
        """Log price level touch"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.TOUCH,
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            exchange=exchange,
            symbol=symbol,
            data={
                "level": level,
                "side": side,
            },
            session_id=self.session_id,
            tags=["touch", "market_data"]
        )

        await self._store_event(event)

    async def log_signal(
        self,
        exchange: str,
        symbol: str,
        signal_type: str,
        direction: str,
        strength: Optional[Decimal] = None,
        strategy_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log strategy signal"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.SIGNAL,
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            exchange=exchange,
            symbol=symbol,
            data={
                "signal_type": signal_type,
                "direction": direction,
                "strength": strength,
                **kwargs
            },
            session_id=self.session_id,
            tags=["signal", "strategy"]
        )

        await self._store_event(event)

    async def log_error(
        self,
        error_type: str,
        message: str,
        exchange: str = "",
        symbol: str = "",
        strategy_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log error event"""
        if not self.enabled:
            return

        event = LoggedEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.ERROR,
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            exchange=exchange,
            symbol=symbol,
            data={
                "error_type": error_type,
                "message": message,
                **kwargs
            },
            session_id=self.session_id,
            tags=["error"]
        )

        await self._store_event(event)

        logger.error(f"Logged ERROR: {error_type} - {message}")

    async def _store_event(self, event: LoggedEvent) -> None:
        """Store event to all backends"""
        self.events_logged += 1
        self.events_by_type[event.event_type] = self.events_by_type.get(event.event_type, 0) + 1

        # Store to all backends
        for backend in self.storage_backends:
            try:
                await backend.store(event)
            except Exception as e:
                logger.error(f"Failed to store event to backend {backend}: {e}")

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        self._event_counter += 1
        return f"{self.session_id}_{self._event_counter:08d}"

    @staticmethod
    def _generate_session_id() -> str:
        """Generate session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "session_id": self.session_id,
            "enabled": self.enabled,
            "events_logged": self.events_logged,
            "events_by_type": {k.value: v for k, v in self.events_by_type.items()},
            "backends": len(self.storage_backends),
        }

    async def close(self) -> None:
        """Close all storage backends"""
        logger.info("Closing event logger...")

        for backend in self.storage_backends:
            try:
                if hasattr(backend, 'close'):
                    await backend.close()
            except Exception as e:
                logger.error(f"Error closing backend {backend}: {e}")

        logger.info(f"Event logger closed. Total events logged: {self.events_logged}")


# Global event logger instance
_global_event_logger: Optional[EventLogger] = None


def get_event_logger() -> Optional[EventLogger]:
    """Get global event logger instance"""
    return _global_event_logger


def set_event_logger(logger: EventLogger) -> None:
    """Set global event logger instance"""
    global _global_event_logger
    _global_event_logger = logger


logger.info("Event logger module loaded")
