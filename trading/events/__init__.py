"""
Trading Event System

Unified event-driven architecture for the trading platform.
All components communicate via typed events through an event bus.

Event Types:
- MarketData: Ticks, trades, order books, bars
- Order: Order lifecycle events
- Position: Position state changes
- Portfolio: Portfolio value and risk events
- System: Connection, error, heartbeat events

Usage:
    from trading.events import TickEvent, OrderFilledEvent, InMemoryEventBus

    bus = InMemoryEventBus()

    # Subscribe to events
    async def handle_tick(event: TickEvent):
        print(f"Tick: {event.symbol} @ {event.last}")

    await bus.subscribe("tick", handle_tick)

    # Publish events
    tick = TickEvent(
        exchange="binance",
        symbol="BTC/USDT",
        raw_symbol="BTCUSDT",
        bid=Decimal("50000"),
        ask=Decimal("50001"),
        last=Decimal("50000.5"),
        exchange_timestamp=datetime.utcnow()
    )
    await bus.publish(tick)
"""

# Base events
from .base import (
    BaseEvent,
    EventPriority,
    SystemEvent,
    ConnectionEvent,
    ErrorEvent,
    HeartbeatEvent,
)

# Market data events
from .market_data import (
    MarketDataEvent,
    TickEvent,
    TradeEvent,
    OrderBookEvent,
    OrderBookLevel,
    BarEvent,
    QuoteEvent,
)

# Order events
from .order import (
    OrderEvent,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderSubmittedEvent,
    OrderAcceptedEvent,
    OrderPartiallyFilledEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
    OrderRejectedEvent,
    OrderExpiredEvent,
    OrderModifiedEvent,
    OrderErrorEvent,
)

# Position and portfolio events
from .position import (
    PositionEvent,
    PositionSide,
    PositionOpenedEvent,
    PositionModifiedEvent,
    PositionClosedEvent,
    PortfolioEvent,
    PortfolioValueEvent,
    RiskLimitEvent,
    MarginCallEvent,
)

__all__ = [
    # Base
    "BaseEvent",
    "EventPriority",
    "SystemEvent",
    "ConnectionEvent",
    "ErrorEvent",
    "HeartbeatEvent",
    # Market Data
    "MarketDataEvent",
    "TickEvent",
    "TradeEvent",
    "OrderBookEvent",
    "OrderBookLevel",
    "BarEvent",
    "QuoteEvent",
    # Orders
    "OrderEvent",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "OrderSubmittedEvent",
    "OrderAcceptedEvent",
    "OrderPartiallyFilledEvent",
    "OrderFilledEvent",
    "OrderCancelledEvent",
    "OrderRejectedEvent",
    "OrderExpiredEvent",
    "OrderModifiedEvent",
    "OrderErrorEvent",
    # Positions & Portfolio
    "PositionEvent",
    "PositionSide",
    "PositionOpenedEvent",
    "PositionModifiedEvent",
    "PositionClosedEvent",
    "PortfolioEvent",
    "PortfolioValueEvent",
    "RiskLimitEvent",
    "MarginCallEvent",
]
