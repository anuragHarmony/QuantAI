"""
Event System Example

Demonstrates how to use the trading event system.

Run with:
    python examples/event_system_example.py
"""
import asyncio
from decimal import Decimal
from datetime import datetime

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.events import (
    TickEvent,
    TradeEvent,
    OrderSubmittedEvent,
    OrderFilledEvent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from trading.events.bus import InMemoryEventBus
from trading.events.filters import (
    symbol_filter,
    exchange_filter,
    and_filter,
)


async def main():
    """Run example"""
    print("=== Event System Example ===\n")

    # Create event bus
    bus = InMemoryEventBus()

    # Example 1: Subscribe to all ticks
    print("1. Subscribe to ALL tick events:")

    async def handle_all_ticks(event: TickEvent):
        print(f"   📈 Tick: {event.symbol} @ {event.last}")

    await bus.subscribe("tick", handle_all_ticks)

    # Example 2: Subscribe to BTC ticks only (using filter)
    print("\n2. Subscribe to BTC/USDT ticks only:")

    async def handle_btc_ticks(event: TickEvent):
        print(f"   ₿ BTC Tick: Bid={event.bid}, Ask={event.ask}")

    await bus.subscribe(
        "tick",
        handle_btc_ticks,
        filter_func=symbol_filter("BTC/USDT")
    )

    # Example 3: Subscribe to Binance ticks only
    print("\n3. Subscribe to Binance ticks only:")

    async def handle_binance_ticks(event: TickEvent):
        print(f"   🔶 Binance: {event.symbol} @ {event.last}")

    await bus.subscribe(
        "tick",
        handle_binance_ticks,
        filter_func=exchange_filter("binance")
    )

    # Example 4: Subscribe to order events
    print("\n4. Subscribe to order filled events:")

    async def handle_order_filled(event: OrderFilledEvent):
        print(f"   ✅ Order Filled: {event.client_order_id}")
        print(f"      Symbol: {event.symbol}")
        print(f"      Side: {event.side}")
        print(f"      Quantity: {event.total_filled}")
        print(f"      Price: {event.average_fill_price}")

    await bus.subscribe("order_filled", handle_order_filled)

    # Publish some events
    print("\n\n=== Publishing Events ===\n")

    # Publish BTC tick from Binance
    print("Publishing BTC/USDT tick from Binance...")
    btc_tick = TickEvent(
        exchange="binance",
        symbol="BTC/USDT",
        raw_symbol="BTCUSDT",
        bid=Decimal("50000.5"),
        ask=Decimal("50000.6"),
        last=Decimal("50000.55"),
        exchange_timestamp=datetime.utcnow()
    )
    await bus.publish(btc_tick)

    await asyncio.sleep(0.1)  # Let handlers run

    # Publish ETH tick from Binance
    print("\nPublishing ETH/USDT tick from Binance...")
    eth_tick = TickEvent(
        exchange="binance",
        symbol="ETH/USDT",
        raw_symbol="ETHUSDT",
        bid=Decimal("3000.1"),
        ask=Decimal("3000.2"),
        last=Decimal("3000.15"),
        exchange_timestamp=datetime.utcnow()
    )
    await bus.publish(eth_tick)

    await asyncio.sleep(0.1)

    # Publish BTC tick from OKEx
    print("\nPublishing BTC/USDT tick from OKEx...")
    btc_okex_tick = TickEvent(
        exchange="okex",
        symbol="BTC/USDT",
        raw_symbol="BTC-USDT",
        bid=Decimal("50001.0"),
        ask=Decimal("50001.1"),
        last=Decimal("50001.05"),
        exchange_timestamp=datetime.utcnow()
    )
    await bus.publish(btc_okex_tick)

    await asyncio.sleep(0.1)

    # Publish order filled event
    print("\nPublishing order filled event...")
    order_filled = OrderFilledEvent(
        client_order_id="order_123",
        exchange_order_id="binance_456",
        exchange="binance",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.001"),
        price=Decimal("50000"),
        total_filled=Decimal("0.001"),
        average_fill_price=Decimal("50000"),
        fill_timestamp=datetime.utcnow()
    )
    await bus.publish(order_filled)

    await asyncio.sleep(0.1)

    # Example 5: Batch publish
    print("\n\n=== Batch Publishing ===\n")
    print("Publishing 5 ticks in batch...")

    batch_ticks = [
        TickEvent(
            exchange="binance",
            symbol=f"SYMBOL{i}",
            raw_symbol=f"SYMBOL{i}",
            last=Decimal(f"{1000 + i}"),
            exchange_timestamp=datetime.utcnow()
        )
        for i in range(5)
    ]

    await bus.publish_batch(batch_ticks)
    await asyncio.sleep(0.1)

    # Get statistics
    print("\n\n=== Event Bus Statistics ===")
    stats = bus.get_statistics()
    print(f"Active subscriptions: {stats['num_subscriptions']}")
    print(f"Bus closed: {stats['closed']}")

    # Cleanup
    await bus.close()
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
