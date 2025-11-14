"""
Example: Order Management System Usage

Demonstrates how to:
1. Set up OMS with risk checks
2. Submit orders through OMS
3. Track order lifecycle via events
4. Handle risk violations

The OMS provides:
- Pre-trade risk validation
- Smart order routing
- Order state tracking
- Event-driven updates
"""
import asyncio
from decimal import Decimal
from loguru import logger

from trading.events import InMemoryEventBus, OrderFilledEvent, OrderRejectedEvent
from trading.events.order import OrderSide, OrderType, TimeInForce
from trading.exchanges import SimulatedExchange, Order
from trading.oms import (
    OrderManager,
    RiskChecker,
    create_default_risk_checker,
    OrderRouter,
)


async def handle_order_filled(event: OrderFilledEvent) -> None:
    """Handle order filled events"""
    logger.info(
        f"✓ ORDER FILLED: {event.order_id} | "
        f"{event.side.value} {event.quantity} {event.symbol}"
    )


async def handle_order_rejected(event: OrderRejectedEvent) -> None:
    """Handle order rejected events"""
    logger.warning(
        f"✗ ORDER REJECTED: {event.order_id} | "
        f"{event.symbol} - {getattr(event, 'reason', 'Unknown reason')}"
    )


async def basic_oms_example():
    """Basic OMS usage example"""

    logger.info("=== Basic OMS Example ===\n")

    # 1. Create components
    event_bus = InMemoryEventBus()

    # Create simulated exchange
    exchange = SimulatedExchange(
        fill_model="aggressive",  # Fill all orders immediately
        slippage_model="none",    # No slippage for testing
        initial_balance=Decimal("100000")
    )
    await exchange.connect()

    # Create risk checker with limits
    risk_checker = create_default_risk_checker(
        max_position_size=Decimal("10"),      # Max 10 units per position
        max_order_size=Decimal("5"),          # Max 5 units per order
        max_daily_loss=Decimal("1000"),       # Max $1000 loss per day
        max_concentration_pct=Decimal("0.3"), # Max 30% in single position
        min_margin_ratio=Decimal("0.2")       # Min 20% margin
    )

    # Create order router
    order_router = OrderRouter(exchanges={"simulated": exchange})

    # Create OMS
    oms = OrderManager(
        event_bus=event_bus,
        risk_checker=risk_checker,
        order_router=order_router,
        portfolio_manager=None  # We'll add this in the next example
    )

    # Subscribe to order events
    await event_bus.subscribe("order_filled", handle_order_filled)
    await event_bus.subscribe("order_rejected", handle_order_rejected)

    # Start OMS
    await oms.start()

    logger.info("OMS initialized and started\n")

    # 2. Submit valid order
    logger.info("--- Submitting Valid Order ---")

    result = await oms.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        metadata={"exchange": "simulated"}
    )

    if result.success:
        logger.info(f"Order accepted: {result.order_id}\n")
    else:
        logger.error(f"Order failed: {result.message}\n")

    await asyncio.sleep(0.5)  # Let events process

    # 3. Submit order that violates risk checks
    logger.info("--- Submitting Order That Violates Risk ---")

    result = await oms.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("20.0"),  # Exceeds max_order_size and max_position_size
        metadata={"exchange": "simulated"}
    )

    if result.success:
        logger.info(f"Order accepted: {result.order_id}\n")
    else:
        logger.error(f"Order rejected: {result.message}\n")

    await asyncio.sleep(0.5)

    # 4. Check OMS statistics
    logger.info("--- OMS Statistics ---")
    stats = oms.get_statistics()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    # 5. Cleanup
    await oms.stop()
    await exchange.disconnect()
    await event_bus.close()

    logger.info("\nExample complete!")


async def risk_checks_example():
    """Demonstrate different risk check violations"""

    logger.info("\n=== Risk Checks Example ===\n")

    # Create components
    event_bus = InMemoryEventBus()
    exchange = SimulatedExchange()
    await exchange.connect()

    # Create strict risk checker
    risk_checker = create_default_risk_checker(
        max_position_size=Decimal("2"),
        max_order_size=Decimal("1"),
        max_daily_loss=Decimal("100"),
        max_concentration_pct=Decimal("0.2"),
        min_margin_ratio=Decimal("0.5")
    )

    order_router = OrderRouter(exchanges={"simulated": exchange})

    oms = OrderManager(
        event_bus=event_bus,
        risk_checker=risk_checker,
        order_router=order_router
    )

    await oms.start()

    # Test 1: Exceed order size limit
    logger.info("Test 1: Exceed order size limit")
    result = await oms.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("5.0"),  # > max_order_size
        metadata={"exchange": "simulated"}
    )
    logger.info(f"Result: {result.message}\n")

    # Test 2: Valid order within limits
    logger.info("Test 2: Valid order within limits")
    result = await oms.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.5"),  # Within limits
        metadata={"exchange": "simulated"}
    )
    logger.info(f"Result: {result.message}\n")

    await asyncio.sleep(0.5)

    # Test 3: Build up position
    logger.info("Test 3: Build up position")
    for i in range(3):
        result = await oms.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.8"),
            metadata={"exchange": "simulated"}
        )
        logger.info(f"Order {i+1}: {result.message}")
        await asyncio.sleep(0.1)

    logger.info()

    # Test 4: Close position
    logger.info("Test 4: Close position")
    result = await oms.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("2.0"),
        metadata={"exchange": "simulated"}
    )
    logger.info(f"Result: {result.message}\n")

    # Statistics
    logger.info("--- Final Statistics ---")
    stats = oms.get_statistics()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    # Cleanup
    await oms.stop()
    await exchange.disconnect()
    await event_bus.close()


async def order_lifecycle_example():
    """Track complete order lifecycle"""

    logger.info("\n=== Order Lifecycle Example ===\n")

    # Create components
    event_bus = InMemoryEventBus()
    exchange = SimulatedExchange()
    await exchange.connect()

    risk_checker = create_default_risk_checker()
    order_router = OrderRouter(exchanges={"simulated": exchange})

    oms = OrderManager(
        event_bus=event_bus,
        risk_checker=risk_checker,
        order_router=order_router
    )

    # Track all order events
    async def track_event(event) -> None:
        logger.info(
            f"EVENT: {event.event_type} | "
            f"{getattr(event, 'order_id', 'N/A')} | "
            f"{getattr(event, 'symbol', 'N/A')}"
        )

    await event_bus.subscribe("order_*", track_event)
    await oms.start()

    # Submit order
    logger.info("Submitting order...\n")

    result = await oms.submit_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        limit_price=Decimal("3000"),
        metadata={"exchange": "simulated"}
    )

    logger.info(f"\nOrder submitted: {result.order_id}\n")

    # Let events process
    await asyncio.sleep(0.5)

    # Query order status
    order = await oms.get_order(result.order_id)
    if order:
        logger.info(f"\nOrder Status:")
        logger.info(f"  Status: {order.status.value}")
        logger.info(f"  Symbol: {order.symbol}")
        logger.info(f"  Side: {order.side.value}")
        logger.info(f"  Quantity: {order.quantity}")
        logger.info(f"  Filled: {order.filled_quantity}")

    # Cancel order
    logger.info(f"\nCancelling order {result.order_id}...\n")
    cancel_result = await oms.cancel_order(result.order_id)
    logger.info(f"Cancel result: {cancel_result.message}")

    await asyncio.sleep(0.5)

    # Final state
    order = await oms.get_order(result.order_id)
    if order:
        logger.info(f"\nFinal order status: {order.status.value}")

    # Cleanup
    await oms.stop()
    await exchange.disconnect()
    await event_bus.close()


async def main():
    """Main entry point"""

    # Run different examples
    await basic_oms_example()

    await risk_checks_example()

    await order_lifecycle_example()


if __name__ == "__main__":
    asyncio.run(main())
