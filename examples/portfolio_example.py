"""
Example: Portfolio Manager Usage

Demonstrates how to:
1. Track positions in real-time
2. Calculate P&L automatically
3. Monitor portfolio value
4. Get performance metrics

The Portfolio Manager updates automatically via events.
"""
import asyncio
from decimal import Decimal
from loguru import logger

from trading.events import InMemoryEventBus, TickEvent
from trading.events.order import OrderSide
from trading.exchanges import SimulatedExchange, OrderType, TimeInForce
from trading.portfolio import PortfolioManager
from trading.oms import OrderManager, OrderRouter, create_default_risk_checker


async def portfolio_tracking_example():
    """Basic portfolio tracking example"""

    logger.info("=== Portfolio Tracking Example ===\n")

    # Create components
    event_bus = InMemoryEventBus()
    initial_capital = Decimal("100000")

    # Create portfolio manager
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=initial_capital,
        update_interval=1.0
    )

    await portfolio.start()

    # Create exchange and OMS for trading
    exchange = SimulatedExchange(
        fill_model="aggressive",
        slippage_model="none",
        initial_balance=initial_capital
    )
    await exchange.connect()

    risk_checker = create_default_risk_checker()
    order_router = OrderRouter(exchanges={"simulated": exchange})

    oms = OrderManager(
        event_bus=event_bus,
        risk_checker=risk_checker,
        order_router=order_router,
        portfolio_manager=portfolio
    )
    await oms.start()

    logger.info(f"Initial capital: ${initial_capital:,.2f}\n")

    # Execute some trades
    logger.info("--- Executing Trades ---\n")

    # Trade 1: Buy BTC
    logger.info("Trade 1: Buy 0.5 BTC @ $50,000")
    await oms.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.5"),
        metadata={"exchange": "simulated"}
    )
    await asyncio.sleep(0.2)

    # Simulate price movement
    market_data = exchange.get_market_data_connector()
    tick = TickEvent(
        exchange="simulated",
        symbol="BTC/USDT",
        raw_symbol="BTC/USDT",
        bid=Decimal("52000"),
        ask=Decimal("52001"),
        last=Decimal("52000"),
        exchange_timestamp=datetime.utcnow()
    )
    await market_data._handle_event(tick)
    await asyncio.sleep(0.2)

    # Trade 2: Buy ETH
    logger.info("Trade 2: Buy 2 ETH @ $3,000")
    await oms.submit_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("2.0"),
        metadata={"exchange": "simulated"}
    )
    await asyncio.sleep(0.2)

    # Simulate ETH price movement
    tick = TickEvent(
        exchange="simulated",
        symbol="ETH/USDT",
        raw_symbol="ETH/USDT",
        bid=Decimal("3100"),
        ask=Decimal("3101"),
        last=Decimal("3100"),
        exchange_timestamp=datetime.utcnow()
    )
    await market_data._handle_event(tick)
    await asyncio.sleep(0.2)

    # Query portfolio state
    logger.info("\n--- Portfolio State ---\n")

    positions = await portfolio.get_all_positions()
    logger.info(f"Open positions: {len(positions)}")

    for symbol, position in positions.items():
        logger.info(
            f"{symbol}: {position.quantity} @ ${position.average_entry_price:,.2f} | "
            f"Unrealized P&L: ${position.unrealized_pnl:,.2f}"
        )

    account = await portfolio.get_account_state()
    logger.info(f"\nAccount:")
    logger.info(f"  Balance: ${account['balance']:,.2f}")
    logger.info(f"  Equity: ${account['equity']:,.2f}")

    # Trade 3: Close BTC position
    logger.info("\nTrade 3: Sell 0.5 BTC @ $52,000")
    await oms.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.5"),
        metadata={"exchange": "simulated"}
    )
    await asyncio.sleep(0.2)

    # Get metrics
    logger.info("\n--- Performance Metrics ---\n")

    metrics = await portfolio.get_metrics()

    logger.info(f"Realized P&L: ${metrics.realized_pnl:,.2f}")
    logger.info(f"Unrealized P&L: ${metrics.unrealized_pnl:,.2f}")
    logger.info(f"Total P&L: ${metrics.total_pnl:,.2f}")
    logger.info(f"Return: {metrics.return_pct:.2f}%")
    logger.info(f"\nWin Rate: {metrics.win_rate:.1f}%")
    logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")
    logger.info(f"Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2f}%)")

    # Cleanup
    await oms.stop()
    await portfolio.stop()
    await exchange.disconnect()
    await event_bus.close()

    logger.info("\nExample complete!")


from datetime import datetime


async def real_time_updates_example():
    """Demonstrate real-time portfolio updates"""

    logger.info("\n=== Real-Time Portfolio Updates ===\n")

    # Create components
    event_bus = InMemoryEventBus()
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=Decimal("100000"),
        update_interval=0.5  # Update every 0.5s
    )

    # Subscribe to portfolio value events
    async def handle_portfolio_value(event):
        logger.info(
            f"Portfolio Update | "
            f"Value: ${event.total_value:,.2f} | "
            f"P&L: ${event.unrealized_pnl + event.realized_pnl:,.2f}"
        )

    await event_bus.subscribe("portfolio_value", handle_portfolio_value)

    await portfolio.start()

    # Simulate trading activity
    # (In practice, this would come from actual trades)

    logger.info("Starting portfolio monitoring...\n")

    # Let it run for a few seconds
    await asyncio.sleep(3)

    # Cleanup
    await portfolio.stop()
    await event_bus.close()

    logger.info("\nMonitoring stopped!")


async def statistics_example():
    """Get comprehensive portfolio statistics"""

    logger.info("\n=== Portfolio Statistics ===\n")

    event_bus = InMemoryEventBus()
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=Decimal("100000")
    )

    await portfolio.start()

    # Simulate some trading...
    # (Add positions via events)

    # Get statistics
    stats = portfolio.get_statistics()

    logger.info("Portfolio Statistics:")
    for key, value in stats.items():
        if isinstance(value, Decimal):
            logger.info(f"  {key}: ${value:,.2f}")
        else:
            logger.info(f"  {key}: {value}")

    await portfolio.stop()
    await event_bus.close()


async def main():
    """Main entry point"""

    # Run examples
    await portfolio_tracking_example()

    await real_time_updates_example()

    await statistics_example()


if __name__ == "__main__":
    asyncio.run(main())
