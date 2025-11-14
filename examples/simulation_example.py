"""
Example: Backtesting with Simulated Exchange

Demonstrates how to:
1. Load historical market data
2. Run a simple strategy in simulation
3. Analyze backtest results

This shows the complete simulation workflow including:
- Market data replay
- Order execution with realistic fills
- Position tracking
- P&L calculation
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from loguru import logger

from trading.events import TickEvent
from trading.events.persistence import EventReader
from trading.exchanges import (
    SimulatedExchange,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)


class SimpleMovingAverageCrossStrategy:
    """
    Simple MA cross strategy for demonstration

    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    """

    def __init__(self, exchange: SimulatedExchange, symbol: str):
        self.exchange = exchange
        self.symbol = symbol

        self.fast_window = 10
        self.slow_window = 30
        self.prices = []

        self.position_size = Decimal("0.1")
        self.in_position = False

    async def on_tick(self, tick: TickEvent) -> None:
        """Handle tick update"""
        if tick.symbol != self.symbol:
            return

        if not tick.last:
            return

        # Update price history
        self.prices.append(float(tick.last))

        # Need enough history for slow MA
        if len(self.prices) < self.slow_window:
            return

        # Calculate MAs
        fast_ma = sum(self.prices[-self.fast_window:]) / self.fast_window
        slow_ma = sum(self.prices[-self.slow_window:]) / self.slow_window

        prev_fast_ma = sum(self.prices[-self.fast_window-1:-1]) / self.fast_window
        prev_slow_ma = sum(self.prices[-self.slow_window-1:-1]) / self.slow_window

        # Check for crossover
        if not self.in_position and fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
            # Buy signal
            await self.buy()

        elif self.in_position and fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
            # Sell signal
            await self.sell()

    async def buy(self) -> None:
        """Open long position"""
        logger.info(f"BUY SIGNAL: {self.symbol}")

        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=self.position_size,
            time_in_force=TimeInForce.GTC,
        )

        execution = self.exchange.get_execution_connector()
        result = await execution.place_order(order)

        if result.success:
            self.in_position = True
            logger.info(f"Order placed: {result.order_id}")
        else:
            logger.error(f"Order failed: {result.message}")

    async def sell(self) -> None:
        """Close long position"""
        logger.info(f"SELL SIGNAL: {self.symbol}")

        order = Order(
            symbol=self.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=self.position_size,
            time_in_force=TimeInForce.GTC,
        )

        execution = self.exchange.get_execution_connector()
        result = await execution.place_order(order)

        if result.success:
            self.in_position = False
            logger.info(f"Order placed: {result.order_id}")
        else:
            logger.error(f"Order failed: {result.message}")


async def run_backtest_example():
    """Run complete backtest example"""

    logger.info("Starting backtest example")

    # 1. Set up simulation
    # NOTE: You need historical data recorded first
    # Use ParquetEventWriter to record live data, or generate synthetic data

    # For this example, we'll create a simulated exchange without replay
    # (In practice, you'd use EventReader with real historical data)
    exchange = SimulatedExchange(
        event_reader=None,  # Would be EventReader("./data/events") with real data
        fill_model="realistic",
        slippage_model="spread",
        initial_balance=Decimal("100000")
    )

    await exchange.connect()

    # 2. Create strategy
    symbol = "BTC/USDT"
    strategy = SimpleMovingAverageCrossStrategy(exchange, symbol)

    # 3. Subscribe to market data
    market_data = exchange.get_market_data_connector()
    await market_data.subscribe_ticks([symbol], strategy.on_tick)

    logger.info(f"Strategy initialized for {symbol}")

    # 4. If we had historical data, we'd replay it here:
    # start = datetime(2024, 1, 1)
    # end = datetime(2024, 1, 31)
    # await market_data.replay_events(start, end, [symbol])

    # 5. For demo, let's simulate some ticks manually
    logger.info("Simulating market data...")

    base_price = Decimal("50000")
    for i in range(100):
        # Create trending prices for MA cross
        if i < 40:
            price = base_price + Decimal(i * 10)  # Uptrend
        elif i < 60:
            price = base_price + Decimal(400) + Decimal((i - 40) * -5)  # Downtrend
        else:
            price = base_price + Decimal(300) + Decimal((i - 60) * 8)  # Uptrend

        tick = TickEvent(
            exchange="simulated",
            symbol=symbol,
            raw_symbol=symbol,
            bid=price - Decimal("1"),
            ask=price + Decimal("1"),
            last=price,
            exchange_timestamp=datetime.utcnow(),
        )

        # Process tick
        await strategy.on_tick(tick)

        # Also update market data connector cache
        market_data.latest_ticks[symbol] = tick

        # Small delay
        await asyncio.sleep(0.01)

    # 6. Get results
    execution = exchange.get_execution_connector()

    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)

    logger.info(f"Initial Balance: ${execution.initial_balance:,.2f}")
    logger.info(f"Final Balance: ${execution.balance:,.2f}")
    logger.info(f"Final Equity: ${execution.get_total_equity():,.2f}")

    total_pnl = execution.get_total_equity() - execution.initial_balance
    pnl_pct = (total_pnl / execution.initial_balance) * 100
    logger.info(f"Total P&L: ${total_pnl:,.2f} ({pnl_pct:,.2f}%)")

    logger.info(f"\nTotal Orders: {len(execution.orders)}")
    filled_orders = [o for o in execution.orders.values() if o.status.value == "filled"]
    logger.info(f"Filled Orders: {len(filled_orders)}")

    logger.info(f"\nOpen Positions: {len(execution.positions)}")
    for symbol, position in execution.positions.items():
        logger.info(
            f"  {symbol}: {position.quantity} @ ${position.average_entry_price:,.2f} "
            f"(Realized P&L: ${position.realized_pnl:,.2f})"
        )

    if filled_orders:
        logger.info("\nOrder History:")
        for order in filled_orders[:10]:  # Show first 10
            logger.info(
                f"  {order.submitted_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{order.side.value.upper()} {order.quantity} @ "
                f"${order.average_filled_price:,.2f}"
            )

    logger.info("="*60)

    # 7. Cleanup
    await exchange.disconnect()

    logger.info("Backtest complete!")


async def run_with_real_data_example():
    """
    Example with real historical data

    This shows how to use EventReader to replay actual recorded events
    """

    logger.info("Backtest with historical data")

    # Create event reader
    # (Assumes you've recorded data using ParquetEventWriter)
    reader = EventReader(
        base_path="./data/events",
        speedup=10.0  # 10x speed
    )

    # Create simulated exchange with replay
    exchange = SimulatedExchange(
        event_reader=reader,
        fill_model="realistic",
        slippage_model="hybrid",  # Most realistic
        initial_balance=Decimal("100000")
    )

    # Run backtest for date range
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)
    symbols = ["BTC/USDT", "ETH/USDT"]

    results = await exchange.run_backtest(start, end, symbols)

    # Print results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    logger.info("="*60)


async def main():
    """Main entry point"""
    # Run simple example with simulated ticks
    await run_backtest_example()

    # Uncomment to run with real historical data:
    # await run_with_real_data_example()


if __name__ == "__main__":
    asyncio.run(main())
