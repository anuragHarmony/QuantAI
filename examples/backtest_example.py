"""
Backtesting Example

Demonstrates how to backtest a trading strategy using the backtesting framework.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from loguru import logger

from trading.backtest import BacktestEngine, BacktestConfig
from trading.strategy.base import BaseStrategy
from trading.strategy.config import StrategyConfig
from trading.events import TickEvent


class SimpleMovingAverageCrossStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy for backtesting demo

    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    """

    def __init__(self, config, event_bus, portfolio_manager, order_manager):
        super().__init__(config, event_bus, portfolio_manager, order_manager)

        # Strategy parameters
        self.fast_period = config.parameters.get("fast_period", 10)
        self.slow_period = config.parameters.get("slow_period", 30)

        # Price history for MA calculation
        self.prices = {}
        self.positions = {}

        logger.info(f"Initialized SMA Cross Strategy: fast={self.fast_period}, slow={self.slow_period}")

    async def on_start(self):
        """Called when strategy starts"""
        logger.info("SMA Cross Strategy started")

    async def on_stop(self):
        """Called when strategy stops"""
        logger.info("SMA Cross Strategy stopped")

    async def on_tick(self, tick: TickEvent):
        """Handle tick event"""
        symbol = tick.symbol

        # Update price history
        if symbol not in self.prices:
            self.prices[symbol] = []

        self.prices[symbol].append(tick.last)

        # Keep only recent prices
        max_period = max(self.fast_period, self.slow_period)
        if len(self.prices[symbol]) > max_period * 2:
            self.prices[symbol] = self.prices[symbol][-max_period:]

        # Need enough history
        if len(self.prices[symbol]) < self.slow_period:
            return

        # Calculate moving averages
        fast_ma = self._calculate_sma(self.prices[symbol], self.fast_period)
        slow_ma = self._calculate_sma(self.prices[symbol], self.slow_period)

        # Generate signals
        await self._check_signals(symbol, tick, fast_ma, slow_ma)

    def _calculate_sma(self, prices, period):
        """Calculate simple moving average"""
        if len(prices) < period:
            return Decimal("0")

        recent_prices = prices[-period:]
        return sum(recent_prices) / len(recent_prices)

    async def _check_signals(self, symbol, tick, fast_ma, slow_ma):
        """Check for trading signals"""
        # Get previous values for crossover detection
        if len(self.prices[symbol]) < self.slow_period + 1:
            return

        prev_fast_ma = self._calculate_sma(self.prices[symbol][:-1], self.fast_period)
        prev_slow_ma = self._calculate_sma(self.prices[symbol][:-1], self.slow_period)

        # Check for crossover
        bullish_cross = prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma
        bearish_cross = prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma

        # Get current position
        position = self.portfolio_manager.position_tracker.get_position(symbol, "simulated")
        has_position = position and position.quantity > 0

        # Buy signal
        if bullish_cross and not has_position:
            # Calculate position size (1% of portfolio)
            stats = self.portfolio_manager.get_statistics()
            position_value = stats["current_equity"] * Decimal("0.01")
            quantity = position_value / tick.last

            logger.info(f"🟢 BUY SIGNAL: {symbol} @ {tick.last} (Fast MA: {fast_ma:.2f}, Slow MA: {slow_ma:.2f})")

            # Place buy order
            await self.place_market_order(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                exchange="simulated"
            )

        # Sell signal
        elif bearish_cross and has_position:
            logger.info(f"🔴 SELL SIGNAL: {symbol} @ {tick.last} (Fast MA: {fast_ma:.2f}, Slow MA: {slow_ma:.2f})")

            # Place sell order to close position
            await self.place_market_order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                exchange="simulated"
            )


async def run_backtest_example():
    """Run backtest example"""
    logger.info("=" * 70)
    logger.info("BACKTESTING EXAMPLE - SMA Crossover Strategy")
    logger.info("=" * 70)

    # Strategy configuration
    strategy_config = StrategyConfig(
        name="SMA_Cross_10_30",
        strategy_id="sma_cross_demo",
        enabled=True,
        mode="simulation",
        subscriptions={
            "symbols": ["BTC/USDT"],
            "exchanges": ["simulated"],
            "data_types": ["tick"]
        },
        parameters={
            "fast_period": 10,
            "slow_period": 30
        }
    )

    # Backtest configuration
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),  # 1 month backtest
        initial_capital=Decimal("100000"),
        fill_model="realistic",
        slippage_model="spread",
        maker_fee=Decimal("0.0001"),  # 1 bps
        taker_fee=Decimal("0.0002"),  # 2 bps
        data_frequency="1m",
        enable_risk_checks=True
    )

    # Create and run backtest
    engine = BacktestEngine(
        strategy_class=SimpleMovingAverageCrossStrategy,
        strategy_config=strategy_config,
        config=backtest_config
    )

    result = await engine.run()

    # Results are automatically printed by the engine
    # You can also access them programmatically
    logger.info(f"\n📊 Final P&L: ${result.total_pnl:,.2f}")
    logger.info(f"📈 Return: {result.total_return_pct:.2f}%")
    logger.info(f"🎯 Win Rate: {result.win_rate:.2f}%")
    logger.info(f"📉 Max Drawdown: {result.max_drawdown_pct:.2f}%")

    return result


async def run_parameter_optimization_example():
    """Example of parameter optimization"""
    logger.info("\n" + "=" * 70)
    logger.info("PARAMETER OPTIMIZATION EXAMPLE")
    logger.info("=" * 70)

    from trading.backtest import BacktestRunner

    runner = BacktestRunner()

    # Test different MA periods
    fast_periods = [5, 10, 15, 20]
    slow_periods = [20, 30, 40, 50]

    logger.info(f"\nTesting {len(fast_periods) * len(slow_periods)} parameter combinations...")

    best_result = None
    best_sharpe = Decimal("-999")

    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue  # Skip invalid combinations

            logger.info(f"\nTesting: Fast={fast}, Slow={slow}")

            # Strategy config with parameters
            strategy_config = StrategyConfig(
                name=f"SMA_Cross_{fast}_{slow}",
                strategy_id=f"sma_{fast}_{slow}",
                enabled=True,
                mode="simulation",
                subscriptions={
                    "symbols": ["BTC/USDT"],
                    "exchanges": ["simulated"],
                    "data_types": ["tick"]
                },
                parameters={
                    "fast_period": fast,
                    "slow_period": slow
                }
            )

            # Backtest config
            backtest_config = BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 15),  # 2 weeks for speed
                initial_capital=Decimal("100000"),
                fill_model="realistic"
            )

            # Run backtest
            result = await runner.run_backtest(
                SimpleMovingAverageCrossStrategy,
                strategy_config,
                backtest_config
            )

            # Track best result
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_result = result

    # Show comparison
    logger.info("\n" + runner.compare_results())

    if best_result:
        logger.info(f"\n🏆 BEST PARAMETERS:")
        logger.info(f"   Strategy: {best_result.strategy_name}")
        logger.info(f"   Return: {best_result.total_return_pct:.2f}%")
        logger.info(f"   Sharpe: {best_result.sharpe_ratio:.2f}")
        logger.info(f"   Max DD: {best_result.max_drawdown_pct:.2f}%")


async def main():
    """Main entry point"""
    # Run simple backtest
    await run_backtest_example()

    # Uncomment to run parameter optimization
    # await run_parameter_optimization_example()


if __name__ == "__main__":
    asyncio.run(main())
