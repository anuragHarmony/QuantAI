"""
Mean Reversion Strategy Example

Implements a Bollinger Bands + RSI mean reversion strategy:
- Buy when price touches lower Bollinger Band and RSI is oversold
- Sell when price touches upper Bollinger Band and RSI is overbought
- Exit when price returns to middle band (mean)

Strategy Logic:
1. Calculate Bollinger Bands (20-period SMA ± 2 std dev)
2. Calculate RSI (14-period)
3. Enter long when: price < lower band AND RSI < 30
4. Exit long when: price > middle band OR RSI > 70
5. Position sizing: 5% of portfolio per trade
6. Risk management: Stop loss at 2% below entry

This is a complete, production-ready strategy example.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from loguru import logger
from collections import deque

from trading.strategy.base import BaseStrategy
from trading.strategy.config import StrategyConfig
from trading.events import TickEvent, FillEvent
from trading.backtest import BacktestEngine, BacktestConfig


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI Mean Reversion Strategy

    This strategy exploits short-term price deviations from the mean,
    expecting prices to revert back to their average over time.
    """

    def __init__(self, config, event_bus, portfolio_manager, order_manager):
        super().__init__(config, event_bus, portfolio_manager, order_manager)

        # Strategy parameters (configurable via config.parameters)
        self.bb_period = config.parameters.get("bb_period", 20)
        self.bb_std_dev = Decimal(str(config.parameters.get("bb_std_dev", 2.0)))
        self.rsi_period = config.parameters.get("rsi_period", 14)
        self.rsi_oversold = Decimal(str(config.parameters.get("rsi_oversold", 30)))
        self.rsi_overbought = Decimal(str(config.parameters.get("rsi_overbought", 70)))

        # Position sizing
        self.position_size_pct = Decimal(str(config.parameters.get("position_size_pct", 0.05)))  # 5%
        self.stop_loss_pct = Decimal(str(config.parameters.get("stop_loss_pct", 0.02)))  # 2%

        # Price history per symbol
        self.prices: Dict[str, deque] = {}
        self.rsi_gains: Dict[str, deque] = {}
        self.rsi_losses: Dict[str, deque] = {}

        # Trading state
        self.entry_prices: Dict[str, Decimal] = {}
        self.stop_losses: Dict[str, Decimal] = {}

        # Statistics
        self.signals_generated = 0
        self.trades_entered = 0

        logger.info(
            f"Initialized Mean Reversion Strategy: "
            f"BB({self.bb_period}, {self.bb_std_dev}), "
            f"RSI({self.rsi_period}, {self.rsi_oversold}/{self.rsi_overbought})"
        )

    async def on_start(self):
        """Called when strategy starts"""
        logger.info("=" * 70)
        logger.info("Mean Reversion Strategy Started")
        logger.info(f"Subscribed symbols: {self.config.subscriptions.get('symbols', [])}")
        logger.info("=" * 70)

    async def on_stop(self):
        """Called when strategy stops"""
        logger.info("=" * 70)
        logger.info("Mean Reversion Strategy Stopped")
        logger.info(f"Signals generated: {self.signals_generated}")
        logger.info(f"Trades entered: {self.trades_entered}")
        logger.info("=" * 70)

    async def on_tick(self, tick: TickEvent):
        """Process tick event and generate signals"""
        symbol = tick.symbol

        # Initialize data structures for this symbol
        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=self.bb_period * 2)
            self.rsi_gains[symbol] = deque(maxlen=self.rsi_period)
            self.rsi_losses[symbol] = deque(maxlen=self.rsi_period)

        # Update price history
        self.prices[symbol].append(tick.last)

        # Need enough history for indicators
        if len(self.prices[symbol]) < max(self.bb_period, self.rsi_period + 1):
            return

        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(symbol)
        rsi = self._calculate_rsi(symbol)

        if bb_upper is None or rsi is None:
            return

        # Check for signals
        await self._check_entry_signals(symbol, tick, bb_upper, bb_middle, bb_lower, rsi)
        await self._check_exit_signals(symbol, tick, bb_middle, rsi)
        await self._check_stop_loss(symbol, tick)

    async def on_fill(self, fill: FillEvent):
        """Handle order fill events"""
        symbol = fill.symbol

        if fill.side == "buy":
            # Record entry price and set stop loss
            self.entry_prices[symbol] = fill.price
            self.stop_losses[symbol] = fill.price * (Decimal("1") - self.stop_loss_pct)

            logger.info(
                f"📍 Position opened: {symbol} @ {fill.price:.2f} "
                f"(Stop: {self.stop_losses[symbol]:.2f})"
            )

        elif fill.side == "sell":
            # Clear entry tracking
            entry_price = self.entry_prices.get(symbol)
            if entry_price:
                pnl_pct = ((fill.price - entry_price) / entry_price) * Decimal("100")
                logger.info(
                    f"📍 Position closed: {symbol} @ {fill.price:.2f} "
                    f"(Entry: {entry_price:.2f}, P&L: {pnl_pct:+.2f}%)"
                )

            self.entry_prices.pop(symbol, None)
            self.stop_losses.pop(symbol, None)

    def _calculate_bollinger_bands(self, symbol: str) -> tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculate Bollinger Bands

        Returns:
            (upper_band, middle_band, lower_band) or (None, None, None)
        """
        prices = list(self.prices[symbol])

        if len(prices) < self.bb_period:
            return None, None, None

        # Get recent prices
        recent_prices = prices[-self.bb_period:]

        # Calculate SMA (middle band)
        sma = sum(recent_prices) / len(recent_prices)

        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = variance ** Decimal("0.5")

        # Calculate bands
        upper_band = sma + (self.bb_std_dev * std_dev)
        lower_band = sma - (self.bb_std_dev * std_dev)

        return upper_band, sma, lower_band

    def _calculate_rsi(self, symbol: str) -> Optional[Decimal]:
        """
        Calculate RSI (Relative Strength Index)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        prices = list(self.prices[symbol])

        if len(prices) < self.rsi_period + 1:
            return None

        # Calculate price changes
        for i in range(len(prices) - self.rsi_period, len(prices)):
            if i == 0:
                continue

            change = prices[i] - prices[i - 1]

            if change > 0:
                if len(self.rsi_gains[symbol]) >= self.rsi_period:
                    self.rsi_gains[symbol].popleft()
                self.rsi_gains[symbol].append(change)

                if len(self.rsi_losses[symbol]) >= self.rsi_period:
                    self.rsi_losses[symbol].popleft()
                self.rsi_losses[symbol].append(Decimal("0"))
            else:
                if len(self.rsi_gains[symbol]) >= self.rsi_period:
                    self.rsi_gains[symbol].popleft()
                self.rsi_gains[symbol].append(Decimal("0"))

                if len(self.rsi_losses[symbol]) >= self.rsi_period:
                    self.rsi_losses[symbol].popleft()
                self.rsi_losses[symbol].append(abs(change))

        if len(self.rsi_gains[symbol]) < self.rsi_period:
            return None

        # Calculate average gain and loss
        avg_gain = sum(self.rsi_gains[symbol]) / len(self.rsi_gains[symbol])
        avg_loss = sum(self.rsi_losses[symbol]) / len(self.rsi_losses[symbol])

        if avg_loss == 0:
            return Decimal("100")

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    async def _check_entry_signals(
        self,
        symbol: str,
        tick: TickEvent,
        bb_upper: Decimal,
        bb_middle: Decimal,
        bb_lower: Decimal,
        rsi: Decimal
    ):
        """Check for entry signals"""

        # Get current position
        position = self.portfolio_manager.position_tracker.get_position(symbol, tick.exchange)
        has_position = position and position.quantity > 0

        if has_position:
            return  # Already in position

        current_price = tick.last

        # LONG ENTRY: Price touches lower band AND RSI oversold
        if current_price <= bb_lower and rsi <= self.rsi_oversold:
            self.signals_generated += 1

            # Calculate position size
            stats = self.portfolio_manager.get_statistics()
            position_value = stats["current_equity"] * self.position_size_pct
            quantity = position_value / current_price

            # Round quantity to reasonable precision
            quantity = quantity.quantize(Decimal("0.00001"))

            if quantity > 0:
                logger.info(
                    f"🟢 LONG SIGNAL: {symbol} @ {current_price:.2f} | "
                    f"BB: [{bb_lower:.2f}, {bb_middle:.2f}, {bb_upper:.2f}] | "
                    f"RSI: {rsi:.1f} | "
                    f"Qty: {quantity:.5f}"
                )

                # Place market buy order
                await self.place_market_order(
                    symbol=symbol,
                    side="buy",
                    quantity=quantity,
                    exchange=tick.exchange
                )

                self.trades_entered += 1

    async def _check_exit_signals(
        self,
        symbol: str,
        tick: TickEvent,
        bb_middle: Decimal,
        rsi: Decimal
    ):
        """Check for exit signals"""

        # Get current position
        position = self.portfolio_manager.position_tracker.get_position(symbol, tick.exchange)

        if not position or position.quantity <= 0:
            return  # No position to exit

        current_price = tick.last

        # EXIT: Price returns to middle band OR RSI overbought
        should_exit = False
        exit_reason = ""

        if current_price >= bb_middle:
            should_exit = True
            exit_reason = "Price at mean"
        elif rsi >= self.rsi_overbought:
            should_exit = True
            exit_reason = "RSI overbought"

        if should_exit:
            logger.info(
                f"🔴 EXIT SIGNAL: {symbol} @ {current_price:.2f} | "
                f"Reason: {exit_reason} | "
                f"RSI: {rsi:.1f} | "
                f"Qty: {position.quantity:.5f}"
            )

            # Place market sell order
            await self.place_market_order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                exchange=tick.exchange
            )

    async def _check_stop_loss(self, symbol: str, tick: TickEvent):
        """Check stop loss"""

        if symbol not in self.stop_losses:
            return

        # Get current position
        position = self.portfolio_manager.position_tracker.get_position(symbol, tick.exchange)

        if not position or position.quantity <= 0:
            return

        current_price = tick.last
        stop_loss = self.stop_losses[symbol]

        # Check if stop loss triggered
        if current_price <= stop_loss:
            entry_price = self.entry_prices.get(symbol, current_price)
            loss_pct = ((current_price - entry_price) / entry_price) * Decimal("100")

            logger.warning(
                f"🛑 STOP LOSS: {symbol} @ {current_price:.2f} | "
                f"Entry: {entry_price:.2f} | "
                f"Loss: {loss_pct:.2f}% | "
                f"Qty: {position.quantity:.5f}"
            )

            # Place market sell order
            await self.place_market_order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                exchange=tick.exchange
            )


async def run_mean_reversion_backtest():
    """Run backtest example with mean reversion strategy"""
    logger.info("=" * 70)
    logger.info("MEAN REVERSION STRATEGY BACKTEST")
    logger.info("=" * 70)

    # Strategy configuration
    strategy_config = StrategyConfig(
        name="BB_RSI_Mean_Reversion",
        strategy_id="mean_reversion_v1",
        enabled=True,
        mode="simulation",
        subscriptions={
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "exchanges": ["simulated"],
            "data_types": ["tick"]
        },
        parameters={
            # Bollinger Bands
            "bb_period": 20,
            "bb_std_dev": 2.0,

            # RSI
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,

            # Position sizing
            "position_size_pct": 0.05,  # 5% per trade
            "stop_loss_pct": 0.02,  # 2% stop loss
        }
    )

    # Backtest configuration
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),  # 3 months
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
        strategy_class=MeanReversionStrategy,
        strategy_config=strategy_config,
        config=backtest_config
    )

    result = await engine.run()

    # Print detailed results
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Final P&L: ${result.total_pnl:,.2f}")
    logger.info(f"📈 Return: {result.total_return_pct:.2f}%")
    logger.info(f"🎯 Win Rate: {result.win_rate:.2f}%")
    logger.info(f"📉 Max Drawdown: {result.max_drawdown_pct:.2f}%")
    logger.info(f"💰 Profit Factor: {result.profit_factor:.2f}")
    logger.info(f"📊 Sharpe Ratio: {result.sharpe_ratio:.3f}")
    logger.info("=" * 70)

    return result


async def run_parameter_optimization():
    """Test different parameter combinations"""
    logger.info("\n" + "=" * 70)
    logger.info("PARAMETER OPTIMIZATION")
    logger.info("=" * 70)

    from trading.backtest import BacktestRunner

    runner = BacktestRunner()

    # Test different combinations
    bb_periods = [15, 20, 25]
    rsi_periods = [10, 14, 18]

    best_result = None
    best_sharpe = Decimal("-999")

    for bb_period in bb_periods:
        for rsi_period in rsi_periods:
            logger.info(f"\n🔍 Testing: BB={bb_period}, RSI={rsi_period}")

            strategy_config = StrategyConfig(
                name=f"MeanRev_BB{bb_period}_RSI{rsi_period}",
                strategy_id=f"mr_{bb_period}_{rsi_period}",
                enabled=True,
                mode="simulation",
                subscriptions={
                    "symbols": ["BTC/USDT"],
                    "exchanges": ["simulated"],
                    "data_types": ["tick"]
                },
                parameters={
                    "bb_period": bb_period,
                    "bb_std_dev": 2.0,
                    "rsi_period": rsi_period,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "position_size_pct": 0.05,
                    "stop_loss_pct": 0.02,
                }
            )

            backtest_config = BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),  # 1 month for speed
                initial_capital=Decimal("100000"),
                fill_model="realistic"
            )

            result = await runner.run_backtest(
                MeanReversionStrategy,
                strategy_config,
                backtest_config
            )

            # Track best
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_result = result
                logger.info(f"✨ New best! Sharpe: {best_sharpe:.3f}")

    # Show comparison
    logger.info("\n" + runner.compare_results())

    if best_result:
        logger.info(f"\n🏆 BEST PARAMETERS:")
        logger.info(f"   Strategy: {best_result.strategy_name}")
        logger.info(f"   Return: {best_result.total_return_pct:.2f}%")
        logger.info(f"   Sharpe: {best_result.sharpe_ratio:.3f}")
        logger.info(f"   Win Rate: {best_result.win_rate:.2f}%")
        logger.info(f"   Max DD: {best_result.max_drawdown_pct:.2f}%")


async def main():
    """Main entry point"""
    # Run backtest
    await run_mean_reversion_backtest()

    # Uncomment to run parameter optimization
    # await run_parameter_optimization()


if __name__ == "__main__":
    asyncio.run(main())
