"""
Momentum Strategy Example

Implements an EMA crossover momentum strategy with ADX trend strength confirmation:
- Fast EMA crosses above slow EMA = Bullish momentum (BUY)
- Fast EMA crosses below slow EMA = Bearish momentum (SELL)
- ADX confirms trend strength (only trade when ADX > threshold)
- Volume confirmation for stronger signals

Strategy Logic:
1. Calculate Fast EMA (12-period) and Slow EMA (26-period)
2. Calculate ADX (14-period) for trend strength
3. Monitor volume vs average volume
4. Enter long when:
   - Fast EMA crosses above Slow EMA
   - ADX > 25 (strong trend)
   - Volume > 1.2x average (optional)
5. Exit when:
   - Fast EMA crosses below Slow EMA
   - Or price moves against us by stop loss %
   - Or profit target reached
6. Position sizing: 10% of portfolio per trade
7. Risk management: 3% stop loss, 9% profit target (1:3 risk/reward)

This strategy works best in trending markets.
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


class MomentumStrategy(BaseStrategy):
    """
    EMA Crossover + ADX Momentum Strategy

    This strategy captures trending moves by identifying momentum shifts
    through moving average crossovers, confirmed by trend strength (ADX).
    """

    def __init__(self, config, event_bus, portfolio_manager, order_manager):
        super().__init__(config, event_bus, portfolio_manager, order_manager)

        # Strategy parameters
        self.fast_ema_period = config.parameters.get("fast_ema_period", 12)
        self.slow_ema_period = config.parameters.get("slow_ema_period", 26)
        self.adx_period = config.parameters.get("adx_period", 14)
        self.adx_threshold = Decimal(str(config.parameters.get("adx_threshold", 25)))

        # Volume confirmation
        self.use_volume_confirmation = config.parameters.get("use_volume_confirmation", True)
        self.volume_multiplier = Decimal(str(config.parameters.get("volume_multiplier", 1.2)))

        # Position sizing and risk
        self.position_size_pct = Decimal(str(config.parameters.get("position_size_pct", 0.10)))  # 10%
        self.stop_loss_pct = Decimal(str(config.parameters.get("stop_loss_pct", 0.03)))  # 3%
        self.profit_target_pct = Decimal(str(config.parameters.get("profit_target_pct", 0.09)))  # 9%

        # Data storage
        self.prices: Dict[str, deque] = {}
        self.volumes: Dict[str, deque] = {}
        self.highs: Dict[str, deque] = {}
        self.lows: Dict[str, deque] = {}

        # Indicator values
        self.fast_ema: Dict[str, Optional[Decimal]] = {}
        self.slow_ema: Dict[str, Optional[Decimal]] = {}
        self.prev_fast_ema: Dict[str, Optional[Decimal]] = {}
        self.prev_slow_ema: Dict[str, Optional[Decimal]] = {}

        # Trading state
        self.entry_prices: Dict[str, Decimal] = {}
        self.stop_losses: Dict[str, Decimal] = {}
        self.profit_targets: Dict[str, Decimal] = {}
        self.in_position: Dict[str, bool] = {}

        # Statistics
        self.signals_generated = 0
        self.trades_entered = 0
        self.stopped_out = 0
        self.profit_targets_hit = 0

        logger.info(
            f"Initialized Momentum Strategy: "
            f"EMA({self.fast_ema_period}/{self.slow_ema_period}), "
            f"ADX({self.adx_period}, threshold={self.adx_threshold}), "
            f"Risk/Reward: {self.stop_loss_pct:.1%}/{self.profit_target_pct:.1%}"
        )

    async def on_start(self):
        """Called when strategy starts"""
        logger.info("=" * 70)
        logger.info("Momentum Strategy Started")
        logger.info(f"Subscribed symbols: {self.config.subscriptions.get('symbols', [])}")
        logger.info(f"Position size: {self.position_size_pct:.1%} per trade")
        logger.info(f"Stop loss: {self.stop_loss_pct:.1%}")
        logger.info(f"Profit target: {self.profit_target_pct:.1%}")
        logger.info("=" * 70)

    async def on_stop(self):
        """Called when strategy stops"""
        logger.info("=" * 70)
        logger.info("Momentum Strategy Stopped")
        logger.info(f"Signals generated: {self.signals_generated}")
        logger.info(f"Trades entered: {self.trades_entered}")
        logger.info(f"Stopped out: {self.stopped_out}")
        logger.info(f"Profit targets hit: {self.profit_targets_hit}")
        logger.info("=" * 70)

    async def on_tick(self, tick: TickEvent):
        """Process tick event and generate signals"""
        symbol = tick.symbol

        # Initialize data structures
        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=max(self.slow_ema_period, self.adx_period) * 3)
            self.volumes[symbol] = deque(maxlen=self.adx_period * 3)
            self.highs[symbol] = deque(maxlen=self.adx_period * 3)
            self.lows[symbol] = deque(maxlen=self.adx_period * 3)
            self.fast_ema[symbol] = None
            self.slow_ema[symbol] = None
            self.prev_fast_ema[symbol] = None
            self.prev_slow_ema[symbol] = None
            self.in_position[symbol] = False

        # Update price data
        self.prices[symbol].append(tick.last)
        self.volumes[symbol].append(tick.volume)
        # For tick data, use last price as high/low approximation
        self.highs[symbol].append(tick.last)
        self.lows[symbol].append(tick.last)

        # Need enough history
        if len(self.prices[symbol]) < self.slow_ema_period:
            return

        # Calculate indicators
        self._calculate_emas(symbol)
        adx = self._calculate_adx(symbol)

        if adx is None or self.fast_ema[symbol] is None or self.slow_ema[symbol] is None:
            return

        # Check for signals
        await self._check_entry_signals(symbol, tick, adx)
        await self._check_exit_signals(symbol, tick)
        await self._check_stop_loss_and_target(symbol, tick)

    async def on_fill(self, fill: FillEvent):
        """Handle order fill events"""
        symbol = fill.symbol

        if fill.side == "buy":
            # Record entry and set risk management levels
            self.entry_prices[symbol] = fill.price
            self.stop_losses[symbol] = fill.price * (Decimal("1") - self.stop_loss_pct)
            self.profit_targets[symbol] = fill.price * (Decimal("1") + self.profit_target_pct)
            self.in_position[symbol] = True

            risk_reward = self.profit_target_pct / self.stop_loss_pct

            logger.info(
                f"📍 LONG Position Opened: {symbol} @ {fill.price:.2f} | "
                f"Stop: {self.stop_losses[symbol]:.2f} | "
                f"Target: {self.profit_targets[symbol]:.2f} | "
                f"R:R = 1:{risk_reward:.1f}"
            )

        elif fill.side == "sell":
            # Calculate P&L
            entry_price = self.entry_prices.get(symbol)
            if entry_price:
                pnl_pct = ((fill.price - entry_price) / entry_price) * Decimal("100")
                logger.info(
                    f"📍 Position Closed: {symbol} @ {fill.price:.2f} | "
                    f"Entry: {entry_price:.2f} | "
                    f"P&L: {pnl_pct:+.2f}%"
                )

            # Clear tracking
            self.entry_prices.pop(symbol, None)
            self.stop_losses.pop(symbol, None)
            self.profit_targets.pop(symbol, None)
            self.in_position[symbol] = False

    def _calculate_emas(self, symbol: str):
        """Calculate Exponential Moving Averages"""
        prices = list(self.prices[symbol])

        # Save previous values
        self.prev_fast_ema[symbol] = self.fast_ema[symbol]
        self.prev_slow_ema[symbol] = self.slow_ema[symbol]

        # Calculate Fast EMA
        if self.fast_ema[symbol] is None:
            # Initialize with SMA
            if len(prices) >= self.fast_ema_period:
                self.fast_ema[symbol] = sum(prices[-self.fast_ema_period:]) / self.fast_ema_period
        else:
            # EMA calculation
            multiplier = Decimal("2") / Decimal(str(self.fast_ema_period + 1))
            self.fast_ema[symbol] = (prices[-1] - self.fast_ema[symbol]) * multiplier + self.fast_ema[symbol]

        # Calculate Slow EMA
        if self.slow_ema[symbol] is None:
            # Initialize with SMA
            if len(prices) >= self.slow_ema_period:
                self.slow_ema[symbol] = sum(prices[-self.slow_ema_period:]) / self.slow_ema_period
        else:
            # EMA calculation
            multiplier = Decimal("2") / Decimal(str(self.slow_ema_period + 1))
            self.slow_ema[symbol] = (prices[-1] - self.slow_ema[symbol]) * multiplier + self.slow_ema[symbol]

    def _calculate_adx(self, symbol: str) -> Optional[Decimal]:
        """
        Calculate ADX (Average Directional Index) for trend strength

        ADX measures the strength of a trend (not direction).
        Values above 25 typically indicate a strong trend.
        """
        if len(self.highs[symbol]) < self.adx_period + 1:
            return None

        highs = list(self.highs[symbol])
        lows = list(self.lows[symbol])

        # Calculate True Range and Directional Movement
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, len(highs)):
            high = highs[i]
            low = lows[i]
            prev_high = highs[i - 1]
            prev_low = lows[i - 1]
            prev_close = self.prices[symbol][i - 1]

            # True Range
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_list.append(tr)

            # Directional Movement
            plus_dm = max(high - prev_high, Decimal("0"))
            minus_dm = max(prev_low - low, Decimal("0"))

            if plus_dm > minus_dm:
                plus_dm_list.append(plus_dm)
                minus_dm_list.append(Decimal("0"))
            elif minus_dm > plus_dm:
                plus_dm_list.append(Decimal("0"))
                minus_dm_list.append(minus_dm)
            else:
                plus_dm_list.append(Decimal("0"))
                minus_dm_list.append(Decimal("0"))

        if len(tr_list) < self.adx_period:
            return None

        # Calculate smoothed TR and DM
        recent_tr = tr_list[-self.adx_period:]
        recent_plus_dm = plus_dm_list[-self.adx_period:]
        recent_minus_dm = minus_dm_list[-self.adx_period:]

        atr = sum(recent_tr) / len(recent_tr)
        plus_di = (sum(recent_plus_dm) / len(recent_plus_dm)) / atr * Decimal("100") if atr > 0 else Decimal("0")
        minus_di = (sum(recent_minus_dm) / len(recent_minus_dm)) / atr * Decimal("100") if atr > 0 else Decimal("0")

        # Calculate DX and ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * Decimal("100") if (plus_di + minus_di) > 0 else Decimal("0")

        # For simplicity, return DX as ADX approximation
        # (proper ADX would smooth DX over period)
        return dx

    def _check_volume_confirmation(self, symbol: str) -> bool:
        """Check if current volume confirms the signal"""
        if not self.use_volume_confirmation:
            return True

        if len(self.volumes[symbol]) < 20:
            return True  # Not enough data, skip confirmation

        current_volume = self.volumes[symbol][-1]
        avg_volume = sum(list(self.volumes[symbol])[-20:]) / 20

        return current_volume >= (avg_volume * self.volume_multiplier)

    async def _check_entry_signals(self, symbol: str, tick: TickEvent, adx: Decimal):
        """Check for entry signals"""

        # Don't enter if already in position
        if self.in_position.get(symbol, False):
            return

        # Need previous EMA values to detect crossover
        if self.prev_fast_ema[symbol] is None or self.prev_slow_ema[symbol] is None:
            return

        fast_ema = self.fast_ema[symbol]
        slow_ema = self.slow_ema[symbol]
        prev_fast = self.prev_fast_ema[symbol]
        prev_slow = self.prev_slow_ema[symbol]

        # Detect bullish crossover: Fast EMA crosses above Slow EMA
        bullish_cross = prev_fast <= prev_slow and fast_ema > slow_ema

        if not bullish_cross:
            return

        # Check ADX confirmation (strong trend)
        if adx < self.adx_threshold:
            logger.debug(f"Signal ignored: {symbol} - ADX too low ({adx:.1f} < {self.adx_threshold})")
            return

        # Check volume confirmation
        if not self._check_volume_confirmation(symbol):
            logger.debug(f"Signal ignored: {symbol} - Volume too low")
            return

        # All conditions met - generate signal
        self.signals_generated += 1

        # Calculate position size
        stats = self.portfolio_manager.get_statistics()
        position_value = stats["current_equity"] * self.position_size_pct
        quantity = position_value / tick.last
        quantity = quantity.quantize(Decimal("0.00001"))

        if quantity > 0:
            logger.info(
                f"🟢 MOMENTUM BUY SIGNAL: {symbol} @ {tick.last:.2f} | "
                f"Fast EMA: {fast_ema:.2f} | Slow EMA: {slow_ema:.2f} | "
                f"ADX: {adx:.1f} | Qty: {quantity:.5f}"
            )

            await self.place_market_order(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                exchange=tick.exchange
            )

            self.trades_entered += 1

    async def _check_exit_signals(self, symbol: str, tick: TickEvent):
        """Check for exit signals (EMA crossover against position)"""

        if not self.in_position.get(symbol, False):
            return

        # Need previous EMA values
        if self.prev_fast_ema[symbol] is None or self.prev_slow_ema[symbol] is None:
            return

        fast_ema = self.fast_ema[symbol]
        slow_ema = self.slow_ema[symbol]
        prev_fast = self.prev_fast_ema[symbol]
        prev_slow = self.prev_slow_ema[symbol]

        # Detect bearish crossover: Fast EMA crosses below Slow EMA
        bearish_cross = prev_fast >= prev_slow and fast_ema < slow_ema

        if bearish_cross:
            position = self.portfolio_manager.position_tracker.get_position(symbol, tick.exchange)

            if position and position.quantity > 0:
                logger.info(
                    f"🔴 MOMENTUM EXIT SIGNAL: {symbol} @ {tick.last:.2f} | "
                    f"Fast EMA: {fast_ema:.2f} | Slow EMA: {slow_ema:.2f} | "
                    f"Reason: Bearish crossover"
                )

                await self.place_market_order(
                    symbol=symbol,
                    side="sell",
                    quantity=position.quantity,
                    exchange=tick.exchange
                )

    async def _check_stop_loss_and_target(self, symbol: str, tick: TickEvent):
        """Check stop loss and profit target"""

        if not self.in_position.get(symbol, False):
            return

        if symbol not in self.stop_losses or symbol not in self.profit_targets:
            return

        position = self.portfolio_manager.position_tracker.get_position(symbol, tick.exchange)
        if not position or position.quantity <= 0:
            return

        current_price = tick.last
        stop_loss = self.stop_losses[symbol]
        profit_target = self.profit_targets[symbol]
        entry_price = self.entry_prices.get(symbol, current_price)

        # Check stop loss
        if current_price <= stop_loss:
            loss_pct = ((current_price - entry_price) / entry_price) * Decimal("100")

            logger.warning(
                f"🛑 STOP LOSS HIT: {symbol} @ {current_price:.2f} | "
                f"Entry: {entry_price:.2f} | Loss: {loss_pct:.2f}%"
            )

            self.stopped_out += 1

            await self.place_market_order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                exchange=tick.exchange
            )

        # Check profit target
        elif current_price >= profit_target:
            profit_pct = ((current_price - entry_price) / entry_price) * Decimal("100")

            logger.info(
                f"🎯 PROFIT TARGET HIT: {symbol} @ {current_price:.2f} | "
                f"Entry: {entry_price:.2f} | Profit: {profit_pct:.2f}%"
            )

            self.profit_targets_hit += 1

            await self.place_market_order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                exchange=tick.exchange
            )


async def run_momentum_backtest():
    """Run backtest with momentum strategy"""
    logger.info("=" * 70)
    logger.info("MOMENTUM STRATEGY BACKTEST")
    logger.info("=" * 70)

    # Strategy configuration
    strategy_config = StrategyConfig(
        name="EMA_ADX_Momentum",
        strategy_id="momentum_v1",
        enabled=True,
        mode="simulation",
        subscriptions={
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "exchanges": ["simulated"],
            "data_types": ["tick"]
        },
        parameters={
            # EMA settings
            "fast_ema_period": 12,
            "slow_ema_period": 26,

            # ADX settings
            "adx_period": 14,
            "adx_threshold": 25,

            # Volume confirmation
            "use_volume_confirmation": True,
            "volume_multiplier": 1.2,

            # Risk management
            "position_size_pct": 0.10,  # 10% per trade
            "stop_loss_pct": 0.03,  # 3% stop
            "profit_target_pct": 0.09,  # 9% target (1:3 R:R)
        }
    )

    # Backtest configuration
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),  # 3 months
        initial_capital=Decimal("100000"),
        fill_model="realistic",
        slippage_model="spread",
        maker_fee=Decimal("0.0001"),
        taker_fee=Decimal("0.0002"),
        data_frequency="1m",
        enable_risk_checks=True
    )

    # Run backtest
    engine = BacktestEngine(
        strategy_class=MomentumStrategy,
        strategy_config=strategy_config,
        config=backtest_config
    )

    result = await engine.run()

    # Print results
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
    """Test different EMA period combinations"""
    logger.info("\n" + "=" * 70)
    logger.info("PARAMETER OPTIMIZATION")
    logger.info("=" * 70)

    from trading.backtest import BacktestRunner

    runner = BacktestRunner()

    # Test different EMA combinations
    fast_periods = [8, 12, 16]
    slow_periods = [21, 26, 30]

    best_result = None
    best_sharpe = Decimal("-999")

    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue

            logger.info(f"\n🔍 Testing: Fast EMA={fast}, Slow EMA={slow}")

            strategy_config = StrategyConfig(
                name=f"Momentum_EMA{fast}_{slow}",
                strategy_id=f"mom_{fast}_{slow}",
                enabled=True,
                mode="simulation",
                subscriptions={
                    "symbols": ["BTC/USDT"],
                    "exchanges": ["simulated"],
                    "data_types": ["tick"]
                },
                parameters={
                    "fast_ema_period": fast,
                    "slow_ema_period": slow,
                    "adx_period": 14,
                    "adx_threshold": 25,
                    "position_size_pct": 0.10,
                    "stop_loss_pct": 0.03,
                    "profit_target_pct": 0.09,
                }
            )

            backtest_config = BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                initial_capital=Decimal("100000"),
                fill_model="realistic"
            )

            result = await runner.run_backtest(
                MomentumStrategy,
                strategy_config,
                backtest_config
            )

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


async def main():
    """Main entry point"""
    # Run backtest
    await run_momentum_backtest()

    # Uncomment to run optimization
    # await run_parameter_optimization()


if __name__ == "__main__":
    asyncio.run(main())
