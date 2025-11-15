"""
Backtesting Engine

Vectorized backtesting for trading strategies.
"""
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from backtesting.metrics import PerformanceMetrics, calculate_comprehensive_metrics


@dataclass
class Trade:
    """A completed trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float  # Position size (positive for long, negative for short)
    direction: str  # 'long' or 'short'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    mae: float = 0.0  # Maximum adverse excursion
    mfe: float = 0.0  # Maximum favorable excursion


@dataclass
class Position:
    """Current open position"""
    entry_time: datetime
    entry_price: float
    size: float
    direction: str
    unrealized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest"""
    # Performance metrics
    metrics: PerformanceMetrics

    # Trade data
    trades: List[Trade]
    equity_curve: pd.Series

    # Strategy info
    strategy_name: str
    parameters: Dict[str, Any]

    # Backtest settings
    initial_capital: float
    commission: float
    slippage: float

    # Data info
    start_date: datetime
    end_date: datetime
    timeframe: str
    symbol: str

    # Additional data
    signals: Optional[pd.DataFrame] = None
    positions: Optional[pd.DataFrame] = None


class BacktestEngine:
    """
    Vectorized backtesting engine

    Single Responsibility: Execute strategy backtests
    Open/Closed: Extend with new execution modes
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
        position_size_pct: float = 1.0,  # 100% of capital per trade
        max_positions: int = 1,
        enable_shorting: bool = True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.enable_shorting = enable_shorting

        logger.info(f"Initialized BacktestEngine (capital: ${initial_capital:,.2f})")

    async def run_backtest(
        self,
        strategy_class: Type,
        data: pd.DataFrame,
        strategy_parameters: Optional[Dict[str, Any]] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h"
    ) -> BacktestResult:
        """
        Run backtest on a strategy

        Args:
            strategy_class: Strategy class to backtest
            data: OHLCV data with DatetimeIndex
            strategy_parameters: Parameters for strategy
            symbol: Symbol being traded
            timeframe: Timeframe of data

        Returns:
            BacktestResult with metrics and trades
        """
        logger.info(f"Running backtest: {strategy_class.__name__}")

        # Validate data
        if data.empty:
            logger.error("Empty data provided")
            return self._empty_result(strategy_class.__name__, symbol, timeframe)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Data missing required columns: {required_cols}")
            return self._empty_result(strategy_class.__name__, symbol, timeframe)

        try:
            # Initialize strategy
            strategy = strategy_class(parameters=strategy_parameters)

            # Calculate indicators
            logger.debug("Calculating indicators...")
            data_with_indicators = strategy.calculate_indicators(data.copy())

            # Generate signals
            logger.debug("Generating signals...")
            data_with_signals = strategy.generate_signals(data_with_indicators)

            if 'signal' not in data_with_signals.columns:
                logger.error("Strategy did not generate 'signal' column")
                return self._empty_result(strategy_class.__name__, symbol, timeframe)

            # Run simulation
            logger.debug("Running simulation...")
            trades, equity_curve, positions = self._simulate_trading(
                data_with_signals,
                strategy_parameters or {}
            )

            # Calculate metrics
            logger.debug("Calculating metrics...")
            metrics = calculate_comprehensive_metrics(
                equity_curve=equity_curve,
                trades=trades,
                initial_capital=self.initial_capital
            )

            # Create result
            result = BacktestResult(
                metrics=metrics,
                trades=trades,
                equity_curve=equity_curve,
                strategy_name=strategy_class.__name__,
                parameters=strategy_parameters or {},
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage,
                start_date=data.index[0],
                end_date=data.index[-1],
                timeframe=timeframe,
                symbol=symbol,
                signals=data_with_signals,
                positions=positions
            )

            logger.success(
                f"Backtest complete: Sharpe={metrics.sharpe_ratio:.2f}, "
                f"Return={metrics.total_return*100:.1f}%, Trades={metrics.total_trades}"
            )

            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(strategy_class.__name__, symbol, timeframe)

    def _simulate_trading(
        self,
        data: pd.DataFrame,
        strategy_params: Dict[str, Any]
    ) -> tuple[List[Trade], pd.Series, pd.DataFrame]:
        """
        Simulate trading based on signals

        Args:
            data: DataFrame with signals
            strategy_params: Strategy parameters (stop_loss_pct, etc.)

        Returns:
            (trades, equity_curve, positions_df)
        """
        trades = []
        equity = self.initial_capital
        equity_curve = []

        current_position: Optional[Position] = None

        # Position tracking
        position_data = []

        # Get parameters
        stop_loss_pct = strategy_params.get('stop_loss_pct', 2.0) / 100
        take_profit_pct = strategy_params.get('take_profit_pct', None)
        if take_profit_pct:
            take_profit_pct = take_profit_pct / 100

        for idx, row in data.iterrows():
            signal = row['signal']
            price = row['close']
            high = row['high']
            low = row['low']

            # Check if we need to exit current position
            if current_position is not None:
                # Calculate unrealized P&L
                if current_position.direction == 'long':
                    unrealized_pnl = (price - current_position.entry_price) * current_position.size
                    pnl_pct = (price - current_position.entry_price) / current_position.entry_price

                    # Check stop loss (worst case: hit low)
                    if (current_position.entry_price - low) / current_position.entry_price >= stop_loss_pct:
                        # Stopped out
                        exit_price = current_position.entry_price * (1 - stop_loss_pct)
                        pnl = (exit_price - current_position.entry_price) * current_position.size
                        pnl -= self._calculate_cost(exit_price, current_position.size)

                        trades.append(Trade(
                            entry_time=current_position.entry_time,
                            exit_time=idx,
                            entry_price=current_position.entry_price,
                            exit_price=exit_price,
                            size=current_position.size,
                            direction=current_position.direction,
                            pnl=pnl,
                            pnl_pct=-stop_loss_pct
                        ))

                        equity += pnl
                        current_position = None
                        continue

                    # Check take profit (best case: hit high)
                    if take_profit_pct and (high - current_position.entry_price) / current_position.entry_price >= take_profit_pct:
                        exit_price = current_position.entry_price * (1 + take_profit_pct)
                        pnl = (exit_price - current_position.entry_price) * current_position.size
                        pnl -= self._calculate_cost(exit_price, current_position.size)

                        trades.append(Trade(
                            entry_time=current_position.entry_time,
                            exit_time=idx,
                            entry_price=current_position.entry_price,
                            exit_price=exit_price,
                            size=current_position.size,
                            direction=current_position.direction,
                            pnl=pnl,
                            pnl_pct=take_profit_pct
                        ))

                        equity += pnl
                        current_position = None
                        continue

                else:  # short
                    unrealized_pnl = (current_position.entry_price - price) * abs(current_position.size)
                    pnl_pct = (current_position.entry_price - price) / current_position.entry_price

                    # Check stop loss (worst case: hit high)
                    if (high - current_position.entry_price) / current_position.entry_price >= stop_loss_pct:
                        exit_price = current_position.entry_price * (1 + stop_loss_pct)
                        pnl = (current_position.entry_price - exit_price) * abs(current_position.size)
                        pnl -= self._calculate_cost(exit_price, abs(current_position.size))

                        trades.append(Trade(
                            entry_time=current_position.entry_time,
                            exit_time=idx,
                            entry_price=current_position.entry_price,
                            exit_price=exit_price,
                            size=current_position.size,
                            direction=current_position.direction,
                            pnl=pnl,
                            pnl_pct=-stop_loss_pct
                        ))

                        equity += pnl
                        current_position = None
                        continue

                current_position.unrealized_pnl = unrealized_pnl

                # Exit signal
                if signal == -1 and current_position.direction == 'long':
                    # Exit long
                    exit_price = price * (1 - self.slippage)
                    pnl = (exit_price - current_position.entry_price) * current_position.size
                    pnl -= self._calculate_cost(exit_price, current_position.size)

                    trades.append(Trade(
                        entry_time=current_position.entry_time,
                        exit_time=idx,
                        entry_price=current_position.entry_price,
                        exit_price=exit_price,
                        size=current_position.size,
                        direction=current_position.direction,
                        pnl=pnl,
                        pnl_pct=pnl_pct
                    ))

                    equity += pnl
                    current_position = None

                elif signal == 1 and current_position.direction == 'short':
                    # Exit short
                    exit_price = price * (1 + self.slippage)
                    pnl = (current_position.entry_price - exit_price) * abs(current_position.size)
                    pnl -= self._calculate_cost(exit_price, abs(current_position.size))

                    trades.append(Trade(
                        entry_time=current_position.entry_time,
                        exit_time=idx,
                        entry_price=current_position.entry_price,
                        exit_price=exit_price,
                        size=current_position.size,
                        direction=current_position.direction,
                        pnl=pnl,
                        pnl_pct=pnl_pct
                    ))

                    equity += pnl
                    current_position = None

            # Enter new position
            if current_position is None and signal != 0:
                if signal == 1:  # Buy
                    entry_price = price * (1 + self.slippage)
                    position_value = equity * self.position_size_pct
                    size = (position_value / entry_price) * 0.99  # 1% buffer for costs

                    cost = self._calculate_cost(entry_price, size)
                    if cost < equity:
                        current_position = Position(
                            entry_time=idx,
                            entry_price=entry_price,
                            size=size,
                            direction='long'
                        )

                elif signal == -1 and self.enable_shorting:  # Sell short
                    entry_price = price * (1 - self.slippage)
                    position_value = equity * self.position_size_pct
                    size = -(position_value / entry_price) * 0.99

                    cost = self._calculate_cost(entry_price, abs(size))
                    if cost < equity:
                        current_position = Position(
                            entry_time=idx,
                            entry_price=entry_price,
                            size=size,
                            direction='short'
                        )

            # Track equity
            current_equity = equity
            if current_position:
                current_equity += current_position.unrealized_pnl

            equity_curve.append(current_equity)

            # Track position
            position_data.append({
                'timestamp': idx,
                'equity': current_equity,
                'position': current_position.direction if current_position else 'flat',
                'position_size': current_position.size if current_position else 0,
                'unrealized_pnl': current_position.unrealized_pnl if current_position else 0
            })

        # Close any remaining position
        if current_position is not None:
            final_price = data.iloc[-1]['close']
            if current_position.direction == 'long':
                exit_price = final_price * (1 - self.slippage)
                pnl = (exit_price - current_position.entry_price) * current_position.size
            else:
                exit_price = final_price * (1 + self.slippage)
                pnl = (current_position.entry_price - exit_price) * abs(current_position.size)

            pnl -= self._calculate_cost(exit_price, abs(current_position.size))
            pnl_pct = pnl / (current_position.entry_price * abs(current_position.size))

            trades.append(Trade(
                entry_time=current_position.entry_time,
                exit_time=data.index[-1],
                entry_price=current_position.entry_price,
                exit_price=exit_price,
                size=current_position.size,
                direction=current_position.direction,
                pnl=pnl,
                pnl_pct=pnl_pct
            ))

            equity += pnl

        # Create equity curve series
        equity_series = pd.Series(equity_curve, index=data.index)

        # Create positions DataFrame
        positions_df = pd.DataFrame(position_data)

        return trades, equity_series, positions_df

    def _calculate_cost(self, price: float, size: float) -> float:
        """Calculate transaction costs (commission + slippage)"""
        trade_value = price * abs(size)
        return trade_value * (self.commission + self.slippage)

    def _empty_result(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str
    ) -> BacktestResult:
        """Create empty result for failed backtests"""
        from backtesting.metrics import _empty_metrics

        return BacktestResult(
            metrics=_empty_metrics(),
            trades=[],
            equity_curve=pd.Series([self.initial_capital]),
            strategy_name=strategy_name,
            parameters={},
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
            start_date=datetime.now(),
            end_date=datetime.now(),
            timeframe=timeframe,
            symbol=symbol
        )


logger.info("Backtesting engine loaded")
