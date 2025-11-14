"""
Backtesting Framework

High-performance backtesting engine for strategy validation before live trading.

Features:
- Historical data replay with accurate timing
- Realistic order execution simulation
- Complete P&L tracking and performance metrics
- Transaction cost and slippage modeling
- Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
- Parameter optimization and grid search
- Comprehensive result analysis and comparison

Usage:
    from trading.backtest import BacktestEngine, BacktestConfig, BacktestResult
    from datetime import datetime
    from decimal import Decimal

    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=Decimal("100000"),
        fill_model="realistic",
        slippage_model="spread"
    )

    engine = BacktestEngine(
        strategy_class=MyStrategy,
        strategy_config=my_strategy_config,
        config=config
    )

    result = await engine.run()
    print(result.summary())
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "BacktestRunner",
]
