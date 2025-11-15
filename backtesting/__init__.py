"""
Backtesting Module

Vectorized backtesting engine for trading strategies.
"""

from .engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    Position,
)

from .metrics import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_comprehensive_metrics,
)

from .strategy_loader import (
    StrategyLoader,
    load_strategy_from_code,
)

from .visualization import BacktestVisualizer

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "Position",

    # Metrics
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_sortino_ratio",
    "calculate_comprehensive_metrics",

    # Strategy loading
    "StrategyLoader",
    "load_strategy_from_code",

    # Visualization
    "BacktestVisualizer",
]
