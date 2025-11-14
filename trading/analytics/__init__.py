"""
Performance Analytics and Reporting System

Comprehensive analytics for strategy performance evaluation:
- Performance metrics (returns, Sharpe, Sortino, Calmar, etc.)
- Risk metrics (volatility, VaR, drawdowns)
- Trade analysis (win rate, profit factor, avg win/loss)
- Attribution analysis (contribution by asset, strategy, time period)
- Report generation (text, JSON, CSV, HTML)
- Real-time performance tracking

Usage:
    from trading.analytics import PerformanceAnalyzer, ReportGenerator

    analyzer = PerformanceAnalyzer(trades, equity_curve)
    metrics = analyzer.calculate_all_metrics()

    generator = ReportGenerator()
    report = generator.generate_text_report(metrics)
"""

from .metrics import (
    PerformanceMetrics,
    RiskMetrics,
    TradeMetrics,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
)

from .analyzer import (
    PerformanceAnalyzer,
    TimeSeriesAnalyzer,
    TradeAnalyzer,
)

from .reporter import (
    ReportGenerator,
    ReportFormat,
    export_to_csv,
    export_to_json,
)

from .realtime import (
    RealTimePerformanceTracker,
    PerformanceSnapshot,
)

__all__ = [
    # Metrics
    "PerformanceMetrics",
    "RiskMetrics",
    "TradeMetrics",
    "calculate_returns",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_max_drawdown",
    "calculate_var",
    "calculate_cvar",

    # Analyzers
    "PerformanceAnalyzer",
    "TimeSeriesAnalyzer",
    "TradeAnalyzer",

    # Reporting
    "ReportGenerator",
    "ReportFormat",
    "export_to_csv",
    "export_to_json",

    # Real-time tracking
    "RealTimePerformanceTracker",
    "PerformanceSnapshot",
]
