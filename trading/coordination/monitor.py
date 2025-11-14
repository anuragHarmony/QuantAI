"""
Strategy Monitor

Monitors performance and health of multiple strategies in real-time.
"""
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger
from collections import defaultdict


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy"""
    strategy_id: str

    # P&L
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Trade statistics
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    win_rate: Decimal = Decimal("0")

    # Risk metrics
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")

    # Recent performance
    recent_pnl: Decimal = Decimal("0")  # Last N trades
    recent_win_rate: Decimal = Decimal("0")

    # Health indicators
    is_healthy: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Timestamps
    last_trade_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.now)


class StrategyMonitor:
    """
    Monitor multiple strategies in real-time

    Tracks performance, detects issues, and generates alerts.
    """

    def __init__(
        self,
        max_drawdown_threshold: Optional[Decimal] = None,
        min_win_rate_threshold: Optional[Decimal] = None,
        inactivity_threshold_minutes: int = 60
    ):
        """
        Initialize monitor

        Args:
            max_drawdown_threshold: Alert if drawdown exceeds this
            min_win_rate_threshold: Alert if win rate falls below this
            inactivity_threshold_minutes: Alert if no trades for this many minutes
        """
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_win_rate_threshold = min_win_rate_threshold
        self.inactivity_threshold_minutes = inactivity_threshold_minutes

        # Metrics storage
        self.metrics: Dict[str, StrategyMetrics] = {}

        # Trade history per strategy
        self.trade_history: Dict[str, List[Dict]] = defaultdict(list)

        logger.info(
            f"Strategy monitor initialized: "
            f"DD threshold: {max_drawdown_threshold}, "
            f"Win rate threshold: {min_win_rate_threshold}"
        )

    def update_metrics(
        self,
        strategy_id: str,
        **kwargs
    ) -> StrategyMetrics:
        """
        Update metrics for a strategy

        Args:
            strategy_id: Strategy ID
            **kwargs: Metric values to update

        Returns:
            Updated metrics
        """
        if strategy_id not in self.metrics:
            self.metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)

        metrics = self.metrics[strategy_id]

        # Update provided metrics
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        metrics.last_update_time = datetime.now()

        # Recalculate derived metrics
        self._calculate_derived_metrics(metrics)

        # Check health
        self._check_health(metrics)

        return metrics

    def record_trade(
        self,
        strategy_id: str,
        pnl: Decimal,
        side: str,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a trade for a strategy

        Args:
            strategy_id: Strategy ID
            pnl: Trade P&L
            side: Trade side
            symbol: Traded symbol
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Initialize metrics if needed
        if strategy_id not in self.metrics:
            self.metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)

        metrics = self.metrics[strategy_id]

        # Record trade
        trade = {
            "pnl": pnl,
            "side": side,
            "symbol": symbol,
            "timestamp": timestamp
        }
        self.trade_history[strategy_id].append(trade)

        # Update metrics
        metrics.num_trades += 1
        metrics.total_pnl += pnl
        metrics.last_trade_time = timestamp

        if pnl > 0:
            metrics.num_wins += 1
        elif pnl < 0:
            metrics.num_losses += 1

        # Recalculate derived metrics
        self._calculate_derived_metrics(metrics)

        # Check health
        self._check_health(metrics)

        logger.debug(
            f"Recorded trade for {strategy_id}: "
            f"P&L: ${pnl:.2f}, Total: {metrics.num_trades}"
        )

    def get_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get metrics for a strategy"""
        return self.metrics.get(strategy_id)

    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies"""
        return self.metrics.copy()

    def get_summary(self) -> Dict[str, any]:
        """Get summary across all strategies"""

        total_pnl = sum(m.total_pnl for m in self.metrics.values())
        total_trades = sum(m.num_trades for m in self.metrics.values())
        total_wins = sum(m.num_wins for m in self.metrics.values())

        overall_win_rate = Decimal("0")
        if total_trades > 0:
            overall_win_rate = (Decimal(str(total_wins)) / Decimal(str(total_trades))) * Decimal("100")

        # Count strategies with issues
        unhealthy = len([m for m in self.metrics.values() if not m.is_healthy])
        with_warnings = len([m for m in self.metrics.values() if m.warnings])
        with_errors = len([m for m in self.metrics.values() if m.errors])

        return {
            "num_strategies": len(self.metrics),
            "total_pnl": float(total_pnl),
            "total_trades": total_trades,
            "overall_win_rate": float(overall_win_rate),
            "unhealthy_strategies": unhealthy,
            "strategies_with_warnings": with_warnings,
            "strategies_with_errors": with_errors,
        }

    def get_ranked_strategies(
        self,
        by: str = "sharpe_ratio",
        limit: int = 10
    ) -> List[StrategyMetrics]:
        """
        Get strategies ranked by a metric

        Args:
            by: Metric to rank by ("sharpe_ratio", "total_pnl", "win_rate")
            limit: Maximum number to return

        Returns:
            List of strategy metrics, ranked
        """
        strategies = list(self.metrics.values())

        # Sort by metric
        if by == "sharpe_ratio":
            strategies.sort(key=lambda m: m.sharpe_ratio, reverse=True)
        elif by == "total_pnl":
            strategies.sort(key=lambda m: m.total_pnl, reverse=True)
        elif by == "win_rate":
            strategies.sort(key=lambda m: m.win_rate, reverse=True)
        else:
            logger.warning(f"Unknown ranking metric: {by}")

        return strategies[:limit]

    def _calculate_derived_metrics(self, metrics: StrategyMetrics) -> None:
        """Calculate derived metrics from raw data"""

        # Win rate
        if metrics.num_trades > 0:
            metrics.win_rate = (Decimal(str(metrics.num_wins)) / Decimal(str(metrics.num_trades))) * Decimal("100")

        # Recent performance (last 10 trades)
        recent_trades = self.trade_history.get(metrics.strategy_id, [])[-10:]
        if recent_trades:
            metrics.recent_pnl = sum(t["pnl"] for t in recent_trades)
            recent_wins = len([t for t in recent_trades if t["pnl"] > 0])
            metrics.recent_win_rate = (Decimal(str(recent_wins)) / Decimal(str(len(recent_trades)))) * Decimal("100")

        # TODO: Calculate Sharpe ratio, max drawdown (requires equity curve)

    def _check_health(self, metrics: StrategyMetrics) -> None:
        """Check strategy health and generate warnings/errors"""

        metrics.warnings = []
        metrics.errors = []
        metrics.is_healthy = True

        # Check drawdown
        if self.max_drawdown_threshold and metrics.current_drawdown > self.max_drawdown_threshold:
            metrics.warnings.append(
                f"Drawdown {metrics.current_drawdown:.2%} exceeds threshold {self.max_drawdown_threshold:.2%}"
            )
            metrics.is_healthy = False

        # Check win rate
        if self.min_win_rate_threshold and metrics.num_trades > 10:
            if metrics.win_rate < self.min_win_rate_threshold:
                metrics.warnings.append(
                    f"Win rate {metrics.win_rate:.1f}% below threshold {self.min_win_rate_threshold:.1f}%"
                )
                metrics.is_healthy = False

        # Check inactivity
        if metrics.last_trade_time:
            inactive_duration = datetime.now() - metrics.last_trade_time
            threshold = timedelta(minutes=self.inactivity_threshold_minutes)

            if inactive_duration > threshold:
                minutes_inactive = inactive_duration.total_seconds() / 60
                metrics.warnings.append(
                    f"No trades for {minutes_inactive:.0f} minutes"
                )

        # Log health issues
        if not metrics.is_healthy:
            logger.warning(
                f"Strategy {metrics.strategy_id} health check failed: "
                f"{', '.join(metrics.warnings)}"
            )

    def reset_metrics(self, strategy_id: str) -> None:
        """Reset metrics for a strategy"""

        if strategy_id in self.metrics:
            self.metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
            self.trade_history[strategy_id] = []
            logger.info(f"Reset metrics for {strategy_id}")

    def reset_all(self) -> None:
        """Reset metrics for all strategies"""

        self.metrics.clear()
        self.trade_history.clear()
        logger.info("Reset all strategy metrics")


logger.info("Strategy monitor module loaded")
