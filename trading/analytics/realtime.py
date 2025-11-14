"""
Real-Time Performance Tracking

Track strategy performance in real-time during live trading:
- Rolling performance metrics
- Real-time risk monitoring
- Performance alerts
- Live dashboards updates
"""
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from loguru import logger

from .metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
)


@dataclass
class PerformanceSnapshot:
    """Real-time performance snapshot"""
    timestamp: datetime
    equity: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    sharpe_ratio: Decimal
    max_drawdown_pct: Decimal
    daily_pnl: Decimal
    num_trades_today: int
    win_rate_today: Decimal


@dataclass
class Alert:
    """Performance alert"""
    timestamp: datetime
    severity: str  # "info", "warning", "critical"
    category: str  # "drawdown", "volatility", "pnl", etc.
    message: str
    value: Optional[Decimal] = None
    threshold: Optional[Decimal] = None


class RealTimePerformanceTracker:
    """
    Real-time performance tracking for live trading

    Monitors strategy performance and generates alerts when thresholds are exceeded.
    """

    def __init__(
        self,
        initial_capital: Decimal,
        max_drawdown_threshold: Optional[Decimal] = None,
        daily_loss_threshold: Optional[Decimal] = None,
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        Initialize real-time tracker

        Args:
            initial_capital: Starting capital
            max_drawdown_threshold: Alert if drawdown exceeds this % (e.g., Decimal("0.10") for 10%)
            daily_loss_threshold: Alert if daily loss exceeds this amount
            alert_callback: Function to call when alerts are generated
        """
        self.initial_capital = initial_capital
        self.max_drawdown_threshold = max_drawdown_threshold
        self.daily_loss_threshold = daily_loss_threshold
        self.alert_callback = alert_callback

        # Equity tracking
        self.equity_history: deque = deque(maxlen=10000)  # Keep last 10k points
        self.equity_timestamps: deque = deque(maxlen=10000)

        # Daily tracking
        self.current_date = datetime.now().date()
        self.daily_start_equity = initial_capital
        self.trades_today = []

        # Peak tracking
        self.peak_equity = initial_capital
        self.peak_timestamp = datetime.now()

        # Alerts
        self.alerts: List[Alert] = []

        # Statistics
        self.last_snapshot: Optional[PerformanceSnapshot] = None

        logger.info(
            f"Initialized RealTimePerformanceTracker: "
            f"Initial capital: ${initial_capital:,.2f}"
        )

        if max_drawdown_threshold:
            logger.info(f"Max drawdown alert threshold: {max_drawdown_threshold * 100:.1f}%")
        if daily_loss_threshold:
            logger.info(f"Daily loss alert threshold: ${daily_loss_threshold:,.2f}")

    def update(self, equity: Decimal, timestamp: Optional[datetime] = None) -> PerformanceSnapshot:
        """
        Update with current equity value

        Args:
            equity: Current equity value
            timestamp: Optional timestamp (default: now)

        Returns:
            Current performance snapshot
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Check if new day
        if timestamp.date() != self.current_date:
            self._handle_new_day(timestamp)

        # Update equity history
        self.equity_history.append(equity)
        self.equity_timestamps.append(timestamp)

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_timestamp = timestamp

        # Calculate metrics
        snapshot = self._calculate_snapshot(equity, timestamp)
        self.last_snapshot = snapshot

        # Check alerts
        self._check_alerts(snapshot)

        return snapshot

    def record_trade(
        self,
        symbol: str,
        side: str,
        pnl: Decimal,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a completed trade

        Args:
            symbol: Traded symbol
            side: Trade side
            pnl: P&L from trade
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Check if new day
        if timestamp.date() != self.current_date:
            self._handle_new_day(timestamp)

        # Record trade
        self.trades_today.append({
            "symbol": symbol,
            "side": side,
            "pnl": pnl,
            "timestamp": timestamp
        })

        logger.debug(f"Recorded trade: {symbol} {side} P&L: ${pnl:,.2f}")

    def get_current_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Get last performance snapshot"""
        return self.last_snapshot

    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """
        Get recent alerts

        Args:
            minutes: Look back this many minutes

        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self.alerts if a.timestamp >= cutoff]

    def _calculate_snapshot(self, equity: Decimal, timestamp: datetime) -> PerformanceSnapshot:
        """Calculate current performance snapshot"""

        # Total return
        total_return = equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * Decimal("100") if self.initial_capital > 0 else Decimal("0")

        # Sharpe ratio (if enough history)
        sharpe = Decimal("0")
        if len(self.equity_history) >= 30:  # Need at least 30 data points
            equity_list = list(self.equity_history)
            returns = calculate_returns(equity_list)
            if returns:
                sharpe = calculate_sharpe_ratio(returns)

        # Max drawdown
        max_dd_pct = Decimal("0")
        if len(self.equity_history) > 0:
            equity_list = list(self.equity_history)
            _, max_dd_pct = calculate_max_drawdown(equity_list)

        # Daily P&L
        daily_pnl = equity - self.daily_start_equity

        # Today's trade stats
        num_trades_today = len(self.trades_today)
        winning_trades_today = len([t for t in self.trades_today if t["pnl"] > 0])
        win_rate_today = Decimal(str(winning_trades_today / num_trades_today * 100)) if num_trades_today > 0 else Decimal("0")

        return PerformanceSnapshot(
            timestamp=timestamp,
            equity=equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd_pct,
            daily_pnl=daily_pnl,
            num_trades_today=num_trades_today,
            win_rate_today=win_rate_today
        )

    def _check_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check for alert conditions"""

        # Max drawdown alert
        if self.max_drawdown_threshold and snapshot.max_drawdown_pct > self.max_drawdown_threshold:
            alert = Alert(
                timestamp=datetime.now(),
                severity="critical",
                category="drawdown",
                message=f"Max drawdown exceeded threshold",
                value=snapshot.max_drawdown_pct,
                threshold=self.max_drawdown_threshold
            )
            self._emit_alert(alert)

        # Daily loss alert
        if self.daily_loss_threshold and snapshot.daily_pnl < -self.daily_loss_threshold:
            alert = Alert(
                timestamp=datetime.now(),
                severity="warning",
                category="daily_loss",
                message=f"Daily loss exceeded threshold",
                value=snapshot.daily_pnl,
                threshold=-self.daily_loss_threshold
            )
            self._emit_alert(alert)

    def _emit_alert(self, alert: Alert) -> None:
        """Emit an alert"""

        # Add to alerts list
        self.alerts.append(alert)

        # Log the alert
        if alert.severity == "critical":
            logger.critical(
                f"🚨 {alert.category.upper()}: {alert.message} "
                f"(Value: {alert.value}, Threshold: {alert.threshold})"
            )
        elif alert.severity == "warning":
            logger.warning(
                f"⚠️  {alert.category.upper()}: {alert.message} "
                f"(Value: {alert.value}, Threshold: {alert.threshold})"
            )
        else:
            logger.info(
                f"ℹ️  {alert.category.upper()}: {alert.message} "
                f"(Value: {alert.value}, Threshold: {alert.threshold})"
            )

        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _handle_new_day(self, timestamp: datetime) -> None:
        """Handle transition to new day"""

        logger.info(f"New trading day started: {timestamp.date()}")

        # Log previous day stats
        if self.trades_today:
            daily_pnl = sum(t["pnl"] for t in self.trades_today)
            num_wins = len([t for t in self.trades_today if t["pnl"] > 0])
            win_rate = (num_wins / len(self.trades_today) * 100) if self.trades_today else 0

            logger.info(
                f"Previous day summary: "
                f"Trades: {len(self.trades_today)}, "
                f"P&L: ${daily_pnl:,.2f}, "
                f"Win rate: {win_rate:.1f}%"
            )

        # Reset daily tracking
        self.current_date = timestamp.date()
        self.daily_start_equity = self.equity_history[-1] if self.equity_history else self.initial_capital
        self.trades_today = []

    def get_statistics(self) -> Dict[str, any]:
        """
        Get current statistics

        Returns:
            Dictionary of current statistics
        """
        if not self.last_snapshot:
            return {}

        snapshot = self.last_snapshot

        return {
            "equity": float(snapshot.equity),
            "total_return": float(snapshot.total_return),
            "total_return_pct": float(snapshot.total_return_pct),
            "sharpe_ratio": float(snapshot.sharpe_ratio),
            "max_drawdown_pct": float(snapshot.max_drawdown_pct),
            "daily_pnl": float(snapshot.daily_pnl),
            "num_trades_today": snapshot.num_trades_today,
            "win_rate_today": float(snapshot.win_rate_today),
            "peak_equity": float(self.peak_equity),
            "current_drawdown_pct": float((self.peak_equity - snapshot.equity) / self.peak_equity * 100) if self.peak_equity > 0 else 0,
            "num_alerts": len(self.alerts),
            "num_recent_alerts": len(self.get_recent_alerts(60))
        }

    def reset(self) -> None:
        """Reset all tracking data"""
        logger.info("Resetting performance tracker")

        self.equity_history.clear()
        self.equity_timestamps.clear()
        self.trades_today = []
        self.alerts = []
        self.peak_equity = self.initial_capital
        self.daily_start_equity = self.initial_capital
        self.last_snapshot = None


logger.info("Real-time performance tracking module loaded")
