"""
Performance Analyzer

Analyzes strategy performance using equity curves and trade data:
- Calculate comprehensive performance metrics
- Analyze trade patterns
- Time series analysis
- Period-based analysis (daily, weekly, monthly)
- Drawdown analysis
"""
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

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
    calculate_cagr,
    calculate_volatility,
    calculate_skewness,
    calculate_kurtosis,
    calculate_omega_ratio,
)


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    side: str  # "long" or "short"
    pnl: Decimal
    pnl_pct: Decimal
    fees: Decimal = Decimal("0")
    exchange: str = ""


@dataclass
class EquityPoint:
    """Equity curve data point"""
    timestamp: datetime
    equity: Decimal
    cash: Decimal
    positions_value: Decimal


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer

    Analyzes strategy performance from equity curve and trade history.
    """

    def __init__(
        self,
        initial_capital: Decimal,
        equity_curve: List[EquityPoint],
        trades: Optional[List[Trade]] = None,
        risk_free_rate: Decimal = Decimal("0.02")
    ):
        """
        Initialize analyzer

        Args:
            initial_capital: Starting capital
            equity_curve: List of equity points over time
            trades: List of completed trades (optional)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.equity_curve = equity_curve
        self.trades = trades or []
        self.risk_free_rate = risk_free_rate

        logger.info(
            f"Initialized PerformanceAnalyzer: "
            f"{len(equity_curve)} equity points, {len(self.trades)} trades"
        )

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all performance, risk, and trade metrics

        Returns:
            Dictionary containing all metrics
        """
        logger.info("Calculating comprehensive performance metrics...")

        performance = self.calculate_performance_metrics()
        risk = self.calculate_risk_metrics()
        trade_metrics = self.calculate_trade_metrics() if self.trades else None

        result = {
            "performance": performance,
            "risk": risk,
            "trades": trade_metrics,
            "summary": self._generate_summary(performance, risk, trade_metrics)
        }

        logger.info("✓ Metrics calculation complete")
        return result

    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics"""

        if len(self.equity_curve) == 0:
            logger.warning("Empty equity curve")
            return self._empty_performance_metrics()

        # Extract equity values
        equity_values = [point.equity for point in self.equity_curve]

        # Calculate returns
        returns = calculate_returns(equity_values)

        # Time period
        start_date = self.equity_curve[0].timestamp
        end_date = self.equity_curve[-1].timestamp
        duration = end_date - start_date
        years = Decimal(str(duration.days / 365.25))

        # Total return
        final_equity = equity_values[-1]
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * Decimal("100") if self.initial_capital > 0 else Decimal("0")

        # CAGR
        cagr = calculate_cagr(self.initial_capital, final_equity, years) if years > 0 else Decimal("0")

        # Average returns
        daily_return_avg = sum(returns) / len(returns) if returns else Decimal("0")
        # Approximate monthly (assuming 21 trading days per month)
        monthly_return_avg = daily_return_avg * Decimal("21")

        # Risk-adjusted metrics
        sharpe = calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = calculate_sortino_ratio(returns, self.risk_free_rate)

        # Max drawdown for Calmar
        _, max_dd_pct = calculate_max_drawdown(equity_values)
        calmar = calculate_calmar_ratio(total_return_pct / Decimal("100"), max_dd_pct, years) if years > 0 else Decimal("0")

        # Omega ratio
        omega = calculate_omega_ratio(returns)

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            cagr=cagr,
            daily_return_avg=daily_return_avg,
            monthly_return_avg=monthly_return_avg,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            start_date=start_date,
            end_date=end_date,
            num_days=duration.days
        )

    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate risk metrics"""

        if len(self.equity_curve) == 0:
            logger.warning("Empty equity curve")
            return self._empty_risk_metrics()

        equity_values = [point.equity for point in self.equity_curve]
        returns = calculate_returns(equity_values)

        # Volatility
        daily_vol, annual_vol = calculate_volatility(returns)

        # Max drawdown
        max_dd_value, max_dd_pct = calculate_max_drawdown(equity_values)

        # Current drawdown
        current_equity = equity_values[-1]
        peak_equity = max(equity_values)
        current_dd = peak_equity - current_equity
        current_dd_pct = (current_dd / peak_equity) if peak_equity > 0 else Decimal("0")

        # VaR and CVaR
        var_95 = calculate_var(returns, Decimal("0.95"))
        var_99 = calculate_var(returns, Decimal("0.99"))
        cvar_95 = calculate_cvar(returns, Decimal("0.95"))
        cvar_99 = calculate_cvar(returns, Decimal("0.99"))

        # Tail risk
        skewness = calculate_skewness(returns)
        kurtosis = calculate_kurtosis(returns)

        return RiskMetrics(
            daily_volatility=daily_vol,
            annual_volatility=annual_vol,
            max_drawdown=max_dd_value,
            max_drawdown_pct=max_dd_pct,
            current_drawdown=current_dd,
            current_drawdown_pct=current_dd_pct,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            skewness=skewness,
            kurtosis=kurtosis
        )

    def calculate_trade_metrics(self) -> Optional[TradeMetrics]:
        """Calculate trade-level metrics"""

        if len(self.trades) == 0:
            return None

        # Separate winning and losing trades
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        num_trades = len(self.trades)

        win_rate = Decimal(str(num_wins / num_trades * 100)) if num_trades > 0 else Decimal("0")

        # P&L calculations
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        net_profit = gross_profit - gross_loss

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else Decimal("999")

        # Average win/loss
        avg_win = (gross_profit / len(winning_trades)) if winning_trades else Decimal("0")
        avg_loss = (gross_loss / len(losing_trades)) if losing_trades else Decimal("0")
        avg_win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else Decimal("999")

        # Largest win/loss
        largest_win = max((t.pnl for t in winning_trades), default=Decimal("0"))
        largest_loss = min((t.pnl for t in losing_trades), default=Decimal("0"))

        # Holding periods
        holding_periods = [(t.exit_time - t.entry_time) for t in self.trades]
        avg_holding = sum(holding_periods, timedelta()) / len(holding_periods) if holding_periods else None
        max_holding = max(holding_periods) if holding_periods else None
        min_holding = min(holding_periods) if holding_periods else None

        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calculate_consecutive_trades()

        # Costs
        total_fees = sum(t.fees for t in self.trades)

        return TradeMetrics(
            num_trades=num_trades,
            num_winning_trades=num_wins,
            num_losing_trades=num_losses,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_win_loss_ratio=avg_win_loss_ratio,
            avg_holding_period=avg_holding,
            max_holding_period=max_holding,
            min_holding_period=min_holding,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            total_fees=total_fees
        )

    def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""

        if not self.trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    def _generate_summary(
        self,
        performance: PerformanceMetrics,
        risk: RiskMetrics,
        trade_metrics: Optional[TradeMetrics]
    ) -> str:
        """Generate text summary"""

        summary = f"""
Performance Summary:
  Total Return: {performance.total_return_pct:.2f}%
  CAGR: {performance.cagr * 100:.2f}%
  Sharpe Ratio: {performance.sharpe_ratio:.3f}
  Sortino Ratio: {performance.sortino_ratio:.3f}

Risk Metrics:
  Max Drawdown: {risk.max_drawdown_pct * 100:.2f}%
  Annual Volatility: {risk.annual_volatility * 100:.2f}%
  95% VaR: {risk.var_95 * 100:.2f}%
"""

        if trade_metrics:
            summary += f"""
Trade Metrics:
  Total Trades: {trade_metrics.num_trades}
  Win Rate: {trade_metrics.win_rate:.2f}%
  Profit Factor: {trade_metrics.profit_factor:.2f}
  Avg Win/Loss: {trade_metrics.avg_win_loss_ratio:.2f}
"""

        return summary

    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_return=Decimal("0"),
            total_return_pct=Decimal("0"),
            cagr=Decimal("0"),
            daily_return_avg=Decimal("0"),
            monthly_return_avg=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            calmar_ratio=Decimal("0")
        )

    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            daily_volatility=Decimal("0"),
            annual_volatility=Decimal("0"),
            max_drawdown=Decimal("0"),
            max_drawdown_pct=Decimal("0")
        )


class TimeSeriesAnalyzer:
    """Analyze performance over different time periods"""

    def __init__(self, equity_curve: List[EquityPoint]):
        self.equity_curve = equity_curve

    def analyze_by_period(self, period: str = "monthly") -> List[Dict[str, Any]]:
        """
        Analyze returns by time period

        Args:
            period: "daily", "weekly", "monthly", "yearly"

        Returns:
            List of period metrics
        """
        # TODO: Implement period-based analysis
        return []


class TradeAnalyzer:
    """Detailed trade analysis"""

    def __init__(self, trades: List[Trade]):
        self.trades = trades

    def analyze_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by symbol"""

        symbol_metrics = {}

        for symbol in set(t.symbol for t in self.trades):
            symbol_trades = [t for t in self.trades if t.symbol == symbol]

            num_trades = len(symbol_trades)
            wins = [t for t in symbol_trades if t.pnl > 0]
            losses = [t for t in symbol_trades if t.pnl < 0]

            total_pnl = sum(t.pnl for t in symbol_trades)
            win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0

            symbol_metrics[symbol] = {
                "num_trades": num_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "num_wins": len(wins),
                "num_losses": len(losses)
            }

        return symbol_metrics

    def analyze_by_time_of_day(self) -> Dict[int, Dict[str, Any]]:
        """Analyze performance by hour of day"""

        hour_metrics = {}

        for hour in range(24):
            hour_trades = [t for t in self.trades if t.entry_time.hour == hour]

            if not hour_trades:
                continue

            num_trades = len(hour_trades)
            total_pnl = sum(t.pnl for t in hour_trades)
            wins = len([t for t in hour_trades if t.pnl > 0])
            win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

            hour_metrics[hour] = {
                "num_trades": num_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate
            }

        return hour_metrics


logger.info("Performance analyzer module loaded")
