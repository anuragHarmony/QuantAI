"""
Performance and Risk Metrics

Comprehensive set of financial metrics for strategy evaluation:
- Return metrics: Total return, CAGR, daily/monthly/yearly returns
- Risk-adjusted metrics: Sharpe, Sortino, Calmar, Omega ratios
- Risk metrics: Volatility, max drawdown, VaR, CVaR
- Trade metrics: Win rate, profit factor, avg win/loss
"""
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import math


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""

    # Return metrics
    total_return: Decimal
    total_return_pct: Decimal
    cagr: Decimal  # Compound Annual Growth Rate
    daily_return_avg: Decimal
    monthly_return_avg: Decimal

    # Risk-adjusted metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    omega_ratio: Optional[Decimal] = None
    information_ratio: Optional[Decimal] = None

    # Time period
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=datetime.now)
    num_days: int = 0

    # Benchmark comparison (if applicable)
    alpha: Optional[Decimal] = None
    beta: Optional[Decimal] = None


@dataclass
class RiskMetrics:
    """Risk metrics container"""

    # Volatility
    daily_volatility: Decimal
    annual_volatility: Decimal

    # Drawdown
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    max_drawdown_duration: Optional[timedelta] = None
    current_drawdown: Decimal = Decimal("0")
    current_drawdown_pct: Decimal = Decimal("0")

    # Value at Risk
    var_95: Optional[Decimal] = None  # 95% VaR
    var_99: Optional[Decimal] = None  # 99% VaR
    cvar_95: Optional[Decimal] = None  # Conditional VaR (Expected Shortfall)
    cvar_99: Optional[Decimal] = None

    # Tail risk
    skewness: Optional[Decimal] = None
    kurtosis: Optional[Decimal] = None


@dataclass
class TradeMetrics:
    """Trade-level metrics container"""

    # Trade counts
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    win_rate: Decimal

    # P&L
    gross_profit: Decimal
    gross_loss: Decimal
    net_profit: Decimal
    profit_factor: Decimal

    # Win/Loss statistics
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    avg_win_loss_ratio: Decimal

    # Holding period
    avg_holding_period: Optional[timedelta] = None
    max_holding_period: Optional[timedelta] = None
    min_holding_period: Optional[timedelta] = None

    # Consecutive trades
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Costs
    total_fees: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")


def calculate_returns(equity_curve: List[Decimal]) -> List[Decimal]:
    """
    Calculate returns from equity curve

    Args:
        equity_curve: List of equity values

    Returns:
        List of returns (as decimals, not percentages)
    """
    if len(equity_curve) < 2:
        return []

    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i-1] != 0:
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)
        else:
            returns.append(Decimal("0"))

    return returns


def calculate_sharpe_ratio(
    returns: List[Decimal],
    risk_free_rate: Decimal = Decimal("0.02"),
    periods_per_year: int = 252
) -> Decimal:
    """
    Calculate Sharpe Ratio

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

    Args:
        returns: List of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) < 2:
        return Decimal("0")

    # Convert to float for calculation
    returns_float = [float(r) for r in returns]

    # Calculate mean return
    mean_return = sum(returns_float) / len(returns_float)

    # Calculate standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns_float) / (len(returns_float) - 1)
    std_dev = variance ** 0.5

    if std_dev == 0:
        return Decimal("0")

    # Annualize
    annual_mean = mean_return * periods_per_year
    annual_std = std_dev * (periods_per_year ** 0.5)

    # Calculate Sharpe
    sharpe = (annual_mean - float(risk_free_rate)) / annual_std

    return Decimal(str(sharpe))


def calculate_sortino_ratio(
    returns: List[Decimal],
    risk_free_rate: Decimal = Decimal("0.02"),
    periods_per_year: int = 252
) -> Decimal:
    """
    Calculate Sortino Ratio

    Similar to Sharpe but only considers downside deviation.
    Penalizes downside volatility but not upside volatility.

    Args:
        returns: List of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio (annualized)
    """
    if len(returns) < 2:
        return Decimal("0")

    returns_float = [float(r) for r in returns]
    mean_return = sum(returns_float) / len(returns_float)

    # Calculate downside deviation (only negative returns)
    downside_returns = [r for r in returns_float if r < 0]

    if len(downside_returns) == 0:
        # No downside - infinite Sortino (cap at high value)
        return Decimal("999")

    downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
    downside_std = downside_variance ** 0.5

    if downside_std == 0:
        return Decimal("0")

    # Annualize
    annual_mean = mean_return * periods_per_year
    annual_downside_std = downside_std * (periods_per_year ** 0.5)

    # Calculate Sortino
    sortino = (annual_mean - float(risk_free_rate)) / annual_downside_std

    return Decimal(str(sortino))


def calculate_calmar_ratio(
    total_return: Decimal,
    max_drawdown: Decimal,
    years: Decimal
) -> Decimal:
    """
    Calculate Calmar Ratio

    Calmar = Annualized Return / Max Drawdown

    Measures return per unit of downside risk.

    Args:
        total_return: Total return as decimal (e.g., 0.5 for 50%)
        max_drawdown: Max drawdown as positive decimal (e.g., 0.2 for 20% DD)
        years: Number of years in period

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0 or years == 0:
        return Decimal("0")

    # Annualized return
    annualized_return = total_return / years

    # Calmar ratio
    calmar = annualized_return / max_drawdown

    return calmar


def calculate_max_drawdown(equity_curve: List[Decimal]) -> Tuple[Decimal, Decimal]:
    """
    Calculate maximum drawdown

    Args:
        equity_curve: List of equity values

    Returns:
        Tuple of (max_drawdown_value, max_drawdown_pct)
    """
    if len(equity_curve) == 0:
        return Decimal("0"), Decimal("0")

    max_equity = equity_curve[0]
    max_dd_value = Decimal("0")
    max_dd_pct = Decimal("0")

    for equity in equity_curve:
        # Update peak
        if equity > max_equity:
            max_equity = equity

        # Calculate drawdown from peak
        dd_value = max_equity - equity
        dd_pct = (dd_value / max_equity) if max_equity > 0 else Decimal("0")

        # Update max drawdown
        if dd_value > max_dd_value:
            max_dd_value = dd_value
            max_dd_pct = dd_pct

    return max_dd_value, max_dd_pct


def calculate_var(
    returns: List[Decimal],
    confidence_level: Decimal = Decimal("0.95")
) -> Decimal:
    """
    Calculate Value at Risk (VaR) using historical method

    VaR answers: "What is the maximum loss we can expect with X% confidence?"

    Args:
        returns: List of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR value (negative number representing loss)
    """
    if len(returns) < 10:
        return Decimal("0")

    # Sort returns
    sorted_returns = sorted([float(r) for r in returns])

    # Find percentile
    index = int(len(sorted_returns) * (1 - float(confidence_level)))
    var = sorted_returns[index]

    return Decimal(str(var))


def calculate_cvar(
    returns: List[Decimal],
    confidence_level: Decimal = Decimal("0.95")
) -> Decimal:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

    CVaR is the expected loss given that VaR has been exceeded.

    Args:
        returns: List of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)

    Returns:
        CVaR value (negative number representing expected loss in tail)
    """
    if len(returns) < 10:
        return Decimal("0")

    # Calculate VaR first
    var = calculate_var(returns, confidence_level)

    # Find all returns worse than VaR
    returns_float = [float(r) for r in returns]
    tail_returns = [r for r in returns_float if r <= float(var)]

    if len(tail_returns) == 0:
        return var

    # Calculate mean of tail
    cvar = sum(tail_returns) / len(tail_returns)

    return Decimal(str(cvar))


def calculate_cagr(
    initial_value: Decimal,
    final_value: Decimal,
    years: Decimal
) -> Decimal:
    """
    Calculate Compound Annual Growth Rate

    CAGR = (Final Value / Initial Value) ^ (1 / Years) - 1

    Args:
        initial_value: Starting value
        final_value: Ending value
        years: Number of years (can be fractional)

    Returns:
        CAGR as decimal (e.g., 0.15 for 15% CAGR)
    """
    if initial_value == 0 or years == 0:
        return Decimal("0")

    ratio = float(final_value / initial_value)
    if ratio <= 0:
        return Decimal("0")

    cagr = (ratio ** (1 / float(years))) - 1

    return Decimal(str(cagr))


def calculate_volatility(
    returns: List[Decimal],
    periods_per_year: int = 252
) -> Tuple[Decimal, Decimal]:
    """
    Calculate volatility (standard deviation of returns)

    Args:
        returns: List of returns
        periods_per_year: Number of periods per year for annualization

    Returns:
        Tuple of (period_volatility, annual_volatility)
    """
    if len(returns) < 2:
        return Decimal("0"), Decimal("0")

    returns_float = [float(r) for r in returns]
    mean = sum(returns_float) / len(returns_float)

    variance = sum((r - mean) ** 2 for r in returns_float) / (len(returns_float) - 1)
    period_vol = variance ** 0.5

    annual_vol = period_vol * (periods_per_year ** 0.5)

    return Decimal(str(period_vol)), Decimal(str(annual_vol))


def calculate_skewness(returns: List[Decimal]) -> Decimal:
    """
    Calculate skewness of returns distribution

    Skewness measures asymmetry:
    - Negative: More extreme losses than gains
    - Positive: More extreme gains than losses
    - Zero: Symmetric distribution

    Args:
        returns: List of returns

    Returns:
        Skewness value
    """
    if len(returns) < 3:
        return Decimal("0")

    returns_float = [float(r) for r in returns]
    n = len(returns_float)
    mean = sum(returns_float) / n

    # Calculate moments
    m2 = sum((r - mean) ** 2 for r in returns_float) / n
    m3 = sum((r - mean) ** 3 for r in returns_float) / n

    if m2 == 0:
        return Decimal("0")

    skew = m3 / (m2 ** 1.5)

    return Decimal(str(skew))


def calculate_kurtosis(returns: List[Decimal]) -> Decimal:
    """
    Calculate excess kurtosis of returns distribution

    Kurtosis measures tail risk:
    - Positive: Fat tails (more extreme events)
    - Negative: Thin tails (fewer extreme events)
    - Zero: Normal distribution

    Args:
        returns: List of returns

    Returns:
        Excess kurtosis value
    """
    if len(returns) < 4:
        return Decimal("0")

    returns_float = [float(r) for r in returns]
    n = len(returns_float)
    mean = sum(returns_float) / n

    # Calculate moments
    m2 = sum((r - mean) ** 2 for r in returns_float) / n
    m4 = sum((r - mean) ** 4 for r in returns_float) / n

    if m2 == 0:
        return Decimal("0")

    # Excess kurtosis (subtract 3 for normal distribution baseline)
    kurt = (m4 / (m2 ** 2)) - 3

    return Decimal(str(kurt))


def calculate_omega_ratio(
    returns: List[Decimal],
    threshold: Decimal = Decimal("0")
) -> Decimal:
    """
    Calculate Omega Ratio

    Omega = Sum(Returns above threshold) / Sum(Returns below threshold)

    Measures probability-weighted gains vs losses.

    Args:
        returns: List of returns
        threshold: Return threshold (default 0)

    Returns:
        Omega ratio
    """
    if len(returns) < 2:
        return Decimal("0")

    returns_float = [float(r) for r in returns]
    threshold_float = float(threshold)

    gains = sum(max(r - threshold_float, 0) for r in returns_float)
    losses = sum(max(threshold_float - r, 0) for r in returns_float)

    if losses == 0:
        return Decimal("999")  # Cap at high value

    omega = gains / losses

    return Decimal(str(omega))


logger.info("Performance metrics module loaded")
