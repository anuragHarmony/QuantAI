"""
Performance Metrics

Calculates trading strategy performance metrics.
"""
from typing import List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Complete set of performance metrics"""
    # Returns
    total_return: float
    annualized_return: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int  # in days
    volatility: float
    downside_deviation: float

    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Profit metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Other
    best_trade: float
    worst_trade: float
    avg_trade_duration: float  # in hours

    # Time-based
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_days: int


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Sharpe ratio

    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Annualized excess return
    excess_return = returns.mean() * periods_per_year - risk_free_rate

    # Annualized volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    if volatility == 0:
        return 0.0

    return excess_return / volatility


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility)

    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    # Annualized excess return
    excess_return = returns.mean() * periods_per_year - risk_free_rate

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if excess_return > 0 else 0.0

    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)

    if downside_deviation == 0:
        return 0.0

    return excess_return / downside_deviation


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """
    Calculate maximum drawdown and duration

    Args:
        equity_curve: Equity curve over time

    Returns:
        (max_drawdown_pct, duration_in_periods)
    """
    if len(equity_curve) == 0:
        return 0.0, 0

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max

    # Maximum drawdown
    max_dd = drawdown.min()

    # Calculate duration
    # Find the point of max drawdown
    max_dd_end = drawdown.idxmin()

    # Find when we were last at the peak before this drawdown
    max_dd_start = (equity_curve[:max_dd_end] == running_max[:max_dd_end]).idxmax()

    # Duration
    if isinstance(max_dd_end, (int, np.integer)) and isinstance(max_dd_start, (int, np.integer)):
        duration = int(max_dd_end - max_dd_start)
    else:
        duration = 0

    return abs(max_dd), duration


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown)

    Args:
        annualized_return: Annual return
        max_drawdown: Maximum drawdown (positive number)

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float('inf') if annualized_return > 0 else 0.0

    return annualized_return / max_drawdown


def calculate_profit_factor(
    winning_trades: List[float],
    losing_trades: List[float]
) -> float:
    """
    Calculate profit factor (total wins / total losses)

    Args:
        winning_trades: List of winning trade P&Ls
        losing_trades: List of losing trade P&Ls

    Returns:
        Profit factor
    """
    total_wins = sum(winning_trades) if winning_trades else 0
    total_losses = abs(sum(losing_trades)) if losing_trades else 0

    if total_losses == 0:
        return float('inf') if total_wins > 0 else 0.0

    return total_wins / total_losses


def calculate_expectancy(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate expectancy per trade

    Args:
        win_rate: Win rate (0-1)
        avg_win: Average winning trade
        avg_loss: Average losing trade (positive number)

    Returns:
        Expected value per trade
    """
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


def calculate_comprehensive_metrics(
    equity_curve: pd.Series,
    trades: List,
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics

    Args:
        equity_curve: Equity curve with DatetimeIndex
        trades: List of Trade objects
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object
    """
    if len(equity_curve) == 0 or len(trades) == 0:
        return _empty_metrics()

    # Calculate returns
    returns = equity_curve.pct_change().dropna()

    # Total return
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital

    # Annualized return
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = max(days / 365.25, 1/365.25)  # Minimum 1 day
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)

    # Sortino ratio
    sortino = calculate_sortino_ratio(returns, risk_free_rate)

    # Max drawdown
    max_dd, dd_duration = calculate_max_drawdown(equity_curve)

    # Calmar ratio
    calmar = calculate_calmar_ratio(annualized_return, max_dd)

    # Volatility
    volatility = returns.std() * np.sqrt(365)

    # Downside deviation
    downside_returns = returns[returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0.0

    # Trade statistics
    trade_pnls = [t.pnl for t in trades]
    winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
    losing_trades = [pnl for pnl in trade_pnls if pnl < 0]

    total_trades = len(trades)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    win_rate = num_winning / total_trades if total_trades > 0 else 0.0

    # Profit metrics
    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
    profit_factor = calculate_profit_factor(winning_trades, losing_trades)
    expectancy = calculate_expectancy(win_rate, avg_win, avg_loss)

    # Best/worst trades
    best_trade = max(trade_pnls) if trade_pnls else 0.0
    worst_trade = min(trade_pnls) if trade_pnls else 0.0

    # Average trade duration
    durations = []
    for trade in trades:
        if trade.exit_time and trade.entry_time:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
            durations.append(duration)

    avg_duration = np.mean(durations) if durations else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        volatility=volatility,
        downside_deviation=downside_dev,
        total_trades=total_trades,
        winning_trades=num_winning,
        losing_trades=num_losing,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        best_trade=best_trade,
        worst_trade=worst_trade,
        avg_trade_duration=avg_duration,
        start_date=equity_curve.index[0],
        end_date=equity_curve.index[-1],
        total_days=days
    )


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics for failed backtests"""
    return PerformanceMetrics(
        total_return=0.0,
        annualized_return=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_duration=0,
        volatility=0.0,
        downside_deviation=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        best_trade=0.0,
        worst_trade=0.0,
        avg_trade_duration=0.0,
        start_date=pd.Timestamp.now(),
        end_date=pd.Timestamp.now(),
        total_days=0
    )


logger.info("Performance metrics module loaded")
