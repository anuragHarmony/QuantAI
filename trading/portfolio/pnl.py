"""
P&L Calculator

Calculates profit and loss metrics:
- Realized P&L (from closed trades)
- Unrealized P&L (from open positions)
- Total P&L
- Return on capital
- Sharpe ratio (for backtests)
- Max drawdown

Also tracks:
- Win rate
- Average win/loss
- Profit factor
"""
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from ..exchanges.base import Position


@dataclass
class PnLMetrics:
    """P&L performance metrics"""

    # Core P&L
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal

    # Returns
    return_pct: Decimal  # % return on initial capital
    return_on_capital: Decimal  # % return on deployed capital

    # Trade statistics
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # Win/Loss analysis
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: Decimal  # Gross profit / gross loss

    # Risk metrics
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    sharpe_ratio: Optional[Decimal] = None

    # Costs
    total_fees: Decimal
    net_pnl: Decimal  # P&L after fees


class PnLCalculator:
    """
    Calculates P&L and performance metrics

    Tracks:
    - All closed trades
    - Historical equity curve
    - Drawdown tracking
    """

    def __init__(self, initial_capital: Decimal):
        """
        Initialize P&L calculator

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Trade history
        self.closed_trades: List[Dict] = []

        # Equity curve
        self.equity_history: List[Tuple[datetime, Decimal]] = [
            (datetime.utcnow(), initial_capital)
        ]

        # Peak tracking for drawdown
        self.peak_equity = initial_capital
        self.max_drawdown = Decimal("0")
        self.max_drawdown_pct = Decimal("0")

        logger.info(f"Initialized PnLCalculator: initial_capital={initial_capital}")

    def record_trade(
        self,
        symbol: str,
        quantity: Decimal,
        entry_price: Decimal,
        exit_price: Decimal,
        fees: Decimal,
        timestamp: Optional[datetime] = None
    ) -> Decimal:
        """
        Record a closed trade

        Args:
            symbol: Trading symbol
            quantity: Trade quantity (signed)
            entry_price: Entry price
            exit_price: Exit price
            fees: Total fees paid
            timestamp: Trade timestamp

        Returns:
            Trade P&L
        """

        # Calculate P&L
        if quantity > 0:
            # Long trade
            pnl = (exit_price - entry_price) * quantity - fees
        else:
            # Short trade
            pnl = (entry_price - exit_price) * abs(quantity) - fees

        # Record trade
        trade = {
            "timestamp": timestamp or datetime.utcnow(),
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "fees": fees,
            "return_pct": (pnl / (entry_price * abs(quantity))) * Decimal("100") if entry_price > 0 else Decimal("0")
        }

        self.closed_trades.append(trade)

        logger.debug(
            f"Trade recorded: {symbol} | "
            f"{'LONG' if quantity > 0 else 'SHORT'} {abs(quantity)} | "
            f"Entry: {entry_price} -> Exit: {exit_price} | "
            f"P&L: {pnl}"
        )

        return pnl

    def update_equity(
        self,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update equity curve

        Args:
            realized_pnl: Cumulative realized P&L
            unrealized_pnl: Current unrealized P&L
            timestamp: Update timestamp
        """

        total_equity = self.initial_capital + realized_pnl + unrealized_pnl
        self.current_capital = total_equity

        # Record in equity curve
        self.equity_history.append(
            (timestamp or datetime.utcnow(), total_equity)
        )

        # Update peak and drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        current_drawdown = self.peak_equity - total_equity
        current_drawdown_pct = (current_drawdown / self.peak_equity) * Decimal("100") \
            if self.peak_equity > 0 else Decimal("0")

        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            self.max_drawdown_pct = current_drawdown_pct

    def calculate_metrics(
        self,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        total_fees: Decimal
    ) -> PnLMetrics:
        """
        Calculate comprehensive P&L metrics

        Args:
            realized_pnl: Realized P&L
            unrealized_pnl: Unrealized P&L
            total_fees: Total fees paid

        Returns:
            PnLMetrics
        """

        total_pnl = realized_pnl + unrealized_pnl
        net_pnl = total_pnl - total_fees

        # Return calculations
        return_pct = (net_pnl / self.initial_capital) * Decimal("100") \
            if self.initial_capital > 0 else Decimal("0")

        # Deployed capital (simplified - could be more sophisticated)
        deployed_capital = self.initial_capital
        return_on_capital = (net_pnl / deployed_capital) * Decimal("100") \
            if deployed_capital > 0 else Decimal("0")

        # Trade statistics
        winning_trades = [t for t in self.closed_trades if t["pnl"] > 0]
        losing_trades = [t for t in self.closed_trades if t["pnl"] < 0]

        num_trades = len(self.closed_trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)

        win_rate = Decimal(num_winning) / Decimal(num_trades) * Decimal("100") \
            if num_trades > 0 else Decimal("0")

        # Win/Loss analysis
        if winning_trades:
            total_wins = sum(t["pnl"] for t in winning_trades)
            average_win = total_wins / Decimal(len(winning_trades))
            largest_win = max(t["pnl"] for t in winning_trades)
        else:
            total_wins = Decimal("0")
            average_win = Decimal("0")
            largest_win = Decimal("0")

        if losing_trades:
            total_losses = sum(abs(t["pnl"]) for t in losing_trades)
            average_loss = total_losses / Decimal(len(losing_trades))
            largest_loss = max(abs(t["pnl"]) for t in losing_trades)
        else:
            total_losses = Decimal("0")
            average_loss = Decimal("0")
            largest_loss = Decimal("0")

        # Profit factor
        profit_factor = total_wins / total_losses if total_losses > 0 else Decimal("999")

        # Sharpe ratio (if sufficient data)
        sharpe = self._calculate_sharpe() if len(self.equity_history) > 10 else None

        return PnLMetrics(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            return_pct=return_pct,
            return_on_capital=return_on_capital,
            num_trades=num_trades,
            winning_trades=num_winning,
            losing_trades=num_losing,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=self.max_drawdown_pct,
            sharpe_ratio=sharpe,
            total_fees=total_fees,
            net_pnl=net_pnl,
        )

    def _calculate_sharpe(self, risk_free_rate: Decimal = Decimal("0")) -> Decimal:
        """
        Calculate Sharpe ratio from equity curve

        Args:
            risk_free_rate: Annual risk-free rate (default 0%)

        Returns:
            Sharpe ratio
        """

        if len(self.equity_history) < 2:
            return Decimal("0")

        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_history)):
            prev_equity = self.equity_history[i-1][1]
            curr_equity = self.equity_history[i][1]

            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(float(ret))

        if not returns:
            return Decimal("0")

        # Calculate mean and std
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return Decimal("0")

        # Sharpe ratio (simplified - not annualized)
        sharpe = mean_return / std_dev

        return Decimal(str(sharpe))

    def get_equity_curve(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Tuple[datetime, Decimal]]:
        """
        Get equity curve for date range

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of (timestamp, equity) tuples
        """

        curve = self.equity_history

        if start:
            curve = [(ts, eq) for ts, eq in curve if ts >= start]
        if end:
            curve = [(ts, eq) for ts, eq in curve if ts <= end]

        return curve

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get trade history

        Args:
            symbol: Filter by symbol
            limit: Max trades to return

        Returns:
            List of trade dicts
        """

        trades = self.closed_trades

        if symbol:
            trades = [t for t in trades if t["symbol"] == symbol]

        # Most recent first
        trades = sorted(trades, key=lambda t: t["timestamp"], reverse=True)

        return trades[:limit]
