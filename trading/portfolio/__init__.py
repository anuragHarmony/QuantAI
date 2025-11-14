"""
Portfolio Manager

Tracks and manages portfolio state with:
- Real-time position tracking
- P&L calculation (realized/unrealized)
- Risk limit enforcement
- Portfolio value updates
- Event-driven updates

Components:
- PortfolioManager: Main portfolio orchestrator
- PositionTracker: Tracks individual positions
- PnLCalculator: Calculates profit and loss

Usage:
    portfolio = PortfolioManager(event_bus, initial_capital)
    await portfolio.start()

    # Portfolio updates automatically via events
    positions = await portfolio.get_all_positions()
    pnl = await portfolio.get_total_pnl()
"""

from .position import PositionTracker
from .pnl import PnLCalculator
from .manager import PortfolioManager

__all__ = [
    "PositionTracker",
    "PnLCalculator",
    "PortfolioManager",
]
