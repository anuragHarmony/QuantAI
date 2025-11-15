"""
Portfolio Manager

Tracks and manages portfolio state with:
- Real-time position tracking
- P&L calculation (realized/unrealized)
- Risk limit enforcement
- Portfolio value updates
- Event-driven updates
- Automatic PNL file recording

Components:
- PortfolioManager: Main portfolio orchestrator
- PositionTracker: Tracks individual positions
- PnLCalculator: Calculates profit and loss
- PnLRecorder: Records PNL to file periodically

Usage:
    portfolio = PortfolioManager(event_bus, initial_capital)
    await portfolio.start()

    # Optional: Record PNL to file every 10 seconds
    pnl_recorder = PnLRecorder(event_bus, dump_interval=10.0)
    await pnl_recorder.start()

    # Portfolio updates automatically via events
    positions = await portfolio.get_all_positions()
    pnl = await portfolio.get_total_pnl()
"""

from .position import PositionTracker
from .pnl import PnLCalculator
from .manager import PortfolioManager
from .pnl_recorder import PnLRecorder, create_pnl_recorder

__all__ = [
    "PositionTracker",
    "PnLCalculator",
    "PortfolioManager",
    "PnLRecorder",
    "create_pnl_recorder",
]
