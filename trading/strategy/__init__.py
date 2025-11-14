"""
Strategy Framework

Event-driven strategy framework with:
- Configuration-based strategy definition
- Multiple strategies running concurrently
- Strategy-level position tracking
- Event subscription helpers
- Portfolio queries

Components:
- StrategyConfig: Configuration schema
- BaseStrategy: Abstract strategy base class
- StrategyManager: Manages multiple strategies

Usage:
    # Define strategy in config file
    config = StrategyConfig.from_file("strategies/my_strategy.yaml")

    # Create and run strategy
    strategy = MyStrategy(config, event_bus, portfolio, oms)
    await strategy.start()

    # Or use StrategyManager for multiple strategies
    manager = StrategyManager(event_bus, portfolio, oms)
    await manager.load_strategy("strategies/strategy1.yaml")
    await manager.load_strategy("strategies/strategy2.yaml")
    await manager.start_all()
"""

from .config import StrategyConfig
from .base import BaseStrategy
from .manager import StrategyManager

__all__ = [
    "StrategyConfig",
    "BaseStrategy",
    "StrategyManager",
]
