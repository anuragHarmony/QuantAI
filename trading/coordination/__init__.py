"""
Multi-Strategy Coordination System

Manages multiple strategies running simultaneously:
- Strategy lifecycle management (start, stop, pause, resume)
- Capital allocation across strategies
- Portfolio-level risk management
- Performance monitoring and comparison
- Resource sharing and coordination
- Dynamic strategy enablement

Usage:
    from trading.coordination import StrategyCoordinator

    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio,
        order_manager=oms,
        event_bus=event_bus
    )

    # Add strategies
    await coordinator.add_strategy(strategy1, allocation=0.4)  # 40% capital
    await coordinator.add_strategy(strategy2, allocation=0.6)  # 60% capital

    # Start all strategies
    await coordinator.start_all()

    # Monitor performance
    performance = coordinator.get_performance_summary()
"""

from .coordinator import (
    StrategyCoordinator,
    StrategyInstance,
    StrategyStatus,
    AllocationMode,
)

from .allocator import (
    CapitalAllocator,
    FixedAllocation,
    DynamicAllocation,
    PerformanceBasedAllocation,
)

from .monitor import (
    StrategyMonitor,
    StrategyMetrics,
)

__all__ = [
    "StrategyCoordinator",
    "StrategyInstance",
    "StrategyStatus",
    "AllocationMode",
    "CapitalAllocator",
    "FixedAllocation",
    "DynamicAllocation",
    "PerformanceBasedAllocation",
    "StrategyMonitor",
    "StrategyMetrics",
]
