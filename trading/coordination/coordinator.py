"""
Strategy Coordinator

Central controller for managing multiple strategies simultaneously.
"""
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import asyncio

from ..strategy.base import BaseStrategy
from ..strategy.config import StrategyConfig
from ..events import EventBus
from ..portfolio import PortfolioManager
from ..oms import OrderManager


class StrategyStatus(Enum):
    """Strategy execution status"""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AllocationMode(Enum):
    """Capital allocation mode"""
    FIXED = "fixed"  # Fixed percentage allocation
    DYNAMIC = "dynamic"  # Dynamically adjust based on performance
    EQUAL = "equal"  # Equal allocation to all strategies


@dataclass
class StrategyInstance:
    """Container for strategy instance with metadata"""
    strategy_id: str
    strategy: BaseStrategy
    config: StrategyConfig
    status: StrategyStatus
    allocation: Decimal  # Percentage of capital (0.0 to 1.0)
    capital: Decimal
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    error: Optional[str] = None

    # Performance metrics
    pnl: Decimal = Decimal("0")
    num_trades: int = 0
    win_rate: Decimal = Decimal("0")


class StrategyCoordinator:
    """
    Multi-strategy coordinator

    Manages multiple strategies with:
    - Lifecycle management
    - Capital allocation
    - Risk coordination
    - Performance monitoring
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        event_bus: EventBus,
        total_capital: Optional[Decimal] = None,
        allocation_mode: AllocationMode = AllocationMode.FIXED,
        rebalance_interval_seconds: int = 3600,  # Rebalance every hour
    ):
        """
        Initialize coordinator

        Args:
            portfolio_manager: Portfolio manager instance
            order_manager: Order manager instance
            event_bus: Event bus for communication
            total_capital: Total capital to manage (default: from portfolio)
            allocation_mode: How to allocate capital
            rebalance_interval_seconds: How often to rebalance allocations
        """
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.event_bus = event_bus

        # Get total capital
        if total_capital is None:
            stats = portfolio_manager.get_statistics()
            self.total_capital = stats["current_equity"]
        else:
            self.total_capital = total_capital

        self.allocation_mode = allocation_mode
        self.rebalance_interval_seconds = rebalance_interval_seconds

        # Strategy registry
        self.strategies: Dict[str, StrategyInstance] = {}

        # Coordination state
        self.is_running = False
        self.rebalance_task: Optional[asyncio.Task] = None

        # Statistics
        self.start_time: Optional[datetime] = None
        self.total_trades = 0

        logger.info(
            f"Strategy Coordinator initialized: "
            f"Capital: ${self.total_capital:,.2f}, "
            f"Allocation mode: {allocation_mode.value}"
        )

    async def add_strategy(
        self,
        strategy: BaseStrategy,
        allocation: Optional[Decimal] = None
    ) -> str:
        """
        Add a strategy to coordination

        Args:
            strategy: Strategy instance
            allocation: Capital allocation (0.0-1.0). If None, auto-calculated based on mode.

        Returns:
            Strategy ID
        """
        strategy_id = strategy.config.strategy_id

        if strategy_id in self.strategies:
            raise ValueError(f"Strategy {strategy_id} already exists")

        # Calculate allocation if not provided
        if allocation is None:
            allocation = self._calculate_default_allocation()

        # Calculate capital
        capital = self.total_capital * allocation

        # Create instance
        instance = StrategyInstance(
            strategy_id=strategy_id,
            strategy=strategy,
            config=strategy.config,
            status=StrategyStatus.CREATED,
            allocation=allocation,
            capital=capital
        )

        self.strategies[strategy_id] = instance

        logger.info(
            f"Added strategy: {strategy_id} | "
            f"Allocation: {allocation * 100:.1f}% | "
            f"Capital: ${capital:,.2f}"
        )

        return strategy_id

    async def remove_strategy(self, strategy_id: str) -> None:
        """
        Remove a strategy

        Args:
            strategy_id: Strategy ID to remove
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        instance = self.strategies[strategy_id]

        # Stop if running
        if instance.status == StrategyStatus.RUNNING:
            await self.stop_strategy(strategy_id)

        del self.strategies[strategy_id]

        logger.info(f"Removed strategy: {strategy_id}")

    async def start_strategy(self, strategy_id: str) -> None:
        """Start a specific strategy"""

        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        instance = self.strategies[strategy_id]

        if instance.status == StrategyStatus.RUNNING:
            logger.warning(f"Strategy {strategy_id} already running")
            return

        logger.info(f"Starting strategy: {strategy_id}")

        try:
            instance.status = StrategyStatus.STARTING
            await instance.strategy.start()
            instance.status = StrategyStatus.RUNNING
            instance.start_time = datetime.now()

            logger.info(f"✓ Strategy started: {strategy_id}")

        except Exception as e:
            instance.status = StrategyStatus.ERROR
            instance.error = str(e)
            logger.error(f"Failed to start strategy {strategy_id}: {e}")
            raise

    async def stop_strategy(self, strategy_id: str) -> None:
        """Stop a specific strategy"""

        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        instance = self.strategies[strategy_id]

        if instance.status != StrategyStatus.RUNNING:
            logger.warning(f"Strategy {strategy_id} not running")
            return

        logger.info(f"Stopping strategy: {strategy_id}")

        try:
            instance.status = StrategyStatus.STOPPING
            await instance.strategy.stop()
            instance.status = StrategyStatus.STOPPED
            instance.stop_time = datetime.now()

            logger.info(f"✓ Strategy stopped: {strategy_id}")

        except Exception as e:
            instance.status = StrategyStatus.ERROR
            instance.error = str(e)
            logger.error(f"Failed to stop strategy {strategy_id}: {e}")
            raise

    async def pause_strategy(self, strategy_id: str) -> None:
        """Pause a strategy (keeps it loaded but stops trading)"""

        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        instance = self.strategies[strategy_id]

        if instance.status != StrategyStatus.RUNNING:
            logger.warning(f"Strategy {strategy_id} not running")
            return

        # Disable the strategy config
        instance.config.enabled = False
        instance.status = StrategyStatus.PAUSED

        logger.info(f"Paused strategy: {strategy_id}")

    async def resume_strategy(self, strategy_id: str) -> None:
        """Resume a paused strategy"""

        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        instance = self.strategies[strategy_id]

        if instance.status != StrategyStatus.PAUSED:
            logger.warning(f"Strategy {strategy_id} not paused")
            return

        # Re-enable the strategy config
        instance.config.enabled = True
        instance.status = StrategyStatus.RUNNING

        logger.info(f"Resumed strategy: {strategy_id}")

    async def start_all(self) -> None:
        """Start all strategies"""

        logger.info("=" * 70)
        logger.info(f"Starting all strategies ({len(self.strategies)} total)")
        logger.info("=" * 70)

        self.is_running = True
        self.start_time = datetime.now()

        # Start each strategy
        for strategy_id in self.strategies:
            try:
                await self.start_strategy(strategy_id)
            except Exception as e:
                logger.error(f"Failed to start {strategy_id}: {e}")

        # Start rebalancing task if using dynamic allocation
        if self.allocation_mode == AllocationMode.DYNAMIC:
            self.rebalance_task = asyncio.create_task(self._rebalance_loop())

        logger.info("=" * 70)
        logger.info("All strategies started")
        logger.info("=" * 70)

    async def stop_all(self) -> None:
        """Stop all strategies"""

        logger.info("=" * 70)
        logger.info("Stopping all strategies")
        logger.info("=" * 70)

        self.is_running = False

        # Cancel rebalancing task
        if self.rebalance_task:
            self.rebalance_task.cancel()
            try:
                await self.rebalance_task
            except asyncio.CancelledError:
                pass

        # Stop each strategy
        for strategy_id in list(self.strategies.keys()):
            try:
                await self.stop_strategy(strategy_id)
            except Exception as e:
                logger.error(f"Failed to stop {strategy_id}: {e}")

        logger.info("=" * 70)
        logger.info("All strategies stopped")
        logger.info("=" * 70)

    def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get status of a specific strategy"""

        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        instance = self.strategies[strategy_id]

        return {
            "strategy_id": strategy_id,
            "name": instance.config.name,
            "status": instance.status.value,
            "allocation": float(instance.allocation),
            "capital": float(instance.capital),
            "pnl": float(instance.pnl),
            "num_trades": instance.num_trades,
            "win_rate": float(instance.win_rate),
            "start_time": instance.start_time.isoformat() if instance.start_time else None,
            "error": instance.error
        }

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all strategies"""

        return [
            self.get_strategy_status(strategy_id)
            for strategy_id in self.strategies
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all strategies"""

        total_pnl = sum(instance.pnl for instance in self.strategies.values())
        total_trades = sum(instance.num_trades for instance in self.strategies.values())

        running_count = len([s for s in self.strategies.values() if s.status == StrategyStatus.RUNNING])
        stopped_count = len([s for s in self.strategies.values() if s.status == StrategyStatus.STOPPED])
        error_count = len([s for s in self.strategies.values() if s.status == StrategyStatus.ERROR])

        # Get portfolio stats
        portfolio_stats = self.portfolio_manager.get_statistics()

        return {
            "total_strategies": len(self.strategies),
            "running": running_count,
            "stopped": stopped_count,
            "errors": error_count,
            "total_capital": float(self.total_capital),
            "current_equity": float(portfolio_stats["current_equity"]),
            "total_pnl": float(total_pnl),
            "total_return_pct": float(portfolio_stats["return_pct"]),
            "total_trades": total_trades,
            "allocation_mode": self.allocation_mode.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
        }

    async def rebalance_allocations(self) -> None:
        """Rebalance capital allocations based on performance"""

        if self.allocation_mode != AllocationMode.DYNAMIC:
            logger.debug("Rebalancing skipped (not in dynamic mode)")
            return

        logger.info("Rebalancing capital allocations...")

        # TODO: Implement performance-based rebalancing
        # This would:
        # 1. Calculate performance metrics for each strategy
        # 2. Adjust allocations based on Sharpe ratios or other metrics
        # 3. Notify strategies of new allocation limits

        logger.info("Rebalancing complete")

    async def _rebalance_loop(self) -> None:
        """Background task for periodic rebalancing"""

        while self.is_running:
            try:
                await asyncio.sleep(self.rebalance_interval_seconds)
                await self.rebalance_allocations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rebalance loop: {e}")

    def _calculate_default_allocation(self) -> Decimal:
        """Calculate default allocation for new strategy"""

        if self.allocation_mode == AllocationMode.EQUAL:
            # Equal allocation
            num_strategies = len(self.strategies) + 1
            return Decimal("1.0") / Decimal(str(num_strategies))

        elif self.allocation_mode == AllocationMode.FIXED:
            # Calculate remaining allocation
            used_allocation = sum(s.allocation for s in self.strategies.values())
            remaining = Decimal("1.0") - used_allocation

            if remaining <= 0:
                raise ValueError("No remaining capital allocation available")

            return remaining

        else:
            # Dynamic - start with equal allocation
            num_strategies = len(self.strategies) + 1
            return Decimal("1.0") / Decimal(str(num_strategies))

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""

        return {
            **self.get_performance_summary(),
            "strategies": self.get_all_status()
        }


logger.info("Strategy coordinator module loaded")
