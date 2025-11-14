"""
Strategy Manager

Manages multiple strategies running concurrently:
- Load strategies from config files
- Start/stop individual strategies
- Monitor strategy performance
- Isolate strategy state
- Handle strategy lifecycle

Each strategy:
- Has unique ID from config
- Runs independently
- Has own position tracking (via metadata)
- Can be started/stopped individually
"""
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

from ..events.bus import IEventBus
from .base import BaseStrategy
from .config import StrategyConfig


class StrategyManager:
    """
    Manages multiple trading strategies

    Responsibilities:
    - Load strategies from config files
    - Initialize strategy instances
    - Start/stop strategies
    - Monitor strategy health
    - Collect strategy metrics
    """

    def __init__(
        self,
        event_bus: IEventBus,
        portfolio_manager: any,
        order_manager: any,
        strategy_factory: Optional[Dict[str, type]] = None
    ):
        """
        Initialize Strategy Manager

        Args:
            event_bus: Event bus
            portfolio_manager: Portfolio manager
            order_manager: Order manager
            strategy_factory: Map of strategy_type -> Strategy class
        """
        self.event_bus = event_bus
        self.portfolio = portfolio_manager
        self.oms = order_manager
        self.strategy_factory = strategy_factory or {}

        # Active strategies
        self.strategies: Dict[str, BaseStrategy] = {}  # strategy_id -> instance

        logger.info("Initialized StrategyManager")

    def register_strategy_class(
        self,
        strategy_type: str,
        strategy_class: type
    ) -> None:
        """
        Register a strategy class

        Args:
            strategy_type: Type identifier (used in config)
            strategy_class: Strategy class (subclass of BaseStrategy)
        """
        self.strategy_factory[strategy_type] = strategy_class
        logger.info(f"Registered strategy type: {strategy_type}")

    async def load_strategy(
        self,
        config_path: str,
        strategy_class: Optional[type] = None
    ) -> str:
        """
        Load strategy from config file

        Args:
            config_path: Path to config file
            strategy_class: Strategy class (if not using factory)

        Returns:
            Strategy ID
        """

        # Load config
        config = StrategyConfig.from_file(config_path)
        config.validate()

        # Check if already loaded
        if config.strategy_id in self.strategies:
            logger.warning(
                f"Strategy {config.strategy_id} already loaded, skipping"
            )
            return config.strategy_id

        # Get strategy class
        if strategy_class is None:
            # Try to get from factory using name
            if config.name in self.strategy_factory:
                strategy_class = self.strategy_factory[config.name]
            else:
                raise ValueError(
                    f"No strategy class provided and {config.name} not in factory. "
                    f"Available types: {list(self.strategy_factory.keys())}"
                )

        # Create strategy instance
        strategy = strategy_class(
            config=config,
            event_bus=self.event_bus,
            portfolio_manager=self.portfolio,
            order_manager=self.oms
        )

        # Store strategy
        self.strategies[config.strategy_id] = strategy

        logger.info(
            f"Loaded strategy: {config.name} (ID: {config.strategy_id}) "
            f"from {config_path}"
        )

        return config.strategy_id

    async def load_all_from_directory(
        self,
        directory: str,
        pattern: str = "*.yaml"
    ) -> List[str]:
        """
        Load all strategies from directory

        Args:
            directory: Directory containing strategy configs
            pattern: File pattern to match (e.g., "*.yaml", "*.json")

        Returns:
            List of loaded strategy IDs
        """

        dir_path = Path(directory)

        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        config_files = list(dir_path.glob(pattern))
        logger.info(f"Found {len(config_files)} strategy configs in {directory}")

        loaded_ids = []

        for config_file in config_files:
            try:
                strategy_id = await self.load_strategy(str(config_file))
                loaded_ids.append(strategy_id)
            except Exception as e:
                logger.error(f"Failed to load {config_file}: {e}")

        logger.info(f"Successfully loaded {len(loaded_ids)} strategies")

        return loaded_ids

    async def start_strategy(self, strategy_id: str) -> bool:
        """
        Start a specific strategy

        Args:
            strategy_id: Strategy ID to start

        Returns:
            True if started successfully
        """

        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False

        strategy = self.strategies[strategy_id]

        if not strategy.config.enabled:
            logger.warning(f"Strategy {strategy_id} is disabled in config")
            return False

        try:
            await strategy.start()
            logger.info(f"Started strategy: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start strategy {strategy_id}: {e}")
            return False

    async def stop_strategy(self, strategy_id: str) -> bool:
        """
        Stop a specific strategy

        Args:
            strategy_id: Strategy ID to stop

        Returns:
            True if stopped successfully
        """

        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False

        strategy = self.strategies[strategy_id]

        try:
            await strategy.stop()
            logger.info(f"Stopped strategy: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop strategy {strategy_id}: {e}")
            return False

    async def start_all(self) -> int:
        """
        Start all loaded strategies

        Returns:
            Number of strategies started
        """

        logger.info(f"Starting {len(self.strategies)} strategies")

        started = 0
        for strategy_id in self.strategies.keys():
            if await self.start_strategy(strategy_id):
                started += 1

        logger.info(f"Started {started}/{len(self.strategies)} strategies")

        return started

    async def stop_all(self) -> int:
        """
        Stop all running strategies

        Returns:
            Number of strategies stopped
        """

        logger.info(f"Stopping {len(self.strategies)} strategies")

        stopped = 0
        for strategy_id in list(self.strategies.keys()):
            if await self.stop_strategy(strategy_id):
                stopped += 1

        logger.info(f"Stopped {stopped}/{len(self.strategies)} strategies")

        return stopped

    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Get strategy by ID"""
        return self.strategies.get(strategy_id)

    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all strategies"""
        return list(self.strategies.values())

    def get_running_strategies(self) -> List[BaseStrategy]:
        """Get all running strategies"""
        return [s for s in self.strategies.values() if s.is_running]

    def get_statistics(self) -> Dict:
        """Get strategy manager statistics"""

        return {
            "total_strategies": len(self.strategies),
            "running_strategies": len(self.get_running_strategies()),
            "stopped_strategies": len(self.strategies) - len(self.get_running_strategies()),
            "registered_types": list(self.strategy_factory.keys()),
        }

    def list_strategies(self) -> List[Dict]:
        """List all strategies with details"""

        strategies = []

        for strategy_id, strategy in self.strategies.items():
            strategies.append({
                "strategy_id": strategy_id,
                "name": strategy.config.name,
                "version": strategy.config.version,
                "enabled": strategy.config.enabled,
                "running": strategy.is_running,
                "symbols": strategy.config.subscriptions.symbols,
                "paper_trading": strategy.config.paper_trading,
            })

        return strategies
