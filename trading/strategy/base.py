"""
Base Strategy Class

Abstract base class for all trading strategies with:
- Event-driven architecture
- Configuration-based setup
- Portfolio integration
- Order management integration
- Automatic state persistence

To create a strategy:
1. Subclass BaseStrategy
2. Implement on_tick(), on_trade(), etc.
3. Use self.submit_order() to place orders
4. Use self.get_position() to query positions

Example:
    class MyStrategy(BaseStrategy):
        async def on_tick(self, tick: TickEvent):
            # Your strategy logic
            if should_buy:
                await self.submit_order(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    quantity=Decimal("0.1")
                )
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from decimal import Decimal
from datetime import datetime
from loguru import logger

from ..events.base import BaseEvent
from ..events.bus import IEventBus
from ..events.market_data import TickEvent, TradeEvent, OrderBookEvent, BarEvent
from ..events.order import OrderSide, OrderType, TimeInForce
from ..exchanges.base import Position, Order

from .config import StrategyConfig


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies

    Provides:
    - Event subscription
    - Order management
    - Position queries
    - State persistence
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: IEventBus,
        portfolio_manager: any,  # PortfolioManager
        order_manager: any,  # OrderManager
    ):
        """
        Initialize strategy

        Args:
            config: Strategy configuration
            event_bus: Event bus
            portfolio_manager: Portfolio manager
            order_manager: Order manager
        """
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio_manager
        self.oms = order_manager

        # Strategy state
        self.is_running = False
        self.strategy_id = config.strategy_id

        # Custom state (for subclasses)
        self.state: Dict = {}

        logger.info(f"Initialized strategy: {config.name} (ID: {self.strategy_id})")

    async def start(self) -> None:
        """
        Start the strategy

        - Subscribe to configured market data
        - Load initial positions if configured
        - Call on_start() hook
        """

        if self.is_running:
            logger.warning(f"Strategy {self.strategy_id} already running")
            return

        logger.info(f"Starting strategy: {self.strategy_id}")

        # Load initial positions if configured
        if self.config.initial_positions_file:
            await self._load_initial_positions()

        # Subscribe to market data
        await self._subscribe_market_data()

        # Call subclass hook
        await self.on_start()

        self.is_running = True
        logger.info(f"Strategy started: {self.strategy_id}")

    async def stop(self) -> None:
        """
        Stop the strategy

        - Unsubscribe from events
        - Save state if configured
        - Call on_stop() hook
        """

        if not self.is_running:
            logger.warning(f"Strategy {self.strategy_id} not running")
            return

        logger.info(f"Stopping strategy: {self.strategy_id}")

        # Call subclass hook
        await self.on_stop()

        # Save state if configured
        if self.config.save_state_on_stop:
            await self._save_state()

        self.is_running = False
        logger.info(f"Strategy stopped: {self.strategy_id}")

    async def _subscribe_market_data(self) -> None:
        """Subscribe to configured market data"""

        sub = self.config.subscriptions

        # Subscribe to ticks
        if "tick" in sub.data_types:
            await self.event_bus.subscribe("tick", self._handle_tick)

        # Subscribe to trades
        if "trade" in sub.data_types:
            await self.event_bus.subscribe("trade", self._handle_trade)

        # Subscribe to order book
        if "orderbook" in sub.data_types:
            await self.event_bus.subscribe("order_book", self._handle_orderbook)

        # Subscribe to bars
        if "bar" in sub.data_types:
            await self.event_bus.subscribe("bar", self._handle_bar)

        logger.debug(
            f"Strategy {self.strategy_id} subscribed to: {sub.data_types} "
            f"for {sub.symbols}"
        )

    async def _handle_tick(self, event: TickEvent) -> None:
        """Handle tick event"""
        # Filter by configured symbols
        if event.symbol not in self.config.subscriptions.symbols:
            return

        # Call strategy implementation
        try:
            await self.on_tick(event)
        except Exception as e:
            logger.error(
                f"Error in {self.strategy_id}.on_tick(): {e}",
                exc_info=True
            )

    async def _handle_trade(self, event: TradeEvent) -> None:
        """Handle trade event"""
        if event.symbol not in self.config.subscriptions.symbols:
            return

        try:
            await self.on_trade(event)
        except Exception as e:
            logger.error(
                f"Error in {self.strategy_id}.on_trade(): {e}",
                exc_info=True
            )

    async def _handle_orderbook(self, event: OrderBookEvent) -> None:
        """Handle order book event"""
        if event.symbol not in self.config.subscriptions.symbols:
            return

        try:
            await self.on_orderbook(event)
        except Exception as e:
            logger.error(
                f"Error in {self.strategy_id}.on_orderbook(): {e}",
                exc_info=True
            )

    async def _handle_bar(self, event: BarEvent) -> None:
        """Handle bar event"""
        if event.symbol not in self.config.subscriptions.symbols:
            return

        try:
            await self.on_bar(event)
        except Exception as e:
            logger.error(
                f"Error in {self.strategy_id}.on_bar(): {e}",
                exc_info=True
            )

    async def _load_initial_positions(self) -> None:
        """Load initial positions from file"""
        from ..portfolio.persistence import seed_portfolio_from_file

        positions_file = self.config.get_positions_file()
        if not positions_file:
            return

        try:
            seed_portfolio_from_file(
                self.portfolio,
                positions_file,
                format="csv"
            )
            logger.info(f"Loaded initial positions for {self.strategy_id}")
        except Exception as e:
            logger.error(f"Failed to load initial positions: {e}")

    async def _save_state(self) -> None:
        """Save strategy state"""
        # TODO: Implement state persistence
        logger.debug(f"Saving state for {self.strategy_id}")

    # Order management methods

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> bool:
        """
        Submit order through OMS

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force

        Returns:
            True if order accepted, False otherwise
        """

        # Add strategy metadata
        metadata = {
            "strategy_id": self.strategy_id,
            "strategy_name": self.config.name,
        }

        result = await self.oms.submit_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            metadata=metadata
        )

        if result.success:
            logger.info(
                f"[{self.strategy_id}] Order submitted: {result.order_id} | "
                f"{side.value} {quantity} {symbol}"
            )
            return True
        else:
            logger.warning(
                f"[{self.strategy_id}] Order failed: {result.message}"
            )
            return False

    # Portfolio query methods

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        return await self.portfolio.get_position(symbol)

    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return await self.portfolio.get_all_positions()

    async def has_position(self, symbol: str) -> bool:
        """Check if position exists"""
        position = await self.get_position(symbol)
        return position is not None and position.quantity != 0

    # Abstract methods - implement in subclass

    @abstractmethod
    async def on_tick(self, tick: TickEvent) -> None:
        """
        Handle tick event

        Override this method with your strategy logic.
        """
        pass

    async def on_trade(self, trade: TradeEvent) -> None:
        """Handle trade event (optional)"""
        pass

    async def on_orderbook(self, orderbook: OrderBookEvent) -> None:
        """Handle order book event (optional)"""
        pass

    async def on_bar(self, bar: BarEvent) -> None:
        """Handle bar/candle event (optional)"""
        pass

    async def on_start(self) -> None:
        """Called when strategy starts (optional)"""
        pass

    async def on_stop(self) -> None:
        """Called when strategy stops (optional)"""
        pass
