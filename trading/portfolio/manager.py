"""
Portfolio Manager

Orchestrates portfolio tracking with:
- Position tracking
- P&L calculation
- Real-time updates via events
- Risk limit monitoring
- Portfolio value tracking

Event-driven:
- Subscribes to order filled events
- Subscribes to market data for P&L updates
- Publishes portfolio events
"""
import asyncio
from typing import Dict, Optional, List
from decimal import Decimal
from datetime import datetime
from loguru import logger

from ..events.base import BaseEvent
from ..events.bus import IEventBus
from ..events.order import OrderFilledEvent, OrderPartiallyFilledEvent, OrderSide
from ..events.market_data import TickEvent
from ..events.position import (
    PositionOpenedEvent,
    PositionModifiedEvent,
    PositionClosedEvent,
    PortfolioValueEvent,
)
from ..exchanges.base import Position

from .position import PositionTracker
from .pnl import PnLCalculator, PnLMetrics


class PortfolioManager:
    """
    Portfolio Manager

    Responsibilities:
    - Track all positions in real-time
    - Calculate P&L continuously
    - Update on order fills
    - Update on market data changes
    - Publish portfolio events
    - Provide portfolio queries for OMS
    """

    def __init__(
        self,
        event_bus: IEventBus,
        initial_capital: Decimal,
        update_interval: float = 1.0  # How often to publish portfolio updates
    ):
        """
        Initialize Portfolio Manager

        Args:
            event_bus: Event bus for pub/sub
            initial_capital: Starting capital
            update_interval: Seconds between portfolio value updates
        """
        self.event_bus = event_bus
        self.initial_capital = initial_capital
        self.update_interval = update_interval

        # Components
        self.position_tracker = PositionTracker()
        self.pnl_calculator = PnLCalculator(initial_capital)

        # Latest market prices
        self.latest_prices: Dict[str, Decimal] = {}

        # State
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None

        logger.info(f"Initialized PortfolioManager: initial_capital={initial_capital}")

    async def start(self) -> None:
        """Start the Portfolio Manager"""
        if self.is_running:
            logger.warning("PortfolioManager already running")
            return

        self.is_running = True

        # Subscribe to events
        await self.event_bus.subscribe("order_filled", self._handle_order_filled)
        await self.event_bus.subscribe("order_partially_filled", self._handle_partial_fill)
        await self.event_bus.subscribe("tick", self._handle_tick)

        # Start periodic update task
        self._update_task = asyncio.create_task(self._update_loop())

        logger.info("PortfolioManager started")

    async def stop(self) -> None:
        """Stop the Portfolio Manager"""
        self.is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("PortfolioManager stopped")

    async def _handle_order_filled(self, event: OrderFilledEvent) -> None:
        """Handle order filled event"""

        logger.debug(
            f"Processing fill: {event.symbol} | "
            f"{event.side.value} {event.quantity} @ {event.average_filled_price}"
        )

        # Update position
        position = self.position_tracker.update_from_fill(
            symbol=event.symbol,
            exchange=event.exchange,
            side=event.side,
            quantity=event.quantity,
            price=event.average_filled_price,
            fees=event.commission
        )

        # Publish position event
        if not self.position_tracker.has_position(event.symbol):
            # Position was closed
            await self._publish_position_closed(event.symbol, position)
        elif position.quantity == event.quantity:
            # New position
            await self._publish_position_opened(event.symbol, position)
        else:
            # Position modified
            await self._publish_position_modified(event.symbol, position)

        # Update P&L
        await self._update_pnl()

    async def _handle_partial_fill(self, event: OrderPartiallyFilledEvent) -> None:
        """Handle partial fill event"""

        # Treat same as full fill but with partial quantity
        logger.debug(
            f"Processing partial fill: {event.symbol} | "
            f"{event.side.value} {event.filled_quantity} @ {event.average_filled_price}"
        )

        # Update position
        position = self.position_tracker.update_from_fill(
            symbol=event.symbol,
            exchange=event.exchange,
            side=event.side,
            quantity=event.filled_quantity,
            price=event.average_filled_price,
            fees=event.commission
        )

        # Publish position modified event
        await self._publish_position_modified(event.symbol, position)

        # Update P&L
        await self._update_pnl()

    async def _handle_tick(self, event: TickEvent) -> None:
        """Handle market tick - update unrealized P&L"""

        # Update latest price
        if event.last:
            self.latest_prices[event.symbol] = event.last

            # Update position's unrealized P&L
            self.position_tracker.update_market_prices(event.symbol, event.last)

    async def _update_loop(self) -> None:
        """Periodic update loop"""

        while self.is_running:
            try:
                await asyncio.sleep(self.update_interval)

                # Publish portfolio value update
                await self._publish_portfolio_value()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in portfolio update loop: {e}", exc_info=True)

    async def _update_pnl(self) -> None:
        """Update P&L calculator"""

        realized_pnl = self.position_tracker.get_total_realized_pnl()
        unrealized_pnl = self.position_tracker.get_total_unrealized_pnl()

        self.pnl_calculator.update_equity(realized_pnl, unrealized_pnl)

    async def _publish_position_opened(self, symbol: str, position) -> None:
        """Publish position opened event"""

        event = PositionOpenedEvent(
            exchange=position.exchange,
            symbol=symbol,
            side="long" if position.quantity > 0 else "short",
            quantity=abs(position.quantity),
            entry_price=position.average_entry_price,
        )

        await self.event_bus.publish(event)

        logger.info(f"Position opened: {symbol} | {event.side} {event.quantity}")

    async def _publish_position_modified(self, symbol: str, position) -> None:
        """Publish position modified event"""

        pos_obj = position.to_position() if hasattr(position, 'to_position') else position

        event = PositionModifiedEvent(
            exchange=pos_obj.exchange,
            symbol=symbol,
            side="long" if pos_obj.quantity > 0 else "short",
            quantity=abs(pos_obj.quantity),
            average_price=pos_obj.average_entry_price,
            unrealized_pnl=pos_obj.unrealized_pnl,
        )

        await self.event_bus.publish(event)

    async def _publish_position_closed(self, symbol: str, position) -> None:
        """Publish position closed event"""

        pos_obj = position.to_position() if hasattr(position, 'to_position') else position

        event = PositionClosedEvent(
            exchange=pos_obj.exchange,
            symbol=symbol,
            realized_pnl=pos_obj.realized_pnl,
        )

        await self.event_bus.publish(event)

        logger.info(f"Position closed: {symbol} | P&L: {pos_obj.realized_pnl}")

    async def _publish_portfolio_value(self) -> None:
        """Publish portfolio value update"""

        realized_pnl = self.position_tracker.get_total_realized_pnl()
        unrealized_pnl = self.position_tracker.get_total_unrealized_pnl()
        total_pnl = realized_pnl + unrealized_pnl

        total_equity = self.initial_capital + total_pnl

        event = PortfolioValueEvent(
            exchange="portfolio",
            symbol="PORTFOLIO",
            total_value=total_equity,
            cash_balance=self.initial_capital + realized_pnl,  # Simplified
            positions_value=unrealized_pnl,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )

        await self.event_bus.publish(event)

    # Query methods for OMS

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.position_tracker.get_position(symbol)

    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.position_tracker.get_all_positions()

    async def get_account_state(self) -> Dict:
        """
        Get account state for risk checks

        Returns dict with:
        - balance: Cash balance
        - equity: Total equity (balance + unrealized P&L)
        - used_margin: Margin in use
        - available_margin: Available for new positions
        """

        realized_pnl = self.position_tracker.get_total_realized_pnl()
        unrealized_pnl = self.position_tracker.get_total_unrealized_pnl()

        balance = self.initial_capital + realized_pnl
        equity = balance + unrealized_pnl

        # Simplified margin calculation
        # In production, calculate based on actual position requirements
        used_margin = abs(unrealized_pnl)  # Placeholder
        available_margin = equity - used_margin

        return {
            "balance": balance,
            "equity": equity,
            "used_margin": used_margin,
            "available_margin": available_margin,
        }

    async def get_metrics(self) -> PnLMetrics:
        """Get comprehensive P&L metrics"""

        realized_pnl = self.position_tracker.get_total_realized_pnl()
        unrealized_pnl = self.position_tracker.get_total_unrealized_pnl()
        total_fees = self.position_tracker.get_total_fees()

        return self.pnl_calculator.calculate_metrics(
            realized_pnl,
            unrealized_pnl,
            total_fees
        )

    async def get_equity_curve(self) -> List:
        """Get equity curve"""
        return self.pnl_calculator.get_equity_curve()

    def get_statistics(self) -> Dict:
        """Get portfolio statistics"""

        position_stats = self.position_tracker.get_statistics()

        realized_pnl = self.position_tracker.get_total_realized_pnl()
        unrealized_pnl = self.position_tracker.get_total_unrealized_pnl()
        total_pnl = realized_pnl + unrealized_pnl

        total_equity = self.initial_capital + total_pnl
        return_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else Decimal("0")

        return {
            **position_stats,
            "initial_capital": self.initial_capital,
            "current_equity": total_equity,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "max_drawdown": self.pnl_calculator.max_drawdown,
            "max_drawdown_pct": self.pnl_calculator.max_drawdown_pct,
        }
