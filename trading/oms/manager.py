"""
Order Manager

Central orchestrator for order management with:
- Pre-trade risk checks
- Order routing
- Order state tracking
- Event-driven updates
- Order persistence

Integrates:
- RiskChecker: Pre-trade validation
- OrderRouter: Exchange selection
- EventBus: Event-driven architecture
- Portfolio: Position tracking
"""
import asyncio
from typing import Optional, Dict, List, Callable, Awaitable
from decimal import Decimal
from datetime import datetime
import uuid
from loguru import logger

from ..events.base import BaseEvent
from ..events.bus import IEventBus
from ..events.order import (
    OrderSubmittedEvent,
    OrderAcceptedEvent,
    OrderPartiallyFilledEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
    OrderRejectedEvent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from ..exchanges.base import (
    IExchange,
    Order,
    OrderResult,
    OrderStatus,
    Position,
)

from .risk import RiskChecker, RiskViolation
from .router import OrderRouter


class OrderManager:
    """
    Order Management System

    Responsibilities:
    - Accept order submissions
    - Run pre-trade risk checks
    - Route orders to exchanges
    - Track order lifecycle
    - Publish order events
    - Handle order updates from exchanges
    """

    def __init__(
        self,
        event_bus: IEventBus,
        risk_checker: RiskChecker,
        order_router: OrderRouter,
        portfolio_manager: Optional[any] = None  # Forward reference, will be typed properly
    ):
        """
        Initialize Order Manager

        Args:
            event_bus: Event bus for publishing/subscribing
            risk_checker: Risk checker for pre-trade validation
            order_router: Order router for exchange selection
            portfolio_manager: Portfolio manager for position queries
        """
        self.event_bus = event_bus
        self.risk_checker = risk_checker
        self.order_router = order_router
        self.portfolio_manager = portfolio_manager

        # Order tracking
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.active_orders: Dict[str, Order] = {}  # Only active orders

        # State
        self.is_running = False

        logger.info("Initialized OrderManager")

    async def start(self) -> None:
        """Start the Order Manager"""
        if self.is_running:
            logger.warning("OrderManager already running")
            return

        self.is_running = True

        # Subscribe to order events from exchanges
        # This allows OMS to track order state changes
        await self.event_bus.subscribe("order_*", self._handle_order_event)

        logger.info("OrderManager started")

    async def stop(self) -> None:
        """Stop the Order Manager"""
        self.is_running = False
        logger.info("OrderManager stopped")

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        metadata: Optional[Dict] = None
    ) -> OrderResult:
        """
        Submit order for execution

        Flow:
        1. Create order object
        2. Run risk checks
        3. Route to exchange
        4. Submit to exchange
        5. Track order state

        Args:
            symbol: Trading symbol
            side: Buy or sell
            order_type: Market, limit, etc
            quantity: Order quantity
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            metadata: Additional metadata

        Returns:
            OrderResult with success status
        """

        if not self.is_running:
            return OrderResult(
                success=False,
                order_id="",
                message="OrderManager not running"
            )

        # Create order object
        order = Order(
            order_id=f"oms_{uuid.uuid4().hex[:16]}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.PENDING,
            metadata=metadata or {}
        )

        logger.info(
            f"Submitting order: {order.order_id} | {side.value} {quantity} {symbol} "
            f"@ {limit_price or 'MARKET'}"
        )

        # Track order
        self.orders[order.order_id] = order
        self.active_orders[order.order_id] = order

        # Publish submitted event
        await self._publish_order_event(OrderSubmittedEvent, order)

        try:
            # Step 1: Risk checks
            violations = await self._check_risk(order)

            if self.risk_checker.has_errors(violations):
                # Risk check failed
                error_msgs = [v.message for v in violations if v.severity == "error"]
                message = "; ".join(error_msgs)

                logger.warning(f"Order rejected by risk checks: {message}")

                order.status = OrderStatus.REJECTED
                order.reject_reason = message

                await self._publish_order_event(OrderRejectedEvent, order)

                return OrderResult(
                    success=False,
                    order_id=order.order_id,
                    message=f"Risk check failed: {message}"
                )

            # Step 2: Route order
            routing = await self.order_router.route_order(order)

            if not routing:
                logger.error("Failed to route order - no suitable exchange")

                order.status = OrderStatus.REJECTED
                order.reject_reason = "No suitable exchange"

                await self._publish_order_event(OrderRejectedEvent, order)

                return OrderResult(
                    success=False,
                    order_id=order.order_id,
                    message="No suitable exchange for routing"
                )

            logger.debug(f"Routed order to {routing.exchange_name}: {routing.reason}")

            # Step 3: Submit to exchange
            exchange = routing.exchange
            execution_connector = exchange.get_execution_connector()

            result = await execution_connector.place_order(order)

            if result.success:
                # Order accepted by exchange
                order.status = OrderStatus.ACCEPTED
                order.exchange_order_id = result.exchange_order_id

                logger.info(
                    f"Order accepted by exchange: {order.order_id} -> "
                    f"{result.exchange_order_id}"
                )

                await self._publish_order_event(OrderAcceptedEvent, order)

                return OrderResult(
                    success=True,
                    order_id=order.order_id,
                    exchange_order_id=result.exchange_order_id,
                    message="Order accepted"
                )
            else:
                # Order rejected by exchange
                logger.warning(
                    f"Order rejected by exchange: {result.message}"
                )

                order.status = OrderStatus.REJECTED
                order.reject_reason = result.message

                await self._publish_order_event(OrderRejectedEvent, order)

                return OrderResult(
                    success=False,
                    order_id=order.order_id,
                    message=f"Exchange rejected: {result.message}"
                )

        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=True)

            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)

            await self._publish_order_event(OrderRejectedEvent, order)

            return OrderResult(
                success=False,
                order_id=order.order_id,
                message=f"Error: {str(e)}"
            )

    async def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel an order

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResult
        """

        if order_id not in self.orders:
            return OrderResult(
                success=False,
                order_id=order_id,
                message="Order not found"
            )

        order = self.orders[order_id]

        if order.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
            return OrderResult(
                success=False,
                order_id=order_id,
                message=f"Cannot cancel order in state: {order.status.value}"
            )

        logger.info(f"Cancelling order: {order_id}")

        # Determine which exchange has the order
        # (In production, track this mapping)
        # For now, try all exchanges

        for exchange_name in self.order_router.get_available_exchanges():
            exchange = self.order_router.exchanges[exchange_name]
            execution = exchange.get_execution_connector()

            result = await execution.cancel_order(order.exchange_order_id or order_id)

            if result.success:
                order.status = OrderStatus.CANCELLED

                await self._publish_order_event(OrderCancelledEvent, order)

                logger.info(f"Order cancelled: {order_id}")

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    message="Order cancelled"
                )

        return OrderResult(
            success=False,
            order_id=order_id,
            message="Failed to cancel order"
        )

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    async def get_active_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get all active orders

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active orders
        """
        orders = list(self.active_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """
        Get order history

        Args:
            symbol: Filter by symbol (optional)
            limit: Max number of orders to return

        Returns:
            List of orders
        """
        orders = list(self.orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        # Sort by submission time, most recent first
        orders.sort(key=lambda o: o.submitted_time or datetime.min, reverse=True)

        return orders[:limit]

    async def _check_risk(self, order: Order) -> List[RiskViolation]:
        """Run risk checks on order"""

        # Get current positions from portfolio manager
        if self.portfolio_manager:
            current_positions = await self.portfolio_manager.get_all_positions()
            account_state = await self.portfolio_manager.get_account_state()
        else:
            # No portfolio manager, use empty state
            current_positions = {}
            account_state = {
                "balance": Decimal("100000"),
                "equity": Decimal("100000"),
                "used_margin": Decimal("0"),
            }

        # Run checks
        violations = await self.risk_checker.check_order(
            order,
            current_positions,
            account_state
        )

        return violations

    async def _handle_order_event(self, event: BaseEvent) -> None:
        """
        Handle order events from exchanges

        Updates local order state based on events.
        """

        # Extract order ID from event
        order_id = getattr(event, "order_id", None)
        if not order_id:
            return

        if order_id not in self.orders:
            # Not our order, ignore
            return

        order = self.orders[order_id]

        # Update order state based on event type
        if event.event_type == "order_accepted":
            order.status = OrderStatus.ACCEPTED

        elif event.event_type == "order_partially_filled":
            order.status = OrderStatus.PARTIALLY_FILLED
            if hasattr(event, "filled_quantity"):
                order.filled_quantity = event.filled_quantity
            if hasattr(event, "average_filled_price"):
                order.average_filled_price = event.average_filled_price

        elif event.event_type == "order_filled":
            order.status = OrderStatus.FILLED
            order.filled_time = datetime.utcnow()
            if hasattr(event, "filled_quantity"):
                order.filled_quantity = event.filled_quantity
            if hasattr(event, "average_filled_price"):
                order.average_filled_price = event.average_filled_price

            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]

        elif event.event_type == "order_cancelled":
            order.status = OrderStatus.CANCELLED

            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]

        elif event.event_type == "order_rejected":
            order.status = OrderStatus.REJECTED
            if hasattr(event, "reason"):
                order.reject_reason = event.reason

            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]

        logger.debug(f"Updated order {order_id} state: {order.status.value}")

    async def _publish_order_event(
        self,
        event_class: type,
        order: Order
    ) -> None:
        """Publish order event to event bus"""

        event = event_class(
            order_id=order.order_id,
            exchange=order.metadata.get("exchange", "unknown"),
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            exchange_order_id=order.exchange_order_id or "",
        )

        await self.event_bus.publish(event)

    def get_statistics(self) -> Dict[str, any]:
        """Get OMS statistics"""
        total_orders = len(self.orders)
        active_orders = len(self.active_orders)

        filled = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        cancelled = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        rejected = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])

        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled,
            "cancelled_orders": cancelled,
            "rejected_orders": rejected,
            "fill_rate": filled / total_orders if total_orders > 0 else 0,
        }
