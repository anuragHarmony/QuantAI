"""
Order matching engine for simulation

Simulates how orders would be filled on a real exchange:
- Market orders: Fill immediately at best available price
- Limit orders: Fill when price crosses limit
- Stop orders: Trigger when price crosses stop
- Partial fills: Based on available liquidity

Supports:
- Configurable fill models (aggressive, conservative, realistic)
- Liquidity modeling
- Price impact simulation
"""
from decimal import Decimal
from typing import Optional, Literal
from datetime import datetime, timedelta
from loguru import logger

from .base import Order, OrderType, OrderSide, OrderStatus
from ..events.market_data import TickEvent, OrderBookEvent


FillModel = Literal["aggressive", "conservative", "realistic"]


class OrderMatcher:
    """
    Simulates order matching based on market data

    Fill Models:
    - aggressive: Fill at best available price immediately
    - conservative: Fill only if price moves beyond limit
    - realistic: Fill with some slippage and partial fills
    """

    def __init__(
        self,
        fill_model: FillModel = "realistic",
        min_fill_ratio: float = 0.3,  # Min % of order that can fill
        latency_ms: int = 50,  # Simulated execution latency
    ):
        """
        Initialize order matcher

        Args:
            fill_model: How aggressively to fill orders
            min_fill_ratio: Minimum ratio for partial fills (0.0-1.0)
            latency_ms: Simulated execution latency in milliseconds
        """
        self.fill_model = fill_model
        self.min_fill_ratio = min_fill_ratio
        self.latency_ms = latency_ms

        logger.info(
            f"Initialized OrderMatcher: model={fill_model}, "
            f"latency={latency_ms}ms"
        )

    def check_fill(
        self,
        order: Order,
        tick: Optional[TickEvent] = None,
        orderbook: Optional[OrderBookEvent] = None
    ) -> tuple[bool, Optional[Decimal], Optional[Decimal]]:
        """
        Check if order should fill and at what price/quantity

        Args:
            order: Order to check
            tick: Current tick data
            orderbook: Current order book (if available)

        Returns:
            Tuple of (should_fill, fill_price, fill_quantity)
        """
        if not tick:
            return False, None, None

        # Market orders always fill immediately
        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, tick, orderbook)

        # Limit orders fill when price crosses
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit_order(order, tick, orderbook)

        # Stop orders trigger then fill as market
        elif order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LIMIT):
            return self._fill_stop_order(order, tick, orderbook)

        return False, None, None

    def _fill_market_order(
        self,
        order: Order,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent]
    ) -> tuple[bool, Optional[Decimal], Optional[Decimal]]:
        """Fill market order at best available price"""

        # Use orderbook if available for more realistic fills
        if orderbook and orderbook.bids and orderbook.asks:
            if order.side == OrderSide.BUY:
                # Buy at best ask
                best_ask = orderbook.asks[0].price
                fill_price = best_ask
            else:
                # Sell at best bid
                best_bid = orderbook.bids[0].price
                fill_price = best_bid
        else:
            # Fall back to tick prices
            if order.side == OrderSide.BUY:
                fill_price = tick.ask or tick.last
            else:
                fill_price = tick.bid or tick.last

        if not fill_price:
            logger.warning(f"No price available to fill market order: {order.order_id}")
            return False, None, None

        # Determine fill quantity based on model
        fill_qty = self._determine_fill_quantity(
            order.quantity,
            orderbook,
            order.side
        )

        return True, fill_price, fill_qty

    def _fill_limit_order(
        self,
        order: Order,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent]
    ) -> tuple[bool, Optional[Decimal], Optional[Decimal]]:
        """Fill limit order if price crosses limit"""

        if not order.limit_price:
            return False, None, None

        # Get relevant market price
        if order.side == OrderSide.BUY:
            # Buy limit fills when ask <= limit
            market_price = tick.ask or tick.last
            can_fill = market_price and market_price <= order.limit_price
            fill_price = min(market_price, order.limit_price) if can_fill else None
        else:
            # Sell limit fills when bid >= limit
            market_price = tick.bid or tick.last
            can_fill = market_price and market_price >= order.limit_price
            fill_price = max(market_price, order.limit_price) if can_fill else None

        if not can_fill or not fill_price:
            return False, None, None

        # Apply fill model
        if self.fill_model == "conservative":
            # Only fill if price moves beyond limit
            if order.side == OrderSide.BUY:
                if not (market_price < order.limit_price):
                    return False, None, None
            else:
                if not (market_price > order.limit_price):
                    return False, None, None

        # Determine fill quantity
        fill_qty = self._determine_fill_quantity(
            order.quantity,
            orderbook,
            order.side
        )

        return True, fill_price, fill_qty

    def _fill_stop_order(
        self,
        order: Order,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent]
    ) -> tuple[bool, Optional[Decimal], Optional[Decimal]]:
        """Fill stop order if price crosses stop"""

        if not order.stop_price:
            return False, None, None

        # Check if stop triggered
        market_price = tick.last
        if not market_price:
            return False, None, None

        triggered = False
        if order.side == OrderSide.BUY:
            # Buy stop triggers when price rises above stop
            triggered = market_price >= order.stop_price
        else:
            # Sell stop triggers when price falls below stop
            triggered = market_price <= order.stop_price

        if not triggered:
            return False, None, None

        # Once triggered, fill as market or limit
        if order.order_type == OrderType.STOP_LIMIT:
            # Fill at limit price if available
            return self._fill_limit_order(order, tick, orderbook)
        else:
            # Fill as market
            return self._fill_market_order(order, tick, orderbook)

    def _determine_fill_quantity(
        self,
        order_qty: Decimal,
        orderbook: Optional[OrderBookEvent],
        side: OrderSide
    ) -> Decimal:
        """Determine how much of order fills based on liquidity"""

        if self.fill_model == "aggressive":
            # Always fill completely
            return order_qty

        if not orderbook:
            # No orderbook, use model-based fills
            if self.fill_model == "realistic":
                # 70% chance of full fill, 30% chance of partial
                import random
                if random.random() < 0.7:
                    return order_qty
                else:
                    # Partial fill between min_fill_ratio and 1.0
                    fill_ratio = Decimal(
                        str(random.uniform(self.min_fill_ratio, 1.0))
                    )
                    return order_qty * fill_ratio
            else:  # conservative
                # Always partial fills
                import random
                fill_ratio = Decimal(
                    str(random.uniform(self.min_fill_ratio, 0.8))
                )
                return order_qty * fill_ratio

        # Calculate available liquidity from orderbook
        levels = orderbook.asks if side == OrderSide.BUY else orderbook.bids

        if not levels:
            return order_qty

        available_qty = sum(level.quantity for level in levels[:3])  # Top 3 levels

        if available_qty >= order_qty:
            # Sufficient liquidity - full fill
            return order_qty
        else:
            # Insufficient liquidity - partial fill
            fill_qty = min(order_qty, available_qty * Decimal("0.8"))
            if fill_qty < order_qty * Decimal(str(self.min_fill_ratio)):
                # Not enough for minimum fill
                return Decimal("0")
            return fill_qty

    def get_fill_timestamp(self, order_timestamp: datetime) -> datetime:
        """Get timestamp for fill (adds simulated latency)"""
        return order_timestamp + timedelta(milliseconds=self.latency_ms)
