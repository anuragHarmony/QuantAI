"""
Order events

Events representing the order lifecycle:
- OrderSubmitted: Order sent to OMS
- OrderAccepted: Exchange confirmed order
- OrderPartiallyFilled: Partial fill received
- OrderFilled: Order completely filled
- OrderCancelled: Order cancelled
- OrderRejected: Exchange rejected order
- OrderExpired: Order expired (GTT orders)
"""
from datetime import datetime
from typing import Optional, Literal
from decimal import Decimal
from enum import Enum

from pydantic import Field

from .base import BaseEvent


class OrderSide(str, Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    """Time in force"""
    GTC = "gtc"  # Good till cancel
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTT = "gtt"  # Good till time


class OrderEvent(BaseEvent):
    """Base class for all order events"""

    # Order identity
    client_order_id: str = Field(description="Client-assigned order ID")
    exchange_order_id: Optional[str] = Field(
        default=None,
        description="Exchange-assigned order ID"
    )

    # Context
    exchange: str = Field(description="Exchange name")
    symbol: str = Field(description="Normalized symbol")

    # Order details
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None  # None for market orders
    stop_price: Optional[Decimal] = None  # For stop orders
    time_in_force: TimeInForce = TimeInForce.GTC

    # Strategy context
    strategy_id: Optional[str] = Field(
        default=None,
        description="Strategy that placed this order"
    )


class OrderSubmittedEvent(OrderEvent):
    """
    Order submitted to OMS

    This is the first event in the order lifecycle, emitted when
    the order manager accepts the order for processing.
    """
    event_type: Literal["order_submitted"] = "order_submitted"


class OrderAcceptedEvent(OrderEvent):
    """
    Order accepted by exchange

    Exchange has confirmed receipt and placed the order in the book.
    """
    event_type: Literal["order_accepted"] = "order_accepted"

    exchange_order_id: str = Field(description="Exchange-assigned order ID")
    exchange_timestamp: datetime = Field(description="Exchange timestamp")


class OrderPartiallyFilledEvent(OrderEvent):
    """
    Order partially filled

    Part of the order has been executed, but some quantity remains open.
    """
    event_type: Literal["order_partially_filled"] = "order_partially_filled"

    exchange_order_id: str

    # Fill details
    fill_id: str = Field(description="Unique fill identifier")
    fill_price: Decimal = Field(description="Price of this fill")
    fill_quantity: Decimal = Field(description="Quantity of this fill")

    # Cumulative
    total_filled: Decimal = Field(description="Total quantity filled so far")
    remaining_quantity: Decimal = Field(description="Remaining unfilled quantity")
    average_fill_price: Decimal = Field(description="Average price of all fills")

    # Fees
    fee: Decimal = Field(default=Decimal("0"), description="Fee for this fill")
    fee_currency: str = Field(default="", description="Currency of fee")

    # Trade info
    trade_id: Optional[str] = Field(default=None, description="Exchange trade ID")
    is_maker: Optional[bool] = Field(default=None, description="Was this maker or taker")

    # Timing
    fill_timestamp: datetime = Field(description="When fill occurred")


class OrderFilledEvent(OrderEvent):
    """
    Order completely filled

    All quantity has been executed. This is a terminal state.
    """
    event_type: Literal["order_filled"] = "order_filled"

    exchange_order_id: str

    # Fill summary
    total_filled: Decimal = Field(description="Total quantity filled")
    average_fill_price: Decimal = Field(description="Average fill price")
    total_fees: Decimal = Field(default=Decimal("0"), description="Total fees paid")
    fee_currency: str = Field(default="", description="Currency of fees")

    # Timing
    fill_timestamp: datetime = Field(description="When final fill occurred")

    # All fills (optional, for detailed tracking)
    fills: list[dict] = Field(
        default_factory=list,
        description="List of all individual fills"
    )


class OrderCancelledEvent(OrderEvent):
    """
    Order cancelled

    Order was cancelled (by user or system). This is a terminal state.
    """
    event_type: Literal["order_cancelled"] = "order_cancelled"

    exchange_order_id: Optional[str] = None

    # Cancellation details
    cancelled_by: Literal["user", "system", "exchange"] = "user"
    reason: Optional[str] = Field(default=None, description="Cancellation reason")

    # Partial fill info (if any fills before cancellation)
    filled_quantity: Decimal = Field(default=Decimal("0"))
    average_fill_price: Optional[Decimal] = None

    # Timing
    cancel_timestamp: datetime = Field(default_factory=datetime.utcnow)


class OrderRejectedEvent(OrderEvent):
    """
    Order rejected

    Exchange or OMS rejected the order. This is a terminal state.
    """
    event_type: Literal["order_rejected"] = "order_rejected"

    # Rejection details
    rejected_by: Literal["oms", "exchange"] = "exchange"
    rejection_reason: str = Field(description="Why order was rejected")
    rejection_code: Optional[str] = Field(
        default=None,
        description="Exchange-specific error code"
    )

    # Timing
    rejection_timestamp: datetime = Field(default_factory=datetime.utcnow)


class OrderExpiredEvent(OrderEvent):
    """
    Order expired

    Order reached its expiration time (GTT orders). This is a terminal state.
    """
    event_type: Literal["order_expired"] = "order_expired"

    exchange_order_id: Optional[str] = None

    # Expiration details
    expiration_time: datetime = Field(description="When order expired")
    filled_quantity: Decimal = Field(
        default=Decimal("0"),
        description="Quantity filled before expiration"
    )


class OrderModifiedEvent(OrderEvent):
    """
    Order modified

    Order parameters were changed (if exchange supports modification).
    """
    event_type: Literal["order_modified"] = "order_modified"

    exchange_order_id: str

    # What changed
    old_quantity: Optional[Decimal] = None
    new_quantity: Optional[Decimal] = None
    old_price: Optional[Decimal] = None
    new_price: Optional[Decimal] = None

    # Timing
    modification_timestamp: datetime = Field(default_factory=datetime.utcnow)


class OrderErrorEvent(OrderEvent):
    """
    Order error

    System error related to order processing.
    This is different from rejection - it's an unexpected error.
    """
    event_type: Literal["order_error"] = "order_error"

    # Error details
    error_message: str
    error_type: str  # e.g., "network_error", "timeout", "unknown"
    is_recoverable: bool = False

    # Timing
    error_timestamp: datetime = Field(default_factory=datetime.utcnow)
