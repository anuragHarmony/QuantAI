"""
Position and Portfolio events

Events representing position and portfolio state changes:
- PositionOpened: New position created
- PositionModified: Position size or state changed
- PositionClosed: Position closed
- PortfolioValueUpdated: Portfolio value changed
- RiskLimitTriggered: Risk limit was hit
"""
from datetime import datetime
from typing import Optional, Literal
from decimal import Decimal

from pydantic import Field

from .base import BaseEvent


class PositionSide(str, Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionEvent(BaseEvent):
    """Base class for position events"""

    # Position identity
    position_id: str = Field(description="Unique position identifier")

    # Context
    exchange: str
    symbol: str

    # Position details
    side: PositionSide
    quantity: Decimal

    # Pricing
    entry_price: Decimal = Field(description="Average entry price")
    current_price: Optional[Decimal] = Field(
        default=None,
        description="Current market price"
    )

    # P&L
    unrealized_pnl: Decimal = Field(description="Unrealized profit/loss")
    realized_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Realized profit/loss"
    )

    # Strategy context
    strategy_id: Optional[str] = None


class PositionOpenedEvent(PositionEvent):
    """
    Position opened

    A new position was created from an order fill.
    """
    event_type: Literal["position_opened"] = "position_opened"

    # Opening details
    opening_order_id: str = Field(description="Order that opened this position")
    opening_timestamp: datetime = Field(default_factory=datetime.utcnow)


class PositionModifiedEvent(PositionEvent):
    """
    Position modified

    Position size or price changed due to additional fills.
    """
    event_type: Literal["position_modified"] = "position_modified"

    # What changed
    old_quantity: Decimal
    new_quantity: Decimal
    old_entry_price: Decimal
    new_entry_price: Decimal

    # Modification details
    modifying_order_id: Optional[str] = Field(
        default=None,
        description="Order that modified this position"
    )
    modification_timestamp: datetime = Field(default_factory=datetime.utcnow)


class PositionClosedEvent(PositionEvent):
    """
    Position closed

    Position was fully closed. This is a terminal state.
    """
    event_type: Literal["position_closed"] = "position_closed"

    # Closing details
    closing_order_id: str = Field(description="Order that closed this position")
    exit_price: Decimal = Field(description="Average exit price")

    # Final P&L
    final_realized_pnl: Decimal = Field(description="Final realized P&L")
    final_return: Decimal = Field(description="Return as percentage")

    # Timing
    open_timestamp: datetime = Field(description="When position was opened")
    close_timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: int = Field(description="How long position was held")


class PortfolioEvent(BaseEvent):
    """Base class for portfolio events"""

    # Portfolio identity
    portfolio_id: str = Field(default="default", description="Portfolio identifier")
    strategy_id: Optional[str] = None


class PortfolioValueEvent(PortfolioEvent):
    """
    Portfolio value updated

    Total portfolio value has changed.
    """
    event_type: Literal["portfolio_value"] = "portfolio_value"

    # Value components
    cash: Decimal = Field(description="Cash balance")
    equity: Decimal = Field(description="Value of positions")
    total_value: Decimal = Field(description="Total portfolio value")

    # P&L
    unrealized_pnl: Decimal = Field(description="Total unrealized P&L")
    realized_pnl: Decimal = Field(description="Total realized P&L today/session")
    total_pnl: Decimal = Field(description="Total P&L")

    # Performance
    return_pct: Decimal = Field(description="Return percentage")
    initial_value: Decimal = Field(description="Starting portfolio value")

    # Update details
    value_timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskLimitEvent(PortfolioEvent):
    """
    Risk limit triggered

    A risk limit was breached or approached.
    """
    event_type: Literal["risk_limit"] = "risk_limit"

    # Limit details
    limit_type: str = Field(
        description="Type of limit (e.g., 'max_drawdown', 'max_exposure')"
    )
    limit_value: Decimal = Field(description="The limit threshold")
    current_value: Decimal = Field(description="Current value that triggered limit")

    # Severity
    severity: Literal["warning", "breach", "critical"] = "breach"

    # Action taken
    action_taken: Optional[str] = Field(
        default=None,
        description="What action was taken (e.g., 'flatten_all', 'block_orders')"
    )

    # Timing
    trigger_timestamp: datetime = Field(default_factory=datetime.utcnow)


class MarginCallEvent(PortfolioEvent):
    """
    Margin call

    Account is below margin requirements (for margin/futures trading).
    """
    event_type: Literal["margin_call"] = "margin_call"

    # Margin details
    required_margin: Decimal
    available_margin: Decimal
    margin_deficit: Decimal

    # Account details
    leverage: Decimal
    equity: Decimal

    # Action required
    action_required: str = Field(
        description="What needs to be done (e.g., 'add_funds', 'close_positions')"
    )

    # Timing
    margin_call_timestamp: datetime = Field(default_factory=datetime.utcnow)


from enum import Enum  # Add this import at the top with others
