"""
Base event models for the trading system

All events inherit from BaseEvent and use Pydantic for validation.
Events are immutable and represent facts that happened in the system.
"""
from datetime import datetime
from typing import Any, Optional, Literal
from decimal import Decimal
from enum import Enum
import uuid

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """
    Base class for all events in the system

    All events are immutable facts that represent something that happened.
    Events should never be modified after creation.
    """

    # Identity
    event_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique event identifier"
    )
    event_type: str = Field(description="Type of event (e.g., 'tick', 'order_filled')")

    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this event was created"
    )

    # Context
    exchange: Optional[str] = Field(
        default=None,
        description="Exchange this event relates to (e.g., 'binance')"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Normalized symbol (e.g., 'BTC/USDT')"
    )

    # Extensibility
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata specific to this event"
    )

    class Config:
        # Make events immutable
        frozen = True
        # Allow Decimal serialization
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class EventPriority(str, Enum):
    """Event priority levels for processing"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SystemEvent(BaseEvent):
    """System-level events (connections, errors, heartbeats)"""
    pass


class ConnectionEvent(SystemEvent):
    """Exchange connection status changed"""
    event_type: Literal["connection"] = "connection"
    exchange: str
    connected: bool
    reason: Optional[str] = None


class ErrorEvent(SystemEvent):
    """System error occurred"""
    event_type: Literal["error"] = "error"
    error_type: str
    error_message: str
    component: str  # Which component had the error
    severity: str = "error"  # "warning", "error", "critical"
    traceback: Optional[str] = None


class HeartbeatEvent(SystemEvent):
    """Periodic heartbeat from a component"""
    event_type: Literal["heartbeat"] = "heartbeat"
    component: str
    status: dict[str, Any] = Field(default_factory=dict)
