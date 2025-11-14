"""
Instrument Definition

Represents a single tradeable instrument with rich metadata:
- Symbol and exchange
- Instrument type (spot, futures, options, etc.)
- Tags for filtering
- Weight for portfolio allocation
- Custom metadata

This provides a unified view of any tradeable asset.
"""
from enum import Enum
from typing import Dict, List, Optional, Set
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime


class InstrumentType(Enum):
    """Types of instruments"""
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTION = "option"
    SWAP = "swap"
    INDEX = "index"


@dataclass
class Instrument:
    """
    Represents a tradeable instrument with metadata

    This is the fundamental building block of the Universe system.
    Each instrument can have:
    - Trading details (symbol, exchange)
    - Classification (type, tags)
    - Portfolio allocation (weight)
    - Custom metadata
    """

    # Core identification
    symbol: str  # Normalized symbol (e.g., "BTC/USDT")
    exchange: str  # Exchange name (e.g., "binance")

    # Classification
    instrument_type: InstrumentType = InstrumentType.SPOT
    base_currency: Optional[str] = None  # e.g., "BTC"
    quote_currency: Optional[str] = None  # e.g., "USDT"

    # Organization
    tags: Set[str] = field(default_factory=set)  # e.g., {"major", "high_liquidity"}

    # Portfolio management
    weight: Optional[Decimal] = None  # Portfolio weight (0.0-1.0)

    # Trading constraints
    min_quantity: Optional[Decimal] = None
    max_quantity: Optional[Decimal] = None
    tick_size: Optional[Decimal] = None

    # Status
    enabled: bool = True

    # Metadata
    metadata: Dict = field(default_factory=dict)

    # Timestamps
    added_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Post-initialization processing"""
        # Parse symbol if currencies not provided
        if not self.base_currency or not self.quote_currency:
            if "/" in self.symbol:
                parts = self.symbol.split("/")
                self.base_currency = self.base_currency or parts[0]
                self.quote_currency = self.quote_currency or parts[1].split(":")[0]

        # Ensure tags is a set
        if isinstance(self.tags, list):
            self.tags = set(self.tags)

    def has_tag(self, tag: str) -> bool:
        """Check if instrument has a specific tag"""
        return tag in self.tags

    def has_any_tag(self, tags: List[str]) -> bool:
        """Check if instrument has any of the specified tags"""
        return bool(self.tags.intersection(set(tags)))

    def has_all_tags(self, tags: List[str]) -> bool:
        """Check if instrument has all of the specified tags"""
        return set(tags).issubset(self.tags)

    def add_tag(self, tag: str) -> None:
        """Add a tag to instrument"""
        self.tags.add(tag)
        self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from instrument"""
        self.tags.discard(tag)
        self.updated_at = datetime.utcnow()

    def matches_pattern(self, pattern: str) -> bool:
        """
        Check if symbol matches pattern

        Supports wildcards:
        - BTC/* matches BTC/USDT, BTC/USD, etc.
        - */USDT matches any/USDT
        - BTC* matches BTCUSDT, BTCUSD, etc.
        """
        import fnmatch
        return fnmatch.fnmatch(self.symbol, pattern)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "instrument_type": self.instrument_type.value,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "tags": list(self.tags),
            "weight": str(self.weight) if self.weight else None,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Instrument":
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            exchange=data["exchange"],
            instrument_type=InstrumentType(data.get("instrument_type", "spot")),
            base_currency=data.get("base_currency"),
            quote_currency=data.get("quote_currency"),
            tags=set(data.get("tags", [])),
            weight=Decimal(data["weight"]) if data.get("weight") else None,
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )

    def __hash__(self):
        """Make instrument hashable"""
        return hash((self.symbol, self.exchange))

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Instrument):
            return False
        return self.symbol == other.symbol and self.exchange == other.exchange

    def __str__(self):
        """String representation"""
        return f"{self.symbol}@{self.exchange}"

    def __repr__(self):
        """Detailed representation"""
        tags_str = f"tags={{{','.join(self.tags)}}}" if self.tags else ""
        weight_str = f"weight={self.weight}" if self.weight else ""
        parts = [self.symbol, self.exchange, self.instrument_type.value]
        if tags_str:
            parts.append(tags_str)
        if weight_str:
            parts.append(weight_str)
        return f"Instrument({', '.join(parts)})"
