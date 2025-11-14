"""
Ticker - Group of Related Instruments

A Ticker represents a group of related instruments that can be:
- Traded together (e.g., "crypto_majors")
- Different products of same underlying (e.g., "btc_products": spot, futures, options)
- Filtered by characteristics (e.g., "high_volume_pairs")

Examples:
- crypto_majors: [BTC/USDT, ETH/USDT, BNB/USDT]
- btc_products: [BTC/USDT, BTC-PERP, BTC/USDT:USDT]
- nasdaq_tech: [AAPL, MSFT, GOOGL]
"""
from typing import List, Dict, Optional, Iterator, Set
from decimal import Decimal
from loguru import logger

from .instrument import Instrument, InstrumentType


class Ticker:
    """
    Group of related instruments

    A Ticker provides:
    - Named grouping of instruments
    - Iteration over instruments
    - Filtering within the group
    - Weight normalization
    """

    def __init__(
        self,
        ticker_id: str,
        instruments: Optional[List[Instrument]] = None,
        description: str = "",
        metadata: Optional[Dict] = None
    ):
        """
        Initialize ticker

        Args:
            ticker_id: Unique identifier (e.g., "crypto_majors")
            instruments: List of instruments in this ticker
            description: Human-readable description
            metadata: Additional metadata
        """
        self.ticker_id = ticker_id
        self.description = description
        self.metadata = metadata or {}

        # Store instruments in dict for fast lookup
        self._instruments: Dict[str, Instrument] = {}

        if instruments:
            for instrument in instruments:
                self.add_instrument(instrument)

        logger.debug(f"Created ticker: {ticker_id} with {len(self._instruments)} instruments")

    def add_instrument(self, instrument: Instrument) -> None:
        """
        Add instrument to ticker

        Args:
            instrument: Instrument to add
        """
        key = f"{instrument.symbol}@{instrument.exchange}"
        self._instruments[key] = instrument

    def remove_instrument(self, symbol: str, exchange: str) -> None:
        """
        Remove instrument from ticker

        Args:
            symbol: Symbol to remove
            exchange: Exchange
        """
        key = f"{symbol}@{exchange}"
        if key in self._instruments:
            del self._instruments[key]

    def get_instrument(self, symbol: str, exchange: str) -> Optional[Instrument]:
        """Get specific instrument"""
        key = f"{symbol}@{exchange}"
        return self._instruments.get(key)

    def has_instrument(self, symbol: str, exchange: str) -> bool:
        """Check if ticker contains instrument"""
        key = f"{symbol}@{exchange}"
        return key in self._instruments

    def get_instruments(
        self,
        enabled_only: bool = True,
        tags: Optional[List[str]] = None,
        instrument_type: Optional[InstrumentType] = None
    ) -> List[Instrument]:
        """
        Get instruments with optional filtering

        Args:
            enabled_only: Only return enabled instruments
            tags: Filter by tags (instrument must have all tags)
            instrument_type: Filter by instrument type

        Returns:
            List of matching instruments
        """
        instruments = list(self._instruments.values())

        # Filter by enabled status
        if enabled_only:
            instruments = [i for i in instruments if i.enabled]

        # Filter by tags
        if tags:
            instruments = [i for i in instruments if i.has_all_tags(tags)]

        # Filter by type
        if instrument_type:
            instruments = [i for i in instruments if i.instrument_type == instrument_type]

        return instruments

    def get_symbols(self, enabled_only: bool = True) -> List[str]:
        """Get list of symbols"""
        instruments = self.get_instruments(enabled_only=enabled_only)
        return [i.symbol for i in instruments]

    def get_by_tag(self, tag: str) -> List[Instrument]:
        """Get all instruments with specific tag"""
        return [i for i in self._instruments.values() if i.has_tag(tag)]

    def get_by_type(self, instrument_type: InstrumentType) -> List[Instrument]:
        """Get all instruments of specific type"""
        return [i for i in self._instruments.values() if i.instrument_type == instrument_type]

    def get_weighted_instruments(self) -> List[Instrument]:
        """Get instruments that have weights defined"""
        return [i for i in self._instruments.values() if i.weight is not None]

    def normalize_weights(self) -> None:
        """
        Normalize weights to sum to 1.0

        Useful for portfolio allocation.
        """
        weighted = self.get_weighted_instruments()

        if not weighted:
            return

        total_weight = sum(i.weight for i in weighted)

        if total_weight == 0:
            return

        for instrument in weighted:
            instrument.weight = instrument.weight / total_weight

        logger.debug(f"Normalized weights for ticker {self.ticker_id}")

    def get_weight_distribution(self) -> Dict[str, Decimal]:
        """Get weight distribution as dict"""
        weighted = self.get_weighted_instruments()
        return {
            str(i): i.weight
            for i in weighted
        }

    def count(self, enabled_only: bool = True) -> int:
        """Get count of instruments"""
        if enabled_only:
            return len([i for i in self._instruments.values() if i.enabled])
        return len(self._instruments)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "ticker_id": self.ticker_id,
            "description": self.description,
            "instruments": [i.to_dict() for i in self._instruments.values()],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Ticker":
        """Create from dictionary"""
        instruments = [
            Instrument.from_dict(i_data)
            for i_data in data.get("instruments", [])
        ]

        return cls(
            ticker_id=data["ticker_id"],
            instruments=instruments,
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )

    # Make ticker iterable
    def __iter__(self) -> Iterator[Instrument]:
        """Iterate over enabled instruments"""
        return iter(self.get_instruments(enabled_only=True))

    def __len__(self) -> int:
        """Get count of enabled instruments"""
        return self.count(enabled_only=True)

    def __contains__(self, item) -> bool:
        """Check if instrument in ticker"""
        if isinstance(item, Instrument):
            return self.has_instrument(item.symbol, item.exchange)
        elif isinstance(item, str):
            # Check by symbol (any exchange)
            return any(i.symbol == item for i in self._instruments.values())
        return False

    def __str__(self):
        """String representation"""
        return f"Ticker({self.ticker_id}, {self.count()} instruments)"

    def __repr__(self):
        """Detailed representation"""
        symbols = ", ".join(self.get_symbols()[:5])
        if len(self._instruments) > 5:
            symbols += f", ... ({len(self._instruments)} total)"
        return f"Ticker(id={self.ticker_id}, instruments=[{symbols}])"
