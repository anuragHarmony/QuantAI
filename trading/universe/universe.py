"""
Universe - Collection of All Instruments

The Universe is the top-level container that manages all instruments
a strategy works with. It provides:

- Multiple tickers (groups)
- Global instrument registry
- Filtering and selection
- Weight management
- Configuration loading

Example:
    universe = Universe.from_config(config)

    # Get all instruments
    all_instruments = universe.get_all_instruments()

    # Get specific ticker
    majors = universe.get_ticker("crypto_majors")
    for instrument in majors:
        # Trade instrument

    # Filter across universe
    btc_products = universe.filter(symbol_pattern="BTC/*")
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from decimal import Decimal
from loguru import logger

from .instrument import Instrument, InstrumentType
from .ticker import Ticker
from .filters import InstrumentFilter, create_filter


@dataclass
class UniverseConfig:
    """Configuration for Universe"""

    name: str = "default"
    description: str = ""

    # Tickers definition
    tickers: List[Dict] = field(default_factory=list)

    # Individual instruments (optional, can also be in tickers)
    instruments: List[Dict] = field(default_factory=list)

    # Default filters to apply
    filters: List[Dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "UniverseConfig":
        """Create from dictionary"""
        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            tickers=data.get("tickers", []),
            instruments=data.get("instruments", []),
            filters=data.get("filters", []),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "tickers": self.tickers,
            "instruments": self.instruments,
            "filters": self.filters,
        }


class Universe:
    """
    Universe of instruments for a strategy

    Manages:
    - All instruments
    - Tickers (groups)
    - Filtering and selection
    - Weights and allocation
    """

    def __init__(self, name: str = "default", config: Optional[UniverseConfig] = None):
        """
        Initialize Universe

        Args:
            name: Universe name
            config: Configuration
        """
        self.name = name
        self.config = config

        # All instruments (symbol@exchange -> Instrument)
        self._instruments: Dict[str, Instrument] = {}

        # Tickers (ticker_id -> Ticker)
        self._tickers: Dict[str, Ticker] = {}

        # Default filters
        self._filters: List[InstrumentFilter] = []

        logger.info(f"Initialized Universe: {name}")

    def add_instrument(self, instrument: Instrument) -> None:
        """
        Add instrument to universe

        Args:
            instrument: Instrument to add
        """
        key = f"{instrument.symbol}@{instrument.exchange}"
        self._instruments[key] = instrument

    def add_instruments(self, instruments: List[Instrument]) -> None:
        """Add multiple instruments"""
        for instrument in instruments:
            self.add_instrument(instrument)

    def get_instrument(self, symbol: str, exchange: str) -> Optional[Instrument]:
        """Get specific instrument"""
        key = f"{symbol}@{exchange}"
        return self._instruments.get(key)

    def has_instrument(self, symbol: str, exchange: str) -> bool:
        """Check if instrument exists"""
        key = f"{symbol}@{exchange}"
        return key in self._instruments

    def get_all_instruments(
        self,
        enabled_only: bool = True,
        apply_filters: bool = False
    ) -> List[Instrument]:
        """
        Get all instruments in universe

        Args:
            enabled_only: Only return enabled instruments
            apply_filters: Apply default filters

        Returns:
            List of instruments
        """
        instruments = list(self._instruments.values())

        if enabled_only:
            instruments = [i for i in instruments if i.enabled]

        if apply_filters and self._filters:
            instruments = self._apply_filters(instruments, self._filters)

        return instruments

    def add_ticker(self, ticker: Ticker) -> None:
        """
        Add ticker to universe

        Args:
            ticker: Ticker to add
        """
        self._tickers[ticker.ticker_id] = ticker

        # Also add ticker's instruments to universe
        for instrument in ticker.get_instruments(enabled_only=False):
            self.add_instrument(instrument)

        logger.debug(f"Added ticker: {ticker.ticker_id} with {ticker.count()} instruments")

    def get_ticker(self, ticker_id: str) -> Optional[Ticker]:
        """Get ticker by ID"""
        return self._tickers.get(ticker_id)

    def get_all_tickers(self) -> List[Ticker]:
        """Get all tickers"""
        return list(self._tickers.values())

    def get_ticker_ids(self) -> List[str]:
        """Get list of ticker IDs"""
        return list(self._tickers.keys())

    def has_ticker(self, ticker_id: str) -> bool:
        """Check if ticker exists"""
        return ticker_id in self._tickers

    def add_filter(self, filter: InstrumentFilter) -> None:
        """Add default filter"""
        self._filters.append(filter)

    def filter(
        self,
        filters: Optional[List[InstrumentFilter]] = None,
        **kwargs
    ) -> List[Instrument]:
        """
        Filter instruments

        Args:
            filters: List of filters to apply
            **kwargs: Quick filter parameters (e.g., symbol_pattern="BTC/*")

        Returns:
            Filtered list of instruments

        Examples:
            # Using filter objects
            universe.filter(filters=[TagFilter(["major"])])

            # Using kwargs
            universe.filter(symbol_pattern="BTC/*")
            universe.filter(exchange=["binance"], tags=["major"])
        """
        instruments = self.get_all_instruments(enabled_only=True, apply_filters=False)

        # Create filters from kwargs
        if kwargs:
            created_filters = []
            if "symbol_pattern" in kwargs:
                from .filters import SymbolPatternFilter
                created_filters.append(SymbolPatternFilter(kwargs["symbol_pattern"]))
            if "exchange" in kwargs:
                from .filters import ExchangeFilter
                exchanges = kwargs["exchange"] if isinstance(kwargs["exchange"], list) else [kwargs["exchange"]]
                created_filters.append(ExchangeFilter(exchanges))
            if "tags" in kwargs:
                from .filters import TagFilter
                tags = kwargs["tags"] if isinstance(kwargs["tags"], list) else [kwargs["tags"]]
                created_filters.append(TagFilter(tags))
            if "instrument_type" in kwargs:
                from .filters import InstrumentTypeFilter
                types = kwargs["instrument_type"] if isinstance(kwargs["instrument_type"], list) else [kwargs["instrument_type"]]
                created_filters.append(InstrumentTypeFilter(types))

            filters = (filters or []) + created_filters

        if filters:
            instruments = self._apply_filters(instruments, filters)

        return instruments

    def _apply_filters(
        self,
        instruments: List[Instrument],
        filters: List[InstrumentFilter]
    ) -> List[Instrument]:
        """Apply filters to instrument list"""
        for filter in filters:
            instruments = [i for i in instruments if filter.matches(i)]
        return instruments

    def get_symbols(self, enabled_only: bool = True) -> List[str]:
        """Get list of all symbols"""
        instruments = self.get_all_instruments(enabled_only=enabled_only)
        return list(set(i.symbol for i in instruments))

    def get_by_exchange(self, exchange: str) -> List[Instrument]:
        """Get all instruments for specific exchange"""
        return [
            i for i in self._instruments.values()
            if i.exchange == exchange and i.enabled
        ]

    def get_by_tag(self, tag: str) -> List[Instrument]:
        """Get all instruments with specific tag"""
        return [
            i for i in self._instruments.values()
            if i.has_tag(tag) and i.enabled
        ]

    def get_by_type(self, instrument_type: InstrumentType) -> List[Instrument]:
        """Get all instruments of specific type"""
        return [
            i for i in self._instruments.values()
            if i.instrument_type == instrument_type and i.enabled
        ]

    def normalize_weights(self) -> None:
        """Normalize all instrument weights to sum to 1.0"""
        instruments = [i for i in self._instruments.values() if i.weight is not None]

        if not instruments:
            return

        total_weight = sum(i.weight for i in instruments)

        if total_weight == 0:
            return

        for instrument in instruments:
            instrument.weight = instrument.weight / total_weight

        logger.info(f"Normalized weights for {len(instruments)} instruments")

    def get_weight_distribution(self) -> Dict[str, Decimal]:
        """Get weight distribution"""
        return {
            str(i): i.weight
            for i in self._instruments.values()
            if i.weight is not None
        }

    def count(self, enabled_only: bool = True) -> int:
        """Get count of instruments"""
        if enabled_only:
            return len([i for i in self._instruments.values() if i.enabled])
        return len(self._instruments)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "tickers": [t.to_dict() for t in self._tickers.values()],
            "instruments": [i.to_dict() for i in self._instruments.values()],
        }

    @classmethod
    def from_config(cls, config: UniverseConfig) -> "Universe":
        """
        Create Universe from configuration

        Args:
            config: Universe configuration

        Returns:
            Universe instance
        """
        universe = cls(name=config.name, config=config)

        # Create instruments from config
        for inst_data in config.instruments:
            instrument = Instrument.from_dict(inst_data)
            universe.add_instrument(instrument)

        # Create tickers from config
        for ticker_data in config.tickers:
            ticker_id = ticker_data["id"]
            description = ticker_data.get("description", "")

            # Create instruments for ticker
            ticker_instruments = []
            for symbol_or_dict in ticker_data.get("instruments", []):
                if isinstance(symbol_or_dict, str):
                    # Simple symbol string
                    # Find in universe or create basic instrument
                    parts = symbol_or_dict.split("@")
                    symbol = parts[0]
                    exchange = parts[1] if len(parts) > 1 else ticker_data.get("exchange", "default")

                    existing = universe.get_instrument(symbol, exchange)
                    if existing:
                        ticker_instruments.append(existing)
                    else:
                        # Create basic instrument
                        instrument = Instrument(
                            symbol=symbol,
                            exchange=exchange,
                            instrument_type=InstrumentType(ticker_data.get("type", "spot")),
                            tags=set(ticker_data.get("tags", []))
                        )
                        ticker_instruments.append(instrument)
                else:
                    # Full instrument dict
                    instrument = Instrument.from_dict(symbol_or_dict)
                    ticker_instruments.append(instrument)

            # Create ticker
            ticker = Ticker(
                ticker_id=ticker_id,
                instruments=ticker_instruments,
                description=description,
                metadata=ticker_data.get("metadata", {})
            )

            universe.add_ticker(ticker)

        # Add default filters
        for filter_data in config.filters:
            filter_type = filter_data.get("type")
            filter_params = {k: v for k, v in filter_data.items() if k != "type"}
            filter_obj = create_filter(filter_type, **filter_params)
            universe.add_filter(filter_obj)

        logger.info(
            f"Created Universe '{config.name}' with "
            f"{universe.count()} instruments in "
            f"{len(universe._tickers)} tickers"
        )

        return universe

    def __len__(self) -> int:
        """Get count of enabled instruments"""
        return self.count(enabled_only=True)

    def __contains__(self, item) -> bool:
        """Check if instrument in universe"""
        if isinstance(item, Instrument):
            return self.has_instrument(item.symbol, item.exchange)
        elif isinstance(item, str):
            # Check if any instrument has this symbol
            return any(i.symbol == item for i in self._instruments.values())
        return False

    def __str__(self):
        """String representation"""
        return f"Universe({self.name}, {self.count()} instruments, {len(self._tickers)} tickers)"

    def __repr__(self):
        """Detailed representation"""
        return (
            f"Universe(name={self.name}, "
            f"instruments={self.count()}, "
            f"tickers={list(self._tickers.keys())})"
        )
