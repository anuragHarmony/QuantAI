"""
Instrument Filters

Dynamic filtering of instruments based on various criteria:
- Volume thresholds
- Price ranges
- Currency filters
- Pattern matching
- Custom predicates

Filters can be combined with AND/OR logic.
"""
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from decimal import Decimal

from .instrument import Instrument, InstrumentType


class InstrumentFilter(ABC):
    """Abstract base class for instrument filters"""

    @abstractmethod
    def matches(self, instrument: Instrument) -> bool:
        """
        Check if instrument matches filter criteria

        Args:
            instrument: Instrument to check

        Returns:
            True if instrument matches
        """
        pass


class SymbolPatternFilter(InstrumentFilter):
    """Filter by symbol pattern (with wildcards)"""

    def __init__(self, pattern: str):
        """
        Args:
            pattern: Pattern with wildcards (e.g., "BTC/*", "*/USDT")
        """
        self.pattern = pattern

    def matches(self, instrument: Instrument) -> bool:
        return instrument.matches_pattern(self.pattern)


class ExchangeFilter(InstrumentFilter):
    """Filter by exchange"""

    def __init__(self, exchanges: List[str]):
        """
        Args:
            exchanges: List of exchange names
        """
        self.exchanges = set(exchanges)

    def matches(self, instrument: Instrument) -> bool:
        return instrument.exchange in self.exchanges


class InstrumentTypeFilter(InstrumentFilter):
    """Filter by instrument type"""

    def __init__(self, instrument_types: List[InstrumentType]):
        """
        Args:
            instrument_types: List of instrument types
        """
        self.types = set(instrument_types)

    def matches(self, instrument: Instrument) -> bool:
        return instrument.instrument_type in self.types


class QuoteCurrencyFilter(InstrumentFilter):
    """Filter by quote currency"""

    def __init__(self, quote_currencies: List[str]):
        """
        Args:
            quote_currencies: List of quote currencies (e.g., ["USDT", "USD"])
        """
        self.quotes = set(quote_currencies)

    def matches(self, instrument: Instrument) -> bool:
        return instrument.quote_currency in self.quotes


class BaseCurrencyFilter(InstrumentFilter):
    """Filter by base currency"""

    def __init__(self, base_currencies: List[str]):
        """
        Args:
            base_currencies: List of base currencies (e.g., ["BTC", "ETH"])
        """
        self.bases = set(base_currencies)

    def matches(self, instrument: Instrument) -> bool:
        return instrument.base_currency in self.bases


class TagFilter(InstrumentFilter):
    """Filter by tags"""

    def __init__(self, tags: List[str], match_all: bool = True):
        """
        Args:
            tags: List of tags to match
            match_all: If True, instrument must have all tags. If False, any tag.
        """
        self.tags = tags
        self.match_all = match_all

    def matches(self, instrument: Instrument) -> bool:
        if self.match_all:
            return instrument.has_all_tags(self.tags)
        else:
            return instrument.has_any_tag(self.tags)


class EnabledFilter(InstrumentFilter):
    """Filter by enabled status"""

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: True to match enabled instruments, False for disabled
        """
        self.enabled = enabled

    def matches(self, instrument: Instrument) -> bool:
        return instrument.enabled == self.enabled


class CustomFilter(InstrumentFilter):
    """Custom filter using a predicate function"""

    def __init__(self, predicate: Callable[[Instrument], bool], description: str = ""):
        """
        Args:
            predicate: Function that takes Instrument and returns bool
            description: Human-readable description of filter
        """
        self.predicate = predicate
        self.description = description

    def matches(self, instrument: Instrument) -> bool:
        return self.predicate(instrument)


# Composite filters

class AndFilter(InstrumentFilter):
    """Combines filters with AND logic"""

    def __init__(self, *filters: InstrumentFilter):
        """
        Args:
            *filters: Filters to combine
        """
        self.filters = filters

    def matches(self, instrument: Instrument) -> bool:
        return all(f.matches(instrument) for f in self.filters)


class OrFilter(InstrumentFilter):
    """Combines filters with OR logic"""

    def __init__(self, *filters: InstrumentFilter):
        """
        Args:
            *filters: Filters to combine
        """
        self.filters = filters

    def matches(self, instrument: Instrument) -> bool:
        return any(f.matches(instrument) for f in self.filters)


class NotFilter(InstrumentFilter):
    """Negates a filter"""

    def __init__(self, filter_to_negate: InstrumentFilter):
        """
        Args:
            filter_to_negate: Filter to negate
        """
        self.filter = filter_to_negate

    def matches(self, instrument: Instrument) -> bool:
        return not self.filter.matches(instrument)


# Factory function

def create_filter(filter_type: str, **kwargs) -> InstrumentFilter:
    """
    Create filter from type and parameters

    Args:
        filter_type: Type of filter
        **kwargs: Filter parameters

    Returns:
        InstrumentFilter instance

    Examples:
        create_filter("symbol_pattern", pattern="BTC/*")
        create_filter("exchange", exchanges=["binance", "okex"])
        create_filter("tags", tags=["major"], match_all=True)
    """
    filters = {
        "symbol_pattern": SymbolPatternFilter,
        "exchange": ExchangeFilter,
        "instrument_type": InstrumentTypeFilter,
        "quote_currency": QuoteCurrencyFilter,
        "base_currency": BaseCurrencyFilter,
        "tags": TagFilter,
        "enabled": EnabledFilter,
    }

    if filter_type not in filters:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return filters[filter_type](**kwargs)


# Convenience functions

def major_cryptos() -> InstrumentFilter:
    """Filter for major cryptocurrencies"""
    return TagFilter(tags=["major"])


def usdt_pairs() -> InstrumentFilter:
    """Filter for USDT pairs"""
    return QuoteCurrencyFilter(quote_currencies=["USDT"])


def spot_only() -> InstrumentFilter:
    """Filter for spot instruments only"""
    return InstrumentTypeFilter(instrument_types=[InstrumentType.SPOT])


def high_liquidity() -> InstrumentFilter:
    """Filter for high liquidity instruments"""
    return TagFilter(tags=["high_liquidity"])
