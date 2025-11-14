"""
Basic unit tests for Universe system
Tests core functionality of Instrument, Ticker, Universe, and Filters
"""
import pytest
from decimal import Decimal
from trading.universe import (
    Instrument,
    InstrumentType,
    Ticker,
    Universe,
    UniverseConfig,
    SymbolPatternFilter,
    ExchangeFilter,
    InstrumentTypeFilter,
    QuoteCurrencyFilter,
    TagFilter,
    AndFilter,
    OrFilter,
)


class TestInstrumentBasic:
    """Test basic Instrument functionality"""

    def test_create_and_use_instrument(self):
        inst = Instrument(symbol="BTC/USDT", exchange="binance", tags={"major"})
        assert inst.symbol == "BTC/USDT"
        assert inst.has_tag("major")
        assert str(inst) == "BTC/USDT@binance"
        assert inst.matches_pattern("BTC/*")


class TestTickerBasic:
    """Test basic Ticker functionality"""

    def test_add_and_get_instruments(self):
        ticker = Ticker("crypto_majors")
        ticker.add_instrument(Instrument("BTC/USDT", "binance"))
        ticker.add_instrument(Instrument("ETH/USDT", "binance"))
        
        assert len(ticker.get_instruments()) == 2
        assert "BTC/USDT" in ticker.get_symbols()


class TestUniverseBasic:
    """Test basic Universe functionality"""

    def test_create_and_query_universe(self):
        universe = Universe("test")
        universe.add_instrument(Instrument("BTC/USDT", "binance", tags={"major"}))
        universe.add_instrument(Instrument("ETH/USDT", "binance", tags={"major"}))
        universe.add_instrument(Instrument("SOL/USDT", "okex", tags={"alt"}))
        
        assert len(universe.get_all_instruments()) == 3
        assert len(universe.get_by_tag("major")) == 2
        assert len(universe.get_by_exchange("binance")) == 2


class TestFiltersBasic:
    """Test basic filter functionality"""

    def test_filters_work(self):
        inst1 = Instrument("BTC/USDT", "binance", tags={"major"}, instrument_type=InstrumentType.SPOT)
        inst2 = Instrument("ETH/USDT", "okex", tags={"alt"}, instrument_type=InstrumentType.SPOT)
        
        # Pattern filter
        assert SymbolPatternFilter("BTC/*").matches(inst1)
        assert not SymbolPatternFilter("ETH/*").matches(inst1)
        
        # Tag filter  
        assert TagFilter(["major"], match_all=True).matches(inst1)
        assert not TagFilter(["major"], match_all=True).matches(inst2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
