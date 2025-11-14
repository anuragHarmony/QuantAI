"""
Unit tests for Universe system
Tests Instrument, Ticker, Universe, and Filters
"""
import pytest
from decimal import Decimal
from trading.universe import (
    Instrument,
    InstrumentType,
    Ticker,
    Universe,
    UniverseConfig,
    InstrumentFilter,
    SymbolPatternFilter,
    ExchangeFilter,
    InstrumentTypeFilter,
    QuoteCurrencyFilter,
    TagFilter,
    AndFilter,
    OrFilter,
    NotFilter,
)


class TestInstrument:
    """Test Instrument class"""

    def test_create_basic_instrument(self):
        """Test creating a basic instrument"""
        inst = Instrument(
            symbol="BTC/USDT",
            exchange="binance",
        )
        assert inst.symbol == "BTC/USDT"
        assert inst.exchange == "binance"
        assert inst.instrument_type == InstrumentType.SPOT
        assert inst.enabled is True
        assert len(inst.tags) == 0

    def test_create_instrument_with_metadata(self):
        """Test creating instrument with full metadata"""
        inst = Instrument(
            symbol="BTC/USDT",
            exchange="binance",
            instrument_type=InstrumentType.PERPETUAL,
            tags={"major", "high_liquidity", "btc"},
            weight=Decimal("0.5"),
        )
        assert inst.instrument_type == InstrumentType.PERPETUAL
        assert "major" in inst.tags
        assert "high_liquidity" in inst.tags
        assert inst.weight == Decimal("0.5")

    def test_has_tag(self):
        """Test tag checking"""
        inst = Instrument(
            symbol="BTC/USDT",
            exchange="binance",
            tags={"major", "high_liquidity"}
        )
        assert inst.has_tag("major")
        assert inst.has_tag("high_liquidity")
        assert not inst.has_tag("alt")

    def test_has_all_tags(self):
        """Test checking multiple tags"""
        inst = Instrument(
            symbol="BTC/USDT",
            exchange="binance",
            tags={"major", "high_liquidity", "btc"}
        )
        assert inst.has_all_tags(["major", "btc"])
        assert not inst.has_all_tags(["major", "alt"])

    def test_has_any_tag(self):
        """Test checking any tag"""
        inst = Instrument(
            symbol="BTC/USDT",
            exchange="binance",
            tags={"major", "high_liquidity"}
        )
        assert inst.has_any_tag(["major", "alt"])
        assert not inst.has_any_tag(["alt", "defi"])

    def test_matches_pattern(self):
        """Test pattern matching"""
        inst = Instrument(symbol="BTC/USDT", exchange="binance")

        assert inst.matches_pattern("BTC/USDT")
        assert inst.matches_pattern("BTC/*")
        assert inst.matches_pattern("*/USDT")
        assert inst.matches_pattern("*/*")
        assert not inst.matches_pattern("ETH/*")

    def test_full_symbol(self):
        """Test full symbol with exchange"""
        inst = Instrument(symbol="BTC/USDT", exchange="binance")
        assert inst.full_symbol == "BTC/USDT@binance"

    def test_enable_disable(self):
        """Test enabling/disabling instruments"""
        inst = Instrument(symbol="BTC/USDT", exchange="binance")

        assert inst.enabled
        inst.disable()
        assert not inst.enabled
        inst.enable()
        assert inst.enabled


class TestTicker:
    """Test Ticker class"""

    def test_create_empty_ticker(self):
        """Test creating empty ticker"""
        ticker = Ticker("crypto_majors")
        assert ticker.ticker_id == "crypto_majors"
        assert len(ticker.get_instruments()) == 0

    def test_add_instrument(self):
        """Test adding instruments"""
        ticker = Ticker("crypto_majors")
        inst = Instrument("BTC/USDT", "binance")

        ticker.add_instrument(inst)
        assert len(ticker.get_instruments()) == 1
        assert ticker.get_instrument("BTC/USDT@binance") is not None

    def test_add_duplicate_instrument(self):
        """Test adding duplicate instruments (should replace)"""
        ticker = Ticker("crypto_majors")
        inst1 = Instrument("BTC/USDT", "binance", weight=Decimal("0.5"))
        inst2 = Instrument("BTC/USDT", "binance", weight=Decimal("0.3"))

        ticker.add_instrument(inst1)
        ticker.add_instrument(inst2)
        assert len(ticker.get_instruments()) == 1
        assert ticker.get_instrument("BTC/USDT@binance").weight == Decimal("0.3")

    def test_remove_instrument(self):
        """Test removing instruments"""
        ticker = Ticker("crypto_majors")
        inst = Instrument("BTC/USDT", "binance")

        ticker.add_instrument(inst)
        assert len(ticker.get_instruments()) == 1

        ticker.remove_instrument("BTC/USDT@binance")
        assert len(ticker.get_instruments()) == 0

    def test_get_instruments_enabled_only(self):
        """Test filtering by enabled status"""
        ticker = Ticker("crypto_majors")
        inst1 = Instrument("BTC/USDT", "binance")
        inst2 = Instrument("ETH/USDT", "binance")
        inst2.disable()

        ticker.add_instrument(inst1)
        ticker.add_instrument(inst2)

        assert len(ticker.get_instruments(enabled_only=True)) == 1
        assert len(ticker.get_instruments(enabled_only=False)) == 2

    def test_get_instruments_by_tag(self):
        """Test filtering by tags"""
        ticker = Ticker("crypto_majors")
        inst1 = Instrument("BTC/USDT", "binance", tags={"major", "btc"})
        inst2 = Instrument("ETH/USDT", "binance", tags={"major", "eth"})
        inst3 = Instrument("SOL/USDT", "binance", tags={"alt"})

        ticker.add_instrument(inst1)
        ticker.add_instrument(inst2)
        ticker.add_instrument(inst3)

        majors = ticker.get_instruments(tags=["major"])
        assert len(majors) == 2

        btc = ticker.get_instruments(tags=["btc"])
        assert len(btc) == 1

    def test_get_symbols(self):
        """Test getting symbol list"""
        ticker = Ticker("crypto_majors")
        ticker.add_instrument(Instrument("BTC/USDT", "binance"))
        ticker.add_instrument(Instrument("ETH/USDT", "binance"))

        symbols = ticker.get_symbols()
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
        assert len(symbols) == 2

    def test_iteration(self):
        """Test ticker iteration"""
        ticker = Ticker("crypto_majors")
        ticker.add_instrument(Instrument("BTC/USDT", "binance"))
        ticker.add_instrument(Instrument("ETH/USDT", "binance"))

        count = 0
        for inst in ticker:
            assert isinstance(inst, Instrument)
            count += 1
        assert count == 2

    def test_ticker_in_operator(self):
        """Test 'in' operator"""
        ticker = Ticker("crypto_majors")
        inst = Instrument("BTC/USDT", "binance")
        ticker.add_instrument(inst)

        assert "BTC/USDT" in ticker
        assert "ETH/USDT" not in ticker


class TestUniverse:
    """Test Universe class"""

    def test_create_empty_universe(self):
        """Test creating empty universe"""
        universe = Universe("test_universe")
        assert universe.name == "test_universe"
        assert len(universe.get_all_instruments()) == 0
        assert len(universe.get_tickers()) == 0

    def test_add_ticker(self):
        """Test adding tickers"""
        universe = Universe("test")
        ticker = Ticker("crypto_majors")
        ticker.add_instrument(Instrument("BTC/USDT", "binance"))

        universe.add_ticker(ticker)
        assert len(universe.get_tickers()) == 1
        assert universe.get_ticker("crypto_majors") is not None

    def test_add_instrument(self):
        """Test adding individual instruments"""
        universe = Universe("test")
        inst = Instrument("BTC/USDT", "binance")

        universe.add_instrument(inst)
        instruments = universe.get_all_instruments()
        assert len(instruments) == 1
        assert instruments[0].symbol == "BTC/USDT"

    def test_get_by_tag(self):
        """Test getting instruments by tag"""
        universe = Universe("test")
        universe.add_instrument(Instrument("BTC/USDT", "binance", tags={"major"}))
        universe.add_instrument(Instrument("ETH/USDT", "binance", tags={"major"}))
        universe.add_instrument(Instrument("SOL/USDT", "binance", tags={"alt"}))

        majors = universe.get_by_tag("major")
        assert len(majors) == 2

    def test_get_by_exchange(self):
        """Test getting instruments by exchange"""
        universe = Universe("test")
        universe.add_instrument(Instrument("BTC/USDT", "binance"))
        universe.add_instrument(Instrument("BTC/USDT", "okex"))
        universe.add_instrument(Instrument("ETH/USDT", "binance"))

        binance_instruments = universe.get_by_exchange("binance")
        assert len(binance_instruments) == 2

    def test_get_by_type(self):
        """Test getting instruments by type"""
        universe = Universe("test")
        universe.add_instrument(Instrument("BTC/USDT", "binance", instrument_type=InstrumentType.SPOT))
        universe.add_instrument(Instrument("BTC/USDT:USDT", "binance", instrument_type=InstrumentType.PERPETUAL))
        universe.add_instrument(Instrument("ETH/USDT", "binance", instrument_type=InstrumentType.SPOT))

        spot_instruments = universe.get_by_type(InstrumentType.SPOT)
        assert len(spot_instruments) == 2

    def test_filter_with_kwargs(self):
        """Test filtering with keyword arguments"""
        universe = Universe("test")
        universe.add_instrument(Instrument("BTC/USDT", "binance", tags={"major"}))
        universe.add_instrument(Instrument("ETH/USDT", "binance", tags={"major"}))
        universe.add_instrument(Instrument("SOL/USDT", "okex", tags={"alt"}))

        # Filter by pattern
        btc_instruments = universe.filter(symbol_pattern="BTC/*")
        assert len(btc_instruments) == 1

        # Filter by exchange
        binance_instruments = universe.filter(exchange="binance")
        assert len(binance_instruments) == 2

        # Filter by tag
        majors = universe.filter(tags=["major"])
        assert len(majors) == 2

    def test_get_symbols(self):
        """Test getting all symbols"""
        universe = Universe("test")
        universe.add_instrument(Instrument("BTC/USDT", "binance"))
        universe.add_instrument(Instrument("ETH/USDT", "binance"))

        symbols = universe.get_symbols()
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
        assert len(symbols) == 2

    def test_universe_from_config(self):
        """Test creating universe from config"""
        config_dict = {
            "name": "crypto_portfolio",
            "tickers": [
                {
                    "id": "crypto_majors",
                    "type": "spot",
                    "exchange": "binance",
                    "tags": ["major", "high_liquidity"],
                    "instruments": ["BTC/USDT", "ETH/USDT"]
                }
            ]
        }

        config = UniverseConfig.from_dict(config_dict)
        universe = Universe.from_config(config)

        assert universe.name == "crypto_portfolio"
        assert len(universe.get_tickers()) == 1
        assert len(universe.get_all_instruments()) == 2


class TestFilters:
    """Test filter classes"""

    @pytest.fixture
    def instruments(self):
        """Create test instruments"""
        return [
            Instrument("BTC/USDT", "binance", tags={"major", "btc"}, instrument_type=InstrumentType.SPOT),
            Instrument("ETH/USDT", "binance", tags={"major", "eth"}, instrument_type=InstrumentType.SPOT),
            Instrument("BTC/USDT:USDT", "binance", tags={"major", "btc"}, instrument_type=InstrumentType.PERPETUAL),
            Instrument("SOL/USDT", "okex", tags={"alt"}, instrument_type=InstrumentType.SPOT),
            Instrument("AVAX/BTC", "binance", tags={"alt"}, instrument_type=InstrumentType.SPOT),
        ]

    def test_symbol_pattern_filter(self, instruments):
        """Test SymbolPatternFilter"""
        filter = SymbolPatternFilter("BTC/*")
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 2  # BTC/USDT and BTC/USDT:USDT

    def test_exchange_filter(self, instruments):
        """Test ExchangeFilter"""
        filter = ExchangeFilter("binance")
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 4

    def test_instrument_type_filter(self, instruments):
        """Test InstrumentTypeFilter"""
        filter = InstrumentTypeFilter(InstrumentType.SPOT)
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 4

    def test_quote_currency_filter(self, instruments):
        """Test QuoteCurrencyFilter"""
        filter = QuoteCurrencyFilter("USDT")
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 4

    def test_tag_filter_match_all(self, instruments):
        """Test TagFilter with match_all=True"""
        filter = TagFilter(["major", "btc"], match_all=True)
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 2  # BTC/USDT and BTC/USDT:USDT

    def test_tag_filter_match_any(self, instruments):
        """Test TagFilter with match_all=False"""
        filter = TagFilter(["btc", "eth"], match_all=False)
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 3  # BTC/USDT, ETH/USDT, BTC/USDT:USDT

    def test_and_filter(self, instruments):
        """Test AndFilter"""
        filter = AndFilter(
            ExchangeFilter("binance"),
            TagFilter(["major"], match_all=True),
            InstrumentTypeFilter(InstrumentType.SPOT)
        )
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 2  # BTC/USDT, ETH/USDT

    def test_or_filter(self, instruments):
        """Test OrFilter"""
        filter = OrFilter(
            SymbolPatternFilter("BTC/*"),
            SymbolPatternFilter("ETH/*")
        )
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 3  # BTC/USDT, ETH/USDT, BTC/USDT:USDT

    def test_not_filter(self, instruments):
        """Test NotFilter"""
        filter = NotFilter(TagFilter(["alt"], match_all=False))
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 3  # All major coins

    def test_complex_filter_combination(self, instruments):
        """Test complex filter combination"""
        # Find: (binance exchange AND spot) OR (quote currency BTC)
        filter = OrFilter(
            AndFilter(
                ExchangeFilter("binance"),
                InstrumentTypeFilter(InstrumentType.SPOT)
            ),
            QuoteCurrencyFilter("BTC")
        )
        filtered = [inst for inst in instruments if filter.matches(inst)]
        assert len(filtered) == 4


class TestUniverseConfig:
    """Test UniverseConfig"""

    def test_parse_simple_config(self):
        """Test parsing simple config"""
        config_dict = {
            "name": "test",
            "tickers": [
                {
                    "id": "crypto_majors",
                    "instruments": ["BTC/USDT", "ETH/USDT"]
                }
            ]
        }

        config = UniverseConfig.from_dict(config_dict)
        assert config.name == "test"
        assert len(config.tickers) == 1
        assert config.tickers[0]["id"] == "crypto_majors"

    def test_parse_complex_config(self):
        """Test parsing complex config with metadata"""
        config_dict = {
            "name": "test",
            "tickers": [
                {
                    "id": "btc_products",
                    "instruments": [
                        {
                            "symbol": "BTC/USDT",
                            "exchange": "binance",
                            "instrument_type": "spot",
                            "tags": ["major", "btc"],
                            "weight": 0.5
                        },
                        {
                            "symbol": "BTC/USDT:USDT",
                            "exchange": "binance",
                            "instrument_type": "perpetual",
                            "tags": ["major", "btc"],
                            "weight": 0.3
                        }
                    ]
                }
            ]
        }

        config = UniverseConfig.from_dict(config_dict)
        assert len(config.tickers) == 1
        ticker = config.tickers[0]
        assert len(ticker["instruments"]) == 2
        assert ticker["instruments"][0]["weight"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
