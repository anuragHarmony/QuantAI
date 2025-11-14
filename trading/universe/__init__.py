"""
Universe and Ticker Management

Provides sophisticated instrument grouping and selection:

- Universe: Collection of all instruments a strategy works with
- Ticker: Group of related instruments (e.g., "crypto_majors", "btc_products")
- Instrument: Single tradeable with metadata

Features:
- Tag-based filtering
- Weight-based portfolio allocation
- Dynamic instrument selection
- Hierarchical grouping

Usage:
    # Define in config
    universe:
      tickers:
        - id: "majors"
          instruments: ["BTC/USDT", "ETH/USDT"]

    # Query in strategy
    majors = self.universe.get_ticker("majors")
    for instrument in majors:
        # Process instrument
"""

from .instrument import Instrument, InstrumentType
from .ticker import Ticker
from .universe import Universe, UniverseConfig
from .filters import InstrumentFilter, create_filter

__all__ = [
    "Instrument",
    "InstrumentType",
    "Ticker",
    "Universe",
    "UniverseConfig",
    "InstrumentFilter",
    "create_filter",
]
