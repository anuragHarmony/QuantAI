# Universe and Ticker System

## Overview

The **Universe and Ticker System** provides a sophisticated way to manage and organize trading instruments. Instead of just listing symbols, you can now:

- **Group related instruments** into tickers (e.g., "crypto_majors", "btc_products")
- **Tag instruments** for easy filtering (e.g., "high_liquidity", "major")
- **Assign weights** for portfolio allocation
- **Filter dynamically** using patterns and criteria
- **Access instrument metadata** (type, tags, constraints)

This system follows SOLID principles and provides a clean, scalable way to manage instruments across strategies.

## Key Concepts

### 1. Instrument
A single tradeable asset with metadata:
```python
Instrument(
    symbol="BTC/USDT",
    exchange="binance",
    instrument_type=InstrumentType.SPOT,
    tags={"major", "high_liquidity"},
    weight=Decimal("0.5"),  # 50% allocation
)
```

### 2. Ticker
A group of related instruments:
```python
ticker = Ticker(
    ticker_id="crypto_majors",
    instruments=[btc, eth, bnb],
    description="Top 3 cryptocurrencies"
)
```

### 3. Universe
Collection of all instruments and tickers:
```python
universe = Universe(name="my_portfolio")
universe.add_ticker(crypto_majors)
universe.add_ticker(crypto_alts)
```

## Configuration Format

### Basic Example

```yaml
universe:
  name: "crypto_portfolio"

  tickers:
    - id: "crypto_majors"
      instruments:
        - "BTC/USDT"
        - "ETH/USDT"
        - "BNB/USDT"

    - id: "crypto_alts"
      instruments:
        - "SOL/USDT"
        - "AVAX/USDT"
```

### Advanced Example with Metadata

```yaml
universe:
  name: "advanced_portfolio"

  tickers:
    # Ticker with full metadata
    - id: "btc_products"
      description: "All BTC trading products"
      type: "multi"
      exchange: "binance"
      tags: ["btc", "flagship"]
      instruments:
        # Simple symbol strings
        - "BTC/USDT"
        - "BTC/USD"

        # Full instrument definition
        - symbol: "BTC/USDT:USDT"
          exchange: "binance"
          instrument_type: "perpetual"
          tags: ["btc", "perp", "futures"]
          weight: 0.3
          enabled: true

    # Layer 1 blockchains with weights
    - id: "layer1_coins"
      instruments:
        - symbol: "ETH/USDT"
          tags: ["layer1", "defi", "major"]
          weight: 0.4
        - symbol: "SOL/USDT"
          tags: ["layer1", "high_tps"]
          weight: 0.3
        - symbol: "AVAX/USDT"
          tags: ["layer1", "evm"]
          weight: 0.2
        - symbol: "NEAR/USDT"
          tags: ["layer1", "sharding"]
          weight: 0.1

  # Individual instruments (optional)
  instruments:
    - symbol: "ETH/BTC"
      exchange: "binance"
      tags: ["cross_pair"]
      enabled: true

  # Filters for dynamic selection
  filters:
    - type: "quote_currency"
      quote_currencies: ["USDT"]
    - type: "enabled"
      enabled: true
```

## Usage in Strategies

### 1. Accessing Tickers

```python
class MyStrategy(BaseStrategy):
    async def on_tick(self, tick: TickEvent):
        # Get specific ticker
        majors = self.get_ticker("crypto_majors")

        if majors:
            for instrument in majors:
                print(f"{instrument.symbol}: {instrument.tags}")

                # Check if current tick is for this instrument
                if tick.symbol == instrument.symbol:
                    # Trade this instrument
                    pass
```

### 2. Filtering by Tags

```python
async def on_start(self):
    # Get all "major" instruments
    majors = self.get_instruments_by_tag("major")

    for instrument in majors:
        print(f"Major: {instrument.symbol}, Weight: {instrument.weight}")

    # Get all high liquidity instruments
    high_liq = self.get_instruments_by_tag("high_liquidity")
```

### 3. Pattern Filtering

```python
async def on_tick(self, tick: TickEvent):
    # Get all BTC products
    btc_products = self.filter_universe(symbol_pattern="BTC/*")

    # Get all USDT pairs
    usdt_pairs = self.filter_universe(tags=["usdt_pair"])

    # Get spot instruments only
    from trading.universe import InstrumentType
    spot_only = self.filter_universe(
        instrument_type=[InstrumentType.SPOT]
    )
```

### 4. Iterating Over Instruments

```python
async def on_start(self):
    # Get all instruments in universe
    all_instruments = self.get_all_instruments()

    for instrument in all_instruments:
        print(f"{instrument.symbol} @ {instrument.exchange}")
        print(f"  Type: {instrument.instrument_type.value}")
        print(f"  Tags: {instrument.tags}")
        print(f"  Weight: {instrument.weight}")
```

### 5. Using Weights for Allocation

```python
async def on_tick(self, tick: TickEvent):
    # Get instrument
    instruments = self.get_all_instruments()
    instrument = next((i for i in instruments if i.symbol == tick.symbol), None)

    if instrument and instrument.weight:
        # Use weight for position sizing
        total_capital = Decimal("100000")
        position_size = instrument.weight * total_capital

        await self.submit_order(
            symbol=tick.symbol,
            side=OrderSide.BUY,
            quantity=position_size / tick.last  # Convert to quantity
        )
```

### 6. Dynamic Ticker Selection

```python
class RotationStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_ticker = "crypto_majors"

    async def rotate_ticker(self):
        """Switch to next ticker"""
        tickers = self.universe.get_ticker_ids()
        current_idx = tickers.index(self.current_ticker)
        next_idx = (current_idx + 1) % len(tickers)
        self.current_ticker = tickers[next_idx]

        logger.info(f"Switched to ticker: {self.current_ticker}")

    async def on_tick(self, tick: TickEvent):
        # Only trade current ticker
        ticker = self.get_ticker(self.current_ticker)
        if ticker and tick.symbol in ticker:
            # Trade this instrument
            pass
```

## Available Filter Types

### Symbol Pattern Filter
```yaml
# In config filters
- type: "symbol_pattern"
  pattern: "BTC/*"  # Matches BTC/USDT, BTC/USD, etc.
```

```python
# In strategy code
btc_instruments = self.filter_universe(symbol_pattern="BTC/*")
```

### Exchange Filter
```yaml
- type: "exchange"
  exchanges: ["binance", "okex"]
```

```python
binance_instruments = self.filter_universe(exchange=["binance"])
```

### Instrument Type Filter
```yaml
- type: "instrument_type"
  instrument_types: ["spot", "perpetual"]
```

```python
from trading.universe import InstrumentType
spot = self.filter_universe(instrument_type=[InstrumentType.SPOT])
```

### Quote Currency Filter
```yaml
- type: "quote_currency"
  quote_currencies: ["USDT", "USD"]
```

```python
usdt_pairs = self.filter_universe(quote_currency=["USDT"])
```

### Tag Filter
```yaml
- type: "tags"
  tags: ["major", "high_liquidity"]
  match_all: true  # Must have ALL tags (false = ANY tag)
```

```python
majors = self.filter_universe(tags=["major"])
```

## Complete Strategy Example

```python
class UniverseStrategy(BaseStrategy):
    """
    Strategy using universe system for multi-asset trading
    """

    async def on_start(self):
        """Log universe information on start"""
        logger.info(f"Universe: {len(self.universe)} instruments")

        # Log all tickers
        for ticker_id in self.universe.get_ticker_ids():
            ticker = self.universe.get_ticker(ticker_id)
            logger.info(f"  {ticker_id}: {ticker.get_symbols()}")

    async def on_tick(self, tick: TickEvent):
        """Trade based on ticker membership and tags"""

        # Example 1: Trade majors differently than alts
        majors_ticker = self.get_ticker("crypto_majors")

        if majors_ticker and tick.symbol in majors_ticker:
            await self._trade_major(tick)
        else:
            await self._trade_alt(tick)

    async def _trade_major(self, tick: TickEvent):
        """Trade major cryptocurrencies"""
        # More aggressive parameters for majors
        position_size = Decimal("1.0")

        # Get instrument for weight
        instruments = self.get_all_instruments()
        instrument = next((i for i in instruments if i.symbol == tick.symbol), None)

        if instrument and instrument.weight:
            position_size = instrument.weight * Decimal("10.0")

        # Your trading logic here
        pass

    async def _trade_alt(self, tick: TickEvent):
        """Trade altcoins"""
        # More conservative parameters for alts
        position_size = Decimal("0.5")
        # Your trading logic here
        pass
```

## API Reference

### Universe Methods

```python
# Get instruments
universe.get_all_instruments()  # All instruments
universe.get_symbols()  # List of symbols
universe.get_instrument(symbol, exchange)  # Specific instrument

# Tickers
universe.get_ticker(ticker_id)  # Get specific ticker
universe.get_all_tickers()  # All tickers
universe.get_ticker_ids()  # List of ticker IDs

# Filtering
universe.get_by_exchange(exchange)
universe.get_by_tag(tag)
universe.get_by_type(instrument_type)
universe.filter(**kwargs)  # Dynamic filtering

# Weight management
universe.normalize_weights()  # Normalize all weights to sum to 1.0
universe.get_weight_distribution()  # Dict of symbol -> weight
```

### Ticker Methods

```python
# Get instruments
ticker.get_instruments()  # All instruments
ticker.get_symbols()  # List of symbols
ticker.get_instrument(symbol, exchange)  # Specific instrument

# Filtering
ticker.get_by_tag(tag)
ticker.get_by_type(instrument_type)
ticker.get_weighted_instruments()  # Only instruments with weights

# Weight management
ticker.normalize_weights()
ticker.get_weight_distribution()

# Iteration
for instrument in ticker:
    # Process each instrument
    pass
```

### Strategy Helper Methods

```python
# In BaseStrategy
self.get_ticker(ticker_id)  # Get specific ticker
self.get_instruments_by_tag(tag)  # Filter by tag
self.filter_universe(**kwargs)  # Dynamic filtering
self.get_all_instruments()  # All instruments
self.get_universe_symbols()  # All symbols
```

## Migration from Legacy Subscriptions

### Old Way (Still Supported)
```yaml
subscriptions:
  exchanges: ["binance"]
  symbols: ["BTC/USDT", "ETH/USDT"]
  data_types: ["tick"]
```

### New Way (Recommended)
```yaml
universe:
  tickers:
    - id: "my_assets"
      instruments:
        - "BTC/USDT"
        - "ETH/USDT"

subscriptions:
  data_types: ["tick"]  # Applied to all universe instruments
```

## Benefits

1. **Organization**: Group related instruments logically
2. **Flexibility**: Easy to add/remove instruments without code changes
3. **Metadata**: Rich information about each instrument
4. **Filtering**: Dynamic instrument selection
5. **Weights**: Built-in portfolio allocation support
6. **Scalability**: Easily manage hundreds of instruments
7. **Reusability**: Share universe configs across strategies

## Best Practices

1. **Use tickers** to organize instruments by category
2. **Add tags** for flexible filtering
3. **Assign weights** for portfolio allocation strategies
4. **Use filters** for dynamic instrument selection
5. **Start small** and expand your universe gradually
6. **Document** your ticker/tag naming conventions

## Examples

See these files for complete examples:
- `configs/strategies/universe_example.yaml` - Basic universe config
- `configs/strategies/advanced_universe_example.yaml` - Advanced features
- `examples/universe_strategy_example.py` - Complete strategy implementation