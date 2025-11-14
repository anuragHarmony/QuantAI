# Module 2: Exchange Connectors (Market Data)

## 🎯 Goal
Build modular, extensible exchange connectors for real-time market data from crypto and traditional exchanges.

## 📋 Detailed Tasks

### Task 2.1: Abstract Connector Interfaces
**Duration:** 3-4 hours

#### Subtasks:
- [ ] **2.1.1** Base market data connector protocol
  ```python
  class IMarketDataConnector(Protocol):
      async def connect(self) -> None
      async def disconnect(self) -> None
      async def subscribe_ticker(self, symbol: str) -> None
      async def subscribe_orderbook(self, symbol: str, depth: int = 20) -> None
      async def subscribe_trades(self, symbol: str) -> None
      async def subscribe_candles(self, symbol: str, interval: str) -> None
  ```

- [ ] **2.1.2** Symbol normalization interface
  ```python
  class ISymbolMapper(Protocol):
      def to_exchange_symbol(self, universal_symbol: str) -> str
      def to_universal_symbol(self, exchange_symbol: str) -> str
  ```

- [ ] **2.1.3** Market data types (domain models)
  - `Ticker`: bid/ask/last price, volume
  - `OrderBook`: bids/asks with depth
  - `Trade`: price, size, side, timestamp
  - `Candle`: OHLCV + timestamp

**Deliverables:**
- `trading/domain/models/market_data.py`
- `trading/infrastructure/exchanges/interfaces.py`

---

### Task 2.2: WebSocket Base Implementation
**Duration:** 6-8 hours

#### Subtasks:
- [ ] **2.2.1** Async WebSocket client
  - Using `websockets` or `aiohttp`
  - Auto-reconnection logic
  - Heartbeat/ping-pong handling
  - Message queue for processing

- [ ] **2.2.2** Connection manager
  - Connection state machine (Disconnected → Connecting → Connected → Reconnecting)
  - Exponential backoff for retries
  - Graceful shutdown
  - Health monitoring

- [ ] **2.2.3** Message parser base class
  - JSON parsing
  - Schema validation
  - Error handling
  - Message routing to handlers

**Deliverables:**
- `trading/infrastructure/exchanges/base/websocket_client.py`
- `trading/infrastructure/exchanges/base/connection_manager.py`
- `trading/infrastructure/exchanges/base/message_parser.py`

---

### Task 2.3: Crypto Exchange Connectors
**Duration:** 12-16 hours (distributed over multiple days)

#### Priority Tier 1 (First to implement):
- [ ] **2.3.1** Binance Spot & Futures
  - Public WebSocket streams
  - Ticker, orderbook, trades, klines
  - Symbol mapping
  - Rate limiting awareness

- [ ] **2.3.2** OKX (OKEx)
  - Public WebSocket API
  - All market data types
  - Multi-symbol subscriptions

- [ ] **2.3.3** Bybit
  - Spot and derivatives
  - WebSocket v5 API
  - Unified data format

#### Priority Tier 2 (Next batch):
- [ ] **2.3.4** Hyperliquid
- [ ] **2.3.5** Kraken
- [ ] **2.3.6** Coinbase
- [ ] **2.3.7** Bitfinex
- [ ] **2.3.8** HTX (Huobi)

#### Priority Tier 3 (Additional):
- [ ] **2.3.9** KuCoin, Gate.io, Bitget, MEXC, dYdX
- [ ] **2.3.10** DEXs: Uniswap, PancakeSwap (via The Graph or direct RPC)

**Structure for each connector:**
```
trading/infrastructure/exchanges/crypto/binance/
├── __init__.py
├── connector.py          # Main connector implementation
├── parsers.py           # Message parsers
├── symbols.py           # Symbol mapper
├── config.py            # Exchange-specific config
└── models.py            # Exchange-specific models
```

**Deliverables per exchange:**
- Connector implementation
- Parser for each message type
- Symbol mapper
- Unit tests with mock data
- Integration test (manual, requires API keys)

---

### Task 2.4: Data Normalization Layer
**Duration:** 4-6 hours

#### Subtasks:
- [ ] **2.4.1** Normalizer interface
  ```python
  class IDataNormalizer(Protocol):
      def normalize_ticker(self, raw: dict, exchange: str) -> Ticker
      def normalize_orderbook(self, raw: dict, exchange: str) -> OrderBook
      def normalize_trade(self, raw: dict, exchange: str) -> Trade
  ```

- [ ] **2.4.2** Universal timestamp handling
  - Convert all timestamps to UTC datetime
  - Handle milliseconds vs seconds
  - Exchange time vs local time

- [ ] **2.4.3** Price and size normalization
  - Decimal precision
  - Quote vs base currency
  - Contract multipliers for futures

- [ ] **2.4.4** Symbol standardization
  - Universal format: `BTC/USDT`, `ETH/USDT:PERP`
  - Exchange-specific mappings
  - Instrument type detection

**Deliverables:**
- `trading/infrastructure/exchanges/normalizers/`
- Normalizer for each exchange
- Comprehensive tests

---

### Task 2.5: Recording Infrastructure
**Duration:** 4-5 hours

#### Subtasks:
- [ ] **2.5.1** Market data recorder interface
  ```python
  class IMarketDataRecorder(Protocol):
      async def record_tick(self, tick: Ticker) -> None
      async def record_orderbook(self, book: OrderBook) -> None
      async def record_trade(self, trade: Trade) -> None
  ```

- [ ] **2.5.2** File-based recorder
  - Parquet format for efficiency
  - Partitioned by date and symbol
  - Compression (snappy or gzip)

- [ ] **2.5.3** Database recorder (optional)
  - TimescaleDB or ClickHouse for high-performance
  - Async inserts with batching

- [ ] **2.5.4** Recording coordinator
  - Start/stop recording
  - Multiple exchanges simultaneously
  - Disk space monitoring

**Deliverables:**
- `trading/infrastructure/recording/market_data_recorder.py`
- Parquet writer
- Example replay scripts

---

## 🏗️ Architecture

### Connector Architecture
```
┌────────────────────────────────────────────────┐
│           Exchange WebSocket API                │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│         WebSocket Client (Base)                   │
│  ┌─────────────────────────────────────────┐    │
│  │  Connection Manager │ Auto-Reconnect    │    │
│  └─────────────────────────────────────────┘    │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│      Exchange-Specific Parser                     │
│  ┌─────────────────────────────────────────┐    │
│  │  JSON → Exchange Models                  │    │
│  └─────────────────────────────────────────┘    │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│           Data Normalizer                         │
│  ┌─────────────────────────────────────────┐    │
│  │  Exchange Models → Universal Models      │    │
│  └─────────────────────────────────────────┘    │
└──────────────────┬───────────────────────────────┘
                   │
                   ├───▶ Event Bus (MarketDataEvent)
                   │
                   └───▶ Recorder (Parquet/DB)
```

### Factory Pattern
```python
class ConnectorFactory:
    @staticmethod
    def create_connector(exchange: str, **config) -> IMarketDataConnector:
        if exchange == "binance":
            return BinanceConnector(**config)
        elif exchange == "okx":
            return OKXConnector(**config)
        # ... etc
```

---

## 🧪 Testing Strategy

### Unit Tests (Per Connector)
- Message parsing with sample data
- Symbol mapping
- Error handling
- Connection state machine

### Integration Tests
- Real WebSocket connection (manual)
- Subscribe to multiple symbols
- Handle reconnection
- Data recording

### Performance Tests
- 1000+ symbols simultaneously
- Message processing latency
- Memory usage over 24 hours

---

## 📦 Dependencies

```toml
# WebSocket and async HTTP
websockets = "^12.0"
aiohttp = "^3.9.0"

# Data handling
pyarrow = "^14.0.0"      # For Parquet
pandas = "^2.2.0"

# Exchange-specific (install as needed)
ccxt = "^4.2.0"          # Optional: for REST API fallback
```

---

## 🎯 Success Criteria

- [ ] 3+ crypto exchanges fully integrated (Binance, OKX, Bybit)
- [ ] Unified data models across all exchanges
- [ ] Auto-reconnection works reliably
- [ ] Can handle 1000+ symbols per exchange
- [ ] Data recording at >10K ticks/sec
- [ ] <5ms p99 message processing latency
- [ ] Memory stable over 24+ hours
- [ ] Comprehensive test coverage

---

## 📝 Implementation Order

### Day 1: Foundation
- Morning: Interfaces and base WebSocket client
- Afternoon: Connection manager and message parser

### Day 2-3: First Exchange (Binance)
- Day 2 AM: Binance WebSocket integration
- Day 2 PM: Binance parsers and symbol mapper
- Day 3 AM: Normalization and testing
- Day 3 PM: Recording infrastructure

### Day 4: Second Exchange (OKX)
- Copy structure from Binance
- Implement OKX-specific parsers
- Test and validate

### Day 5: Third Exchange (Bybit) + Polish
- Implement Bybit
- Refactor common patterns
- Documentation

---

## 🔗 Integration Points

**Produces:**
- `TickEvent` → Event Bus
- `OrderBookEvent` → Event Bus
- `TradeEvent` → Event Bus
- Raw data → Recorder

**Consumes:**
- Configuration (exchange name, API keys, symbols)
- Control commands (start/stop, subscribe/unsubscribe)

**Dependencies:**
- Event Bus (Module 1)
- Event Store (Module 1)
- Domain models (Module 1)

---

## 📚 Exchange Documentation

### Key APIs to Study:
- **Binance:** https://binance-docs.github.io/apidocs/spot/en/
- **OKX:** https://www.okx.com/docs-v5/en/
- **Bybit:** https://bybit-exchange.github.io/docs/v5/intro
- **Hyperliquid:** https://hyperliquid.gitbook.io/hyperliquid-docs/
- **Coinbase:** https://docs.cloud.coinbase.com/exchange/docs

---

## 🚀 Future Extensions

### Traditional Exchanges (Phase 3)
- CME (via CQG or Rithmic)
- Indian exchanges (NSE, BSE via NSE API)
- Interactive Brokers (via IB Gateway)

### Additional Instruments
- Options (chain data, Greeks)
- Futures (open interest, funding rates)
- Perpetuals (funding rates)

---

*Status: Ready for implementation*
*Estimated Total Time: 4-5 days*
