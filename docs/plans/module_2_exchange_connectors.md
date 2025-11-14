# Module 2: Exchange Connector Framework

## Purpose

Create a modular, extensible framework for connecting to multiple exchanges (crypto and traditional) for both market data and order execution.

## Design Goals

1. **Exchange-Agnostic**: Uniform interface regardless of exchange
2. **Instrument-Agnostic**: Support spot, futures, options, perpetuals
3. **SOLID Principles**: Easy to add new exchanges with minimal code
4. **Event-Driven**: Publish events to event bus
5. **Resilient**: Auto-reconnection, error handling
6. **Rate-Limit Aware**: Respect exchange limits

---

## Architecture

```
IExchange (interface)
├── MarketDataConnector (interface)
│   ├── WebSocket support
│   ├── REST API fallback
│   └── Order book management
├── ExecutionConnector (interface)
│   ├── Order placement
│   ├── Order cancellation
│   └── Position query
└── InstrumentMapper (interface)
    ├── Symbol normalization
    └── Instrument metadata
```

---

## Core Interfaces

### IExchange
```python
class ExchangeType(str, Enum):
    CEX = "centralized"  # Binance, OKEx, etc.
    DEX = "decentralized"  # Uniswap, Hyperliquid, etc.
    TRADITIONAL = "traditional"  # CME, NSE, BSE

class IExchange(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name (e.g., 'binance')"""

    @property
    @abstractmethod
    def exchange_type(self) -> ExchangeType:
        """Exchange type"""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to exchange"""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange"""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected"""

    @abstractmethod
    def get_market_data_connector(self) -> IMarketDataConnector:
        """Get market data connector"""

    @abstractmethod
    def get_execution_connector(self) -> IExecutionConnector:
        """Get execution connector"""

    @abstractmethod
    def get_instrument_mapper(self) -> IInstrumentMapper:
        """Get instrument mapper"""
```

### IMarketDataConnector
```python
class IMarketDataConnector(ABC):
    @abstractmethod
    async def subscribe_ticks(self, symbols: list[str]) -> None:
        """Subscribe to tick data"""

    @abstractmethod
    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to trade feed"""

    @abstractmethod
    async def subscribe_order_book(
        self,
        symbols: list[str],
        depth: int = 20
    ) -> None:
        """Subscribe to order book updates"""

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols"""

    @abstractmethod
    async def get_order_book_snapshot(
        self,
        symbol: str,
        depth: int = 20
    ) -> OrderBook:
        """Get current order book snapshot via REST"""

    @abstractmethod
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> list[Trade]:
        """Get recent trades via REST"""
```

### IExecutionConnector
```python
class IExecutionConnector(ABC):
    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Place an order"""

    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
        symbol: str
    ) -> CancelResult:
        """Cancel an order"""

    @abstractmethod
    async def get_order_status(
        self,
        order_id: str,
        symbol: str
    ) -> OrderStatus:
        """Get order status"""

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> list[OrderStatus]:
        """Get all open orders"""

    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> list[Position]:
        """Get current positions"""
```

### IInstrumentMapper
```python
class InstrumentType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTION = "option"

class Instrument(BaseModel):
    symbol: str  # Normalized (e.g., "BTC/USDT")
    raw_symbol: str  # Exchange-specific
    base_currency: str  # "BTC"
    quote_currency: str  # "USDT"
    instrument_type: InstrumentType
    contract_size: Optional[Decimal] = None
    tick_size: Decimal
    min_order_size: Decimal
    max_order_size: Optional[Decimal] = None
    price_precision: int
    quantity_precision: int
    metadata: dict[str, Any] = Field(default_factory=dict)

class IInstrumentMapper(ABC):
    @abstractmethod
    async def load_instruments(self) -> None:
        """Load instrument info from exchange"""

    @abstractmethod
    def normalize_symbol(self, raw_symbol: str) -> str:
        """Convert exchange symbol to normalized (e.g., BTCUSDT -> BTC/USDT)"""

    @abstractmethod
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert normalized to exchange symbol"""

    @abstractmethod
    def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument metadata"""

    @abstractmethod
    def list_instruments(
        self,
        instrument_type: Optional[InstrumentType] = None
    ) -> list[Instrument]:
        """List all available instruments"""
```

---

## Data Models

### Order
```python
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"

class TimeInForce(str, Enum):
    GTC = "gtc"  # Good till cancel
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill

class Order(BaseModel):
    client_order_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    symbol: str  # Normalized
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None  # Required for limit orders
    stop_price: Optional[Decimal] = None  # For stop orders
    time_in_force: TimeInForce = TimeInForce.GTC
    strategy_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### OrderBook
```python
class OrderBookLevel(BaseModel):
    price: Decimal
    quantity: Decimal
    num_orders: Optional[int] = None

class OrderBook(BaseModel):
    symbol: str
    exchange: str
    timestamp: datetime
    bids: list[OrderBookLevel]  # Sorted high to low
    asks: list[OrderBookLevel]  # Sorted low to high

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
```

---

## Exchange Implementations

### Reference Implementation: Binance

```
trading/exchanges/
├── __init__.py
├── base.py              # IExchange, IMarketDataConnector, etc.
├── models.py            # Order, OrderBook, Instrument, etc.
├── binance/
│   ├── __init__.py
│   ├── exchange.py      # BinanceExchange
│   ├── market_data.py   # BinanceMarketDataConnector
│   ├── execution.py     # BinanceExecutionConnector
│   ├── mapper.py        # BinanceInstrumentMapper
│   └── utils.py         # Binance-specific helpers
├── okex/
│   └── ...
├── hyperliquid/
│   └── ...
└── registry.py          # ExchangeRegistry
```

---

## Rate Limiting

### IRateLimiter Interface
```python
class IRateLimiter(ABC):
    @abstractmethod
    async def acquire(self, endpoint: str, weight: int = 1) -> None:
        """Acquire permission to make request"""

    @abstractmethod
    def get_remaining(self, endpoint: str) -> int:
        """Get remaining requests"""
```

### Token Bucket Implementation
```python
class TokenBucketRateLimiter(IRateLimiter):
    """
    Token bucket algorithm for rate limiting

    Exchange-specific limits:
    - Binance: 1200 weight/minute, 10 orders/sec
    - OKEx: 20 requests/2sec per endpoint
    - etc.
    """

    def __init__(self, limits: dict[str, RateLimit]):
        self.limits = limits
        self.buckets: dict[str, TokenBucket] = {}

    async def acquire(self, endpoint: str, weight: int = 1) -> None:
        bucket = self._get_bucket(endpoint)
        await bucket.consume(weight)
```

---

## WebSocket Connection Management

### IWebSocketConnector
```python
class IWebSocketConnector(ABC):
    @abstractmethod
    async def connect(self, url: str) -> None:
        """Connect to WebSocket"""

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None:
        """Send message"""

    @abstractmethod
    async def close(self) -> None:
        """Close connection"""

    @abstractmethod
    def set_message_handler(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Set handler for incoming messages"""
```

### Resilient WebSocket
```python
class ResilientWebSocket(IWebSocketConnector):
    """
    WebSocket with auto-reconnection and heartbeat

    Features:
    - Automatic reconnection on disconnect
    - Exponential backoff
    - Heartbeat/ping-pong
    - Message buffering during reconnection
    - Connection state callbacks
    """

    async def _reconnect_loop(self):
        backoff = 1
        while self.should_reconnect:
            try:
                await self._connect()
                backoff = 1  # Reset on success
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # Max 60s
```

---

## Implementation Plan

### Step 1: Base Interfaces (Day 1)
```
trading/exchanges/
├── __init__.py
├── base.py
└── models.py
```

**Tasks:**
- [ ] Define IExchange interface
- [ ] Define IMarketDataConnector interface
- [ ] Define IExecutionConnector interface
- [ ] Define IInstrumentMapper interface
- [ ] Define all data models (Order, OrderBook, Instrument, etc.)
- [ ] Add comprehensive docstrings

### Step 2: Rate Limiter (Day 2)
```
trading/exchanges/
└── rate_limiter.py
```

**Tasks:**
- [ ] Define IRateLimiter interface
- [ ] Implement TokenBucketRateLimiter
- [ ] Add configurable limits per exchange
- [ ] Test with various scenarios
- [ ] Performance benchmark

### Step 3: WebSocket Infrastructure (Day 2-3)
```
trading/exchanges/
├── websocket.py
└── connection_manager.py
```

**Tasks:**
- [ ] Define IWebSocketConnector interface
- [ ] Implement ResilientWebSocket
  - [ ] Auto-reconnection logic
  - [ ] Heartbeat handling
  - [ ] Message buffering
  - [ ] State callbacks
- [ ] Test reconnection scenarios
- [ ] Test message handling

### Step 4: Binance Implementation (Day 4-7)
```
trading/exchanges/binance/
├── __init__.py
├── exchange.py
├── market_data.py
├── execution.py
├── mapper.py
└── utils.py
```

**Tasks:**
- [ ] Implement BinanceExchange
- [ ] Implement BinanceMarketDataConnector
  - [ ] WebSocket tick subscription
  - [ ] WebSocket trade subscription
  - [ ] WebSocket order book (diff depth)
  - [ ] REST fallback for snapshots
- [ ] Implement BinanceExecutionConnector
  - [ ] Place order (market, limit, stop)
  - [ ] Cancel order
  - [ ] Query order status
  - [ ] Query open orders
  - [ ] Query positions (futures)
- [ ] Implement BinanceInstrumentMapper
  - [ ] Load from /exchangeInfo
  - [ ] Symbol normalization (BTCUSDT <-> BTC/USDT)
  - [ ] Instrument metadata
- [ ] Add Binance-specific rate limits
- [ ] Comprehensive testing

### Step 5: Exchange Registry (Day 8)
```
trading/exchanges/
└── registry.py
```

**Tasks:**
- [ ] Create ExchangeRegistry
- [ ] Factory pattern for exchange creation
- [ ] Configuration loading (API keys, etc.)
- [ ] Multi-exchange management
- [ ] Example usage

### Step 6: Additional Exchanges (Day 9-14)
Implement 5 more exchanges as proof of extensibility:

**Priority List:**
1. **OKEx** (Major CEX, similar to Binance)
2. **Bybit** (Derivatives focus)
3. **Hyperliquid** (DEX with unique architecture)
4. **Coinbase** (USD pairs, different API style)
5. **Kraken** (Traditional crypto exchange)

Each exchange ~1-2 days

**Tasks per exchange:**
- [ ] Market data connector
- [ ] Execution connector
- [ ] Instrument mapper
- [ ] Rate limits
- [ ] Tests

### Step 7: Traditional Exchange Preparation (Day 15)
```
trading/exchanges/
├── cme/
│   └── README.md  # Placeholder + design
└── nse/
    └── README.md  # Placeholder + design
```

**Tasks:**
- [ ] Research CME API (QuickFIX/FAST)
- [ ] Research NSE/BSE API
- [ ] Design adapter layer
- [ ] Document requirements
- [ ] Create placeholder implementations

### Step 8: Testing & Documentation (Day 16-17)
```
tests/exchanges/
├── test_base.py
├── test_rate_limiter.py
├── test_websocket.py
├── test_binance.py
├── test_okex.py
└── ...
```

**Tasks:**
- [ ] Unit tests for all interfaces
- [ ] Integration tests with mock exchange
- [ ] Live tests with testnet (if available)
- [ ] Test rate limiting under load
- [ ] Test reconnection scenarios
- [ ] Document all exchanges
- [ ] Usage examples

---

## Exchange-Specific Considerations

### Binance
- Rate limit: 1200 weight/min, 10 orders/sec
- WebSocket: User data stream requires listen key
- Futures vs Spot: Different endpoints
- Symbol format: BTCUSDT (no separator)

### OKEx
- Rate limit: Endpoint-specific
- WebSocket: Login required for private channels
- Instruments: Spot, futures, perpetual, options
- Symbol format: BTC-USDT (dash separator)

### Hyperliquid
- DEX architecture (on-chain)
- WebSocket for market data
- Orders via API (signed messages)
- No traditional API keys
- Wallet integration required

### CME (Future)
- FIX protocol
- Market data: FAST/SBE format
- Certified connectivity required
- High barrier to entry

### NSE/BSE (Future)
- REST API
- OAuth2 authentication
- Market segments: EQ, FO, CD
- Different from crypto exchanges

---

## Symbol Normalization Strategy

### Format: `BASE/QUOTE:TYPE:SETTLEMENT`

Examples:
- `BTC/USDT` - Spot
- `BTC/USDT:PERP` - Perpetual
- `BTC/USDT:FUT:20241231` - Futures expiring Dec 31, 2024
- `BTC/USDT:OPT:C:50000:20241231` - Call option, strike 50k, expires Dec 31

Exchange mappings:
- Binance: `BTCUSDT` -> `BTC/USDT`
- Binance Futures: `BTCUSDT` -> `BTC/USDT:PERP`
- OKEx: `BTC-USDT` -> `BTC/USDT`
- OKEx Futures: `BTC-USDT-241231` -> `BTC/USDT:FUT:20241231`

---

## Configuration Example

```yaml
exchanges:
  binance:
    enabled: true
    api_key: "${BINANCE_API_KEY}"
    api_secret: "${BINANCE_API_SECRET}"
    testnet: false
    rate_limits:
      orders_per_second: 10
      requests_per_minute: 1200

  okex:
    enabled: true
    api_key: "${OKEX_API_KEY}"
    api_secret: "${OKEX_API_SECRET}"
    passphrase: "${OKEX_PASSPHRASE}"
    testnet: false
```

---

## Example Usage

### Connecting to Exchange
```python
from trading.exchanges import ExchangeRegistry
from trading.exchanges.binance import BinanceExchange

# Via registry
registry = ExchangeRegistry()
await registry.load_config("config/exchanges.yaml")

binance = registry.get_exchange("binance")
await binance.connect()

# Direct instantiation
binance = BinanceExchange(
    api_key="...",
    api_secret="..."
)
await binance.connect()
```

### Subscribing to Market Data
```python
market_data = binance.get_market_data_connector()

# Subscribe to ticks
await market_data.subscribe_ticks(["BTC/USDT", "ETH/USDT"])

# Subscribe to order book
await market_data.subscribe_order_book(["BTC/USDT"], depth=20)

# Get snapshot
order_book = await market_data.get_order_book_snapshot("BTC/USDT")
print(f"Best bid: {order_book.best_bid}")
print(f"Best ask: {order_book.best_ask}")
```

### Placing Orders
```python
execution = binance.get_execution_connector()

# Market order
order = Order(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("0.001")
)

result = await execution.place_order(order)
print(f"Order ID: {result.order_id}")

# Limit order
order = Order(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=Decimal("0.001"),
    price=Decimal("50000")
)

result = await execution.place_order(order)
```

---

## Performance Targets

- **WebSocket latency**: < 10ms tick-to-event
- **Order placement**: < 50ms (network permitting)
- **Rate limiter overhead**: < 0.1ms
- **Memory**: < 50MB per exchange connection
- **Reconnection time**: < 5s

---

## Dependencies

- `ccxt` - Reference for exchange APIs (optional, for comparison)
- `websockets` - WebSocket client
- `aiohttp` - HTTP client
- `cryptography` - Signing requests
- `pydantic` - Data validation

---

## Success Criteria

- [ ] At least 6 exchanges implemented (Binance, OKEx, Bybit, Hyperliquid, Coinbase, Kraken)
- [ ] Uniform interface across all exchanges
- [ ] Rate limiting works correctly
- [ ] WebSocket auto-reconnection works
- [ ] Can subscribe to market data and receive events
- [ ] Can place/cancel orders on all exchanges
- [ ] Symbol normalization works bidirectionally
- [ ] Comprehensive test coverage (>85%)
- [ ] Documentation with examples

---

**Status**: Ready to implement (depends on Module 1)
**Priority**: Critical
**Estimated Time**: 17 days
**LOC**: ~5000-6000 lines
