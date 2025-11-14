# Module 1: Event System Foundation

## Purpose

Create a unified, high-performance event system that serves as the backbone for all communication between components (market data, orders, portfolio, strategies).

## Design Principles

1. **Type-Safe Events**: Pydantic models for all event types
2. **Async Event Bus**: Non-blocking event delivery
3. **Event Persistence**: Optional replay capability
4. **Topic-Based Routing**: Subscribe to specific event types/symbols
5. **Performance**: Handle 1000+ events/second

---

## Event Types

### Core Event Hierarchy

```python
BaseEvent (abstract)
├── MarketDataEvent (abstract)
│   ├── TickEvent (bid/ask/last price)
│   ├── TradeEvent (actual trade executed on exchange)
│   ├── OrderBookEvent (full order book snapshot/delta)
│   ├── BarEvent (OHLCV aggregated)
│   └── QuoteEvent (best bid/ask)
├── OrderEvent (abstract)
│   ├── OrderSubmittedEvent
│   ├── OrderAcceptedEvent
│   ├── OrderRejectedEvent
│   ├── OrderCancelledEvent
│   ├── OrderExpiredEvent
│   └── OrderFilledEvent (full or partial)
├── PositionEvent
│   ├── PositionOpenedEvent
│   ├── PositionModifiedEvent
│   └── PositionClosedEvent
├── PortfolioEvent
│   ├── PortfolioValueEvent
│   └── RiskLimitEvent
└── SystemEvent
    ├── ConnectionEvent
    ├── ErrorEvent
    └── HeartbeatEvent
```

---

## Event Schema Design

### BaseEvent
```python
class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exchange: Optional[str] = None
    symbol: Optional[str] = None  # Normalized symbol
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### MarketDataEvent
```python
class TickEvent(BaseEvent):
    event_type: Literal["tick"] = "tick"
    exchange: str
    symbol: str  # Normalized (e.g., "BTC/USDT")
    raw_symbol: str  # Exchange-specific (e.g., "BTCUSDT")

    bid: Optional[Decimal]
    ask: Optional[Decimal]
    last: Optional[Decimal]
    volume: Optional[Decimal]

    exchange_timestamp: datetime  # Exchange time
    local_timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### OrderEvent
```python
class OrderSubmittedEvent(BaseEvent):
    event_type: Literal["order_submitted"] = "order_submitted"
    order_id: str
    client_order_id: str
    exchange: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: str  # "market", "limit", "stop_limit", etc.
    quantity: Decimal
    price: Optional[Decimal] = None
    strategy_id: Optional[str] = None
```

### PositionEvent
```python
class PositionModifiedEvent(BaseEvent):
    event_type: Literal["position_modified"] = "position_modified"
    position_id: str
    exchange: str
    symbol: str
    side: Literal["long", "short"]
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
```

---

## Event Bus Architecture

### Interface
```python
class IEventBus(ABC):
    @abstractmethod
    async def publish(self, event: BaseEvent) -> None:
        """Publish event to all subscribers"""

    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[BaseEvent], Awaitable[None]],
        filter: Optional[Callable[[BaseEvent], bool]] = None
    ) -> str:
        """Subscribe to events with optional filter. Returns subscription ID."""

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""

    @abstractmethod
    async def publish_batch(self, events: list[BaseEvent]) -> None:
        """Publish multiple events efficiently"""
```

### Implementation Options

**Option 1: In-Memory Event Bus (Development/Testing)**
- Fast, simple
- No persistence
- Single process only

**Option 2: Redis Pub/Sub (Production)**
- Multi-process support
- Can persist for replay
- Good performance
- Easy to monitor

**Option 3: Hybrid**
- In-memory for speed
- Redis for persistence and multi-process
- Best of both worlds

---

## Event Filtering & Routing

### Topic Structure
```
events.{event_type}.{exchange}.{symbol}

Examples:
events.tick.binance.BTC/USDT
events.order_filled.okex.*
events.*.binance.*
events.position_modified.*.*
```

### Filter Functions
```python
def symbol_filter(symbol: str) -> Callable[[BaseEvent], bool]:
    return lambda event: event.symbol == symbol

def exchange_filter(exchange: str) -> Callable[[BaseEvent], bool]:
    return lambda event: event.exchange == exchange

def combined_filter(*filters) -> Callable[[BaseEvent], bool]:
    return lambda event: all(f(event) for f in filters)
```

---

## Event Persistence

### Requirements
- Optional persistence for simulation replay
- Fast write (don't block event bus)
- Efficient read for replay
- Query by time range, symbol, exchange

### Storage Format

**Option 1: Parquet Files** (RECOMMENDED)
- Columnar storage
- Excellent compression
- Fast queries
- Easy to work with pandas/polars

```
data/events/
  2024/
    01/
      01/
        tick_binance_BTCUSDT.parquet
        order_*.parquet
```

**Option 2: InfluxDB**
- Purpose-built for time series
- Good query performance
- Retention policies
- Downsampling

**Option 3: Arctic (MongoDB-based)**
- Built for financial data
- Versioning support
- Good compression

### Writer Implementation
```python
class EventWriter(ABC):
    @abstractmethod
    async def write_event(self, event: BaseEvent) -> None:
        """Write single event"""

    @abstractmethod
    async def write_batch(self, events: list[BaseEvent]) -> None:
        """Write multiple events efficiently"""

    @abstractmethod
    async def flush(self) -> None:
        """Ensure all events written to storage"""
```

---

## Implementation Plan

### Step 1: Core Event Models (Day 1)
```
trading/events/
├── __init__.py
├── base.py              # BaseEvent
├── market_data.py       # All market data events
├── order.py             # All order events
├── position.py          # Position events
├── portfolio.py         # Portfolio events
└── system.py            # System events
```

**Tasks:**
- [ ] Define BaseEvent with all common fields
- [ ] Create MarketDataEvent hierarchy
- [ ] Create OrderEvent hierarchy
- [ ] Create PositionEvent hierarchy
- [ ] Create PortfolioEvent and SystemEvent
- [ ] Add comprehensive docstrings
- [ ] Validate with Pydantic

### Step 2: Event Bus Implementation (Day 2-3)
```
trading/events/
├── bus.py               # IEventBus interface
├── memory_bus.py        # In-memory implementation
└── redis_bus.py         # Redis pub/sub implementation
```

**Tasks:**
- [ ] Define IEventBus interface
- [ ] Implement InMemoryEventBus
  - [ ] Topic-based routing
  - [ ] Wildcard subscriptions
  - [ ] Async handler execution
  - [ ] Error handling (don't crash bus on handler error)
- [ ] Implement RedisEventBus
  - [ ] Use aioredis
  - [ ] Serialize events with msgpack
  - [ ] Pattern-based subscriptions
  - [ ] Reconnection logic
- [ ] Add event bus factory
- [ ] Thread-safety considerations

### Step 3: Event Filters (Day 3)
```
trading/events/
└── filters.py           # Predefined filters
```

**Tasks:**
- [ ] symbol_filter
- [ ] exchange_filter
- [ ] event_type_filter
- [ ] time_range_filter
- [ ] combined_filter (AND/OR logic)
- [ ] Custom filter examples

### Step 4: Event Persistence (Day 4-5)
```
trading/events/
├── persistence/
│   ├── __init__.py
│   ├── base.py          # IEventWriter interface
│   ├── parquet_writer.py
│   ├── influx_writer.py (optional)
│   └── reader.py        # EventReader for replay
```

**Tasks:**
- [ ] Define IEventWriter interface
- [ ] Implement ParquetEventWriter
  - [ ] Partition by date/exchange/symbol
  - [ ] Batch writes for performance
  - [ ] Background flushing
  - [ ] Rotation policy
- [ ] Implement EventReader
  - [ ] Read by time range
  - [ ] Read by symbol/exchange
  - [ ] Iterator interface
  - [ ] Replay with original timing
- [ ] Add configuration for storage paths

### Step 5: Testing (Day 6)
```
tests/events/
├── test_event_models.py
├── test_memory_bus.py
├── test_redis_bus.py
├── test_filters.py
└── test_persistence.py
```

**Tasks:**
- [ ] Test all event models serialize/deserialize correctly
- [ ] Test event bus publish/subscribe
- [ ] Test wildcard subscriptions
- [ ] Test filtering
- [ ] Test event persistence and replay
- [ ] Performance benchmark (1000+ events/sec)
- [ ] Test error handling (bad handler, connection loss)

---

## Example Usage

### Publishing Events
```python
from trading.events import TickEvent, InMemoryEventBus

bus = InMemoryEventBus()

# Publish tick event
tick = TickEvent(
    exchange="binance",
    symbol="BTC/USDT",
    raw_symbol="BTCUSDT",
    bid=Decimal("50000.5"),
    ask=Decimal("50000.6"),
    last=Decimal("50000.55"),
    exchange_timestamp=datetime.utcnow()
)

await bus.publish(tick)
```

### Subscribing to Events
```python
async def handle_tick(event: TickEvent):
    print(f"Tick: {event.symbol} @ {event.last}")

# Subscribe to all ticks
sub_id = await bus.subscribe("tick", handle_tick)

# Subscribe to BTC ticks only
sub_id = await bus.subscribe(
    "tick",
    handle_tick,
    filter=lambda e: e.symbol == "BTC/USDT"
)

# Subscribe to all events from Binance
sub_id = await bus.subscribe(
    "*",
    handle_all_binance,
    filter=lambda e: e.exchange == "binance"
)
```

### Event Persistence & Replay
```python
from trading.events.persistence import ParquetEventWriter, EventReader

# Record events
writer = ParquetEventWriter(base_path="data/events")
await bus.subscribe("*", writer.write_event)  # Record all events

# Later, replay events
reader = EventReader(base_path="data/events")
async for event in reader.read_range(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 2),
    symbol="BTC/USDT"
):
    await bus.publish(event)  # Replay with original timing
```

---

## Performance Targets

- **Throughput**: 10,000+ events/second
- **Latency**: < 1ms publish-to-handler
- **Memory**: < 100MB for 1M events in memory
- **Persistence**: < 10ms batch write latency

---

## Dependencies

- `pydantic` - Event validation
- `aioredis` - Redis pub/sub
- `msgpack` - Fast serialization
- `pyarrow` / `pandas` - Parquet I/O
- `asyncio` - Event loop

---

## Success Criteria

- [ ] All event types defined and validated
- [ ] Event bus handles 10k+ events/sec
- [ ] Wildcard subscriptions work correctly
- [ ] Events can be persisted and replayed
- [ ] Comprehensive test coverage (>90%)
- [ ] Example usage documented
- [ ] Zero event loss in normal operation

---

**Status**: Ready to implement
**Priority**: Critical (blocking all other modules)
**Estimated Time**: 6 days
**LOC**: ~1500 lines
