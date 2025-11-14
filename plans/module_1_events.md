# Module 1: Event System & Domain Models

## 🎯 Goal
Build the foundational event-driven architecture that powers the entire trading system.

## 📋 Detailed Tasks

### Task 1.1: Event Bus Architecture
**Duration:** 4-6 hours

#### Subtasks:
- [ ] **1.1.1** Design async event bus interface
  ```python
  class IEventBus(Protocol):
      async def publish(self, event: DomainEvent) -> None
      async def subscribe(self, event_type: Type[DomainEvent], handler: EventHandler) -> None
  ```

- [ ] **1.1.2** Implement in-memory event bus
  - Async event queue (asyncio.Queue)
  - Event routing by type
  - Multiple subscriber support
  - Error handling and retry logic

- [ ] **1.1.3** Add priority handling
  - Critical events (order fills) go first
  - Market data can be throttled

**Deliverables:**
- `trading/infrastructure/messaging/event_bus.py`
- `trading/infrastructure/messaging/interfaces.py`
- Unit tests

---

### Task 1.2: Domain Events
**Duration:** 6-8 hours

#### Subtasks:
- [ ] **1.2.1** Base event class
  ```python
  @dataclass(frozen=True)
  class DomainEvent:
      event_id: str
      timestamp: datetime
      event_type: str
  ```

- [ ] **1.2.2** Market data events
  - `TickEvent` (price update)
  - `OrderBookEvent` (depth snapshot)
  - `TradeEvent` (executed trade on exchange)
  - `CandleEvent` (OHLCV candle)

- [ ] **1.2.3** Order events
  - `OrderSubmittedEvent`
  - `OrderAcceptedEvent`
  - `OrderPartiallyFilledEvent`
  - `OrderFilledEvent`
  - `OrderCancelledEvent`
  - `OrderRejectedEvent`

- [ ] **1.2.4** Position events
  - `PositionOpenedEvent`
  - `PositionUpdatedEvent`
  - `PositionClosedEvent`

- [ ] **1.2.5** Risk events
  - `RiskLimitBreachedEvent`
  - `MarginCallEvent`

**Deliverables:**
- `trading/domain/events/market_data.py`
- `trading/domain/events/orders.py`
- `trading/domain/events/positions.py`
- `trading/domain/events/risk.py`
- Pydantic models for validation

---

### Task 1.3: Event Store
**Duration:** 4-6 hours

#### Subtasks:
- [ ] **1.3.1** Event store interface
  ```python
  class IEventStore(Protocol):
      async def append(self, event: DomainEvent) -> None
      async def get_events(self, from_id: Optional[str] = None) -> AsyncIterator[DomainEvent]
  ```

- [ ] **1.3.2** File-based event store
  - Append-only event log (JSONL format)
  - Event indexing by timestamp
  - Efficient read for replay

- [ ] **1.3.3** SQLite event store (optional, better)
  - Single table: `events(id, timestamp, type, data_json)`
  - Indexed by timestamp and type
  - Async support (aiosqlite)

**Deliverables:**
- `trading/infrastructure/persistence/event_store.py`
- File-based and SQLite implementations
- Migration scripts

---

### Task 1.4: Event Replay Mechanism
**Duration:** 3-4 hours

#### Subtasks:
- [ ] **1.4.1** Event replayer interface
  ```python
  class IEventReplayer(Protocol):
      async def replay(
          self,
          from_timestamp: datetime,
          to_timestamp: datetime,
          speed: float = 1.0  # 1.0 = real-time, 0 = as fast as possible
      ) -> AsyncIterator[DomainEvent]
  ```

- [ ] **1.4.2** Time-aware replay
  - Respect original event timing
  - Speed control (fast-forward, slow-mo)
  - Pause/resume capability

- [ ] **1.4.3** Filter and transform
  - Filter by event type
  - Filter by exchange/symbol
  - Event transformation hooks

**Deliverables:**
- `trading/infrastructure/replay/event_replayer.py`
- Integration with event store
- Example replay scripts

---

## 🏗️ Architecture

### Event Flow
```
┌─────────────┐
│  Exchange   │ ─── WebSocket ─┐
└─────────────┘                │
                               ▼
┌─────────────┐         ┌──────────────┐
│  Strategy   │ ◀────── │  Event Bus   │ ◀─── Event Store
└─────────────┘         └──────────────┘
       │                       │
       │                       ▼
       │                ┌──────────────┐
       └───────────────▶│     OMS      │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Portfolio   │
                        └──────────────┘
```

### Event Priorities
1. **Critical** (P0): Order fills, risk breaches
2. **High** (P1): Order updates, position changes
3. **Normal** (P2): Market data ticks
4. **Low** (P3): Metrics, logging

---

## 🧪 Testing Strategy

### Unit Tests
- Event creation and validation
- Event bus publish/subscribe
- Event store append/retrieve
- Event replay timing

### Integration Tests
- End-to-end event flow
- Multiple subscribers
- Event persistence and replay
- Performance under load

### Performance Tests
- 100K events/sec throughput
- <1ms event routing latency
- Memory usage under load

---

## 📦 Dependencies

```toml
# New dependencies for this module
aiosqlite = "^0.19.0"  # Async SQLite
orjson = "^3.9.0"      # Fast JSON serialization
```

---

## 🎯 Success Criteria

- [ ] Events can be published and consumed asynchronously
- [ ] Events are persisted reliably
- [ ] Events can be replayed with accurate timing
- [ ] 100K events/sec sustained throughput
- [ ] <1ms p99 event routing latency
- [ ] Zero data loss under normal operation
- [ ] Comprehensive test coverage (>90%)

---

## 📝 Implementation Order

1. **Day 1 Morning:** Event bus interface and in-memory implementation
2. **Day 1 Afternoon:** Base domain events (market data, orders)
3. **Day 2 Morning:** Event store (file + SQLite)
4. **Day 2 Afternoon:** Event replay + testing

---

## 🔗 Integration Points

**Upstream (Produces Events):**
- Market data connectors → TickEvent, OrderBookEvent
- OMS → OrderEvent variants
- Portfolio → PositionEvent variants

**Downstream (Consumes Events):**
- Strategies → Subscribe to market data
- Portfolio → Subscribe to order events
- Risk manager → Subscribe to all events
- Analytics → Subscribe for metrics

---

*Status: Ready for implementation*
*Estimated Total Time: 2 days*
