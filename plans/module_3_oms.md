# Module 3: Order Management System (OMS)

## 🎯 Goal
Build a robust, institutional-grade Order Management System that handles the entire order lifecycle with pre-trade risk checks, state management, and event generation.

## 📋 Detailed Tasks

### Task 3.1: Order Model & State Machine
**Duration:** 4-6 hours

#### Subtasks:
- [ ] **3.1.1** Order domain model
  ```python
  @dataclass
  class Order:
      order_id: str
      client_order_id: str
      strategy_id: str
      exchange: str
      symbol: str
      side: OrderSide  # BUY, SELL
      order_type: OrderType  # MARKET, LIMIT, STOP, etc.
      quantity: Decimal
      price: Optional[Decimal]
      time_in_force: TimeInForce  # GTC, IOC, FOK
      state: OrderState
      filled_quantity: Decimal = Decimal("0")
      average_fill_price: Optional[Decimal] = None
      created_at: datetime
      updated_at: datetime
      metadata: Dict[str, Any] = field(default_factory=dict)
  ```

- [ ] **3.1.2** Order state machine
  ```
  PENDING → SUBMITTED → ACCEPTED → PARTIALLY_FILLED → FILLED
      │         │           │              │
      └─────────┴───────────┴──────────────┴─────────→ CANCELLED
                │           │              │
                └───────────┴──────────────┴─────────→ REJECTED
  ```

- [ ] **3.1.3** State transition validation
  - Valid transitions only
  - State change events
  - Audit trail

- [ ] **3.1.4** Order types support
  - Market orders
  - Limit orders
  - Stop orders
  - Stop-limit orders
  - Post-only orders

**Deliverables:**
- `trading/domain/models/order.py`
- `trading/domain/models/enums.py`
- State machine implementation
- Validation logic

---

### Task 3.2: Pre-Trade Risk Checks
**Duration:** 6-8 hours

#### Subtasks:
- [ ] **3.2.1** Risk check interface
  ```python
  class IRiskCheck(Protocol):
      async def check(self, order: Order, portfolio: Portfolio) -> RiskCheckResult
  ```

- [ ] **3.2.2** Risk check implementations
  - **Max order size**: Single order can't exceed limit
  - **Max position size**: Total position can't exceed limit
  - **Max open orders**: Limit number of concurrent orders
  - **Daily loss limit**: Stop trading if daily loss exceeds threshold
  - **Leverage check**: Total leverage can't exceed max
  - **Margin check**: Sufficient margin for order
  - **Symbol whitelist**: Only trade approved symbols
  - **Trading hours**: Only trade during allowed hours

- [ ] **3.2.3** Risk check chain
  - Execute checks in sequence
  - Stop at first failure
  - Return detailed reason for rejection
  - Log all checks

- [ ] **3.2.4** Configurable risk limits
  ```python
  @dataclass
  class RiskLimits:
      max_order_size: Decimal
      max_position_size: Decimal
      max_open_orders: int
      max_daily_loss: Decimal
      max_leverage: Decimal
      allowed_symbols: Set[str]
      trading_hours: List[Tuple[time, time]]
  ```

**Deliverables:**
- `trading/application/oms/risk/`
- Individual risk check implementations
- Risk check chain
- Configuration models

---

### Task 3.3: Order Router
**Duration:** 4-5 hours

#### Subtasks:
- [ ] **3.3.1** Router interface
  ```python
  class IOrderRouter(Protocol):
      async def route_order(self, order: Order) -> str  # Returns exchange_order_id
      async def cancel_order(self, order_id: str) -> None
      async def modify_order(self, order_id: str, **changes) -> None
  ```

- [ ] **3.3.2** Exchange router implementation
  - Map order to correct exchange connector
  - Handle exchange-specific requirements
  - Convert universal order to exchange format
  - Rate limiting per exchange

- [ ] **3.3.3** Routing strategies
  - **Direct routing**: Always to specified exchange
  - **Smart routing**: Best price across exchanges (future)
  - **Simulated routing**: For backtesting

- [ ] **3.3.4** Order tracking
  - Map client_order_id to exchange_order_id
  - Track order across its lifecycle
  - Handle order updates from exchange

**Deliverables:**
- `trading/application/oms/routing/order_router.py`
- Exchange router implementations
- Order ID mapping service

---

### Task 3.4: OMS Core Logic
**Duration:** 6-8 hours

#### Subtasks:
- [ ] **3.4.1** OMS interface
  ```python
  class IOrderManagementSystem(Protocol):
      async def submit_order(self, order_request: OrderRequest) -> Order
      async def cancel_order(self, order_id: str) -> None
      async def modify_order(self, order_id: str, **changes) -> None
      async def get_order(self, order_id: str) -> Order
      async def get_active_orders(self, strategy_id: Optional[str] = None) -> List[Order]
  ```

- [ ] **3.4.2** Order submission flow
  1. Validate order request
  2. Create Order object (PENDING state)
  3. Run pre-trade risk checks
  4. If checks pass → route to exchange (SUBMITTED state)
  5. Generate OrderSubmittedEvent
  6. Store order in order store
  7. Return order to caller

- [ ] **3.4.3** Order update handling
  - Listen to order events from execution layer
  - Update order state
  - Generate appropriate events
  - Notify subscribers

- [ ] **3.4.4** Order cancellation
  - Validate can be cancelled (state check)
  - Send cancel request to exchange
  - Handle cancel confirmations/rejections
  - Update order state

- [ ] **3.4.5** Order persistence
  - Store all orders (order store)
  - Retrieve by order_id, strategy_id
  - Query active/filled/cancelled orders
  - Audit trail

**Deliverables:**
- `trading/application/oms/oms.py`
- Order submission logic
- Order update handlers
- Order store interface and implementation

---

### Task 3.5: Event Generation
**Duration:** 3-4 hours

#### Subtasks:
- [ ] **3.5.1** Generate events for all order state changes
  - `OrderSubmittedEvent`
  - `OrderAcceptedEvent`
  - `OrderPartiallyFilledEvent`
  - `OrderFilledEvent`
  - `OrderCancelledEvent`
  - `OrderRejectedEvent`

- [ ] **3.5.2** Event enrichment
  - Include full order details
  - Add context (strategy_id, portfolio_id)
  - Timing information

- [ ] **3.5.3** Event publishing
  - Publish to event bus
  - Ensure guaranteed delivery (retry logic)
  - Handle backpressure

**Deliverables:**
- Event generation logic in OMS
- Integration with event bus

---

## 🏗️ Architecture

### OMS Flow
```
┌─────────────────────────────────────────────────────────────┐
│                        Strategy                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ submit_order(OrderRequest)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Order Management System (OMS)                   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Validate Order Request                          │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 2. Create Order (PENDING state)                    │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 3. Pre-Trade Risk Checks                           │    │
│  │    - Max order size ✓                              │    │
│  │    - Max position size ✓                           │    │
│  │    - Max open orders ✓                             │    │
│  │    - Daily loss limit ✓                            │    │
│  │    - Margin check ✓                                │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 4. Route to Exchange (SUBMITTED state)             │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 5. Generate OrderSubmittedEvent                    │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 6. Store Order                                     │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Execution Connector                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
                 Exchange API
```

### Risk Check Chain
```python
class RiskCheckChain:
    def __init__(self, checks: List[IRiskCheck]):
        self.checks = checks

    async def run_all_checks(self, order: Order, portfolio: Portfolio) -> RiskCheckResult:
        for check in self.checks:
            result = await check.check(order, portfolio)
            if not result.passed:
                return result  # Stop at first failure
        return RiskCheckResult(passed=True)
```

---

## 🧪 Testing Strategy

### Unit Tests
- Order state transitions
- Risk check logic (each check individually)
- Risk check chain
- Order validation

### Integration Tests
- End-to-end order submission
- Order updates from exchange
- Order cancellation
- Risk check integration with portfolio

### Scenario Tests
- Happy path: Order submitted and filled
- Risk rejection: Order exceeds limits
- Cancellation: Order cancelled successfully
- Partial fill: Order partially filled then cancelled

---

## 📦 Dependencies

```toml
# Already have these from Phase 1
pydantic = "^2.12.4"
```

---

## 🎯 Success Criteria

- [ ] All order types supported (market, limit, stop, etc.)
- [ ] State machine works correctly for all transitions
- [ ] Pre-trade risk checks prevent invalid orders
- [ ] Orders routed correctly to exchanges
- [ ] All order events generated and published
- [ ] Order persistence works reliably
- [ ] <10ms p99 latency for order submission (excluding exchange)
- [ ] Zero data loss for order history
- [ ] Comprehensive test coverage (>90%)

---

## 📝 Implementation Order

### Day 1: Order Model & State Machine
- Morning: Order model, enums, state machine
- Afternoon: State transition validation, tests

### Day 2: Risk Checks
- Morning: Risk check interface and individual checks
- Afternoon: Risk check chain, configuration

### Day 3: Order Router & OMS Core
- Morning: Order router and routing logic
- Afternoon: OMS core implementation

### Day 4: Event Generation & Testing
- Morning: Event generation and publishing
- Afternoon: Integration tests, scenario tests

---

## 🔗 Integration Points

**Consumes:**
- `Portfolio` (for risk checks)
- `IExecutionConnector` (for order routing)
- Configuration (risk limits)

**Produces:**
- Order events → Event Bus
- Orders → Order Store
- Execution requests → Execution Connectors

**Dependencies:**
- Event Bus (Module 1)
- Domain Events (Module 1)
- Portfolio Service (Module 4)
- Execution Connectors (Module 5)

---

## 🚀 Advanced Features (Future)

### Smart Order Routing
- Best execution across multiple exchanges
- Price improvement opportunities
- Liquidity aggregation

### Order Algorithms
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg orders
- Sniper orders

### Advanced Risk
- Correlation risk
- Concentration risk
- Stress testing
- Real-time VaR

---

## 📚 Industry Standards

### FIX Protocol (Future)
- Standard messaging protocol for trading
- Can integrate with institutional counterparties
- Python library: `quickfix`

### Best Practices
- Idempotency: Same order request = same result
- Atomicity: Order operations are atomic
- Auditability: Full audit trail of all actions
- Resilience: Handle exchange outages gracefully

---

*Status: Ready for implementation*
*Estimated Total Time: 3-4 days*
