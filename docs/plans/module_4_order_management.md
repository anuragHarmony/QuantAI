# Module 4: Order Management System (OMS)

## Purpose

Build a professional Order Management System that handles the complete order lifecycle, pre-trade risk checks, order routing, state management, and fill handling across multiple exchanges.

## Core Responsibilities

1. **Order Lifecycle Management**: Track orders from creation to completion
2. **Pre-Trade Risk Checks**: Validate orders before submission
3. **Order Routing**: Route orders to appropriate exchanges
4. **State Tracking**: Maintain order state machine
5. **Fill Handling**: Process partial and full fills
6. **Order Events**: Publish all order state changes
7. **Rate Limit Coordination**: Prevent exchange rate limit violations

---

## Architecture

```
Strategy
    ↓ (places order)
OrderManager
    ↓ (pre-trade checks)
RiskChecker
    ↓ (validated order)
RateLimitGuard
    ↓ (rate limit OK)
OrderRouter
    ↓ (route to exchange)
ExecutionConnector
    ↓
Exchange
```

---

## Order State Machine

```
CREATED
    ↓
PENDING_SUBMISSION
    ↓
SUBMITTED (sent to exchange)
    ├→ ACCEPTED (exchange confirmed)
    │   ├→ PARTIALLY_FILLED
    │   │   ├→ FILLED (complete)
    │   │   └→ CANCELLED (partial cancel)
    │   ├→ FILLED
    │   └→ CANCELLED
    ├→ REJECTED (exchange rejected)
    └→ ERROR (system error)
```

---

## Core Interfaces

### IOrderManager
```python
class IOrderManager(ABC):
    @abstractmethod
    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order with full lifecycle management

        Steps:
        1. Validate order
        2. Run pre-trade checks
        3. Check rate limits
        4. Route to exchange
        5. Track state
        6. Publish events
        """

    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
        exchange: str,
        symbol: str
    ) -> CancelResult:
        """Cancel an order"""

    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[Decimal] = None,
        new_price: Optional[Decimal] = None
    ) -> OrderResult:
        """Modify an order (if supported by exchange)"""

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[OrderState]:
        """Get current order state"""

    @abstractmethod
    def get_open_orders(
        self,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> list[OrderState]:
        """Get all open orders with optional filters"""

    @abstractmethod
    async def sync_orders(self, exchange: str) -> None:
        """Sync order states with exchange"""
```

### IRiskChecker
```python
class RiskCheckResult(BaseModel):
    passed: bool
    reason: Optional[str] = None
    checks_run: list[str] = Field(default_factory=list)

class IRiskChecker(ABC):
    @abstractmethod
    async def check_order(
        self,
        order: Order,
        portfolio: Portfolio
    ) -> RiskCheckResult:
        """
        Run pre-trade risk checks

        Checks:
        - Sufficient capital
        - Position limits
        - Order count limits
        - Exposure limits
        - Price validity
        - Quantity validity
        """

    @abstractmethod
    def add_check(
        self,
        name: str,
        check_func: Callable[[Order, Portfolio], Awaitable[bool]]
    ) -> None:
        """Add custom risk check"""
```

### IOrderRouter
```python
class IOrderRouter(ABC):
    @abstractmethod
    async def route_order(self, order: Order) -> str:
        """
        Determine which exchange to route to

        For multi-exchange setups:
        - Smart order routing
        - Best execution
        - Liquidity aggregation
        """

    @abstractmethod
    def register_exchange(
        self,
        exchange_name: str,
        connector: IExecutionConnector
    ) -> None:
        """Register an execution connector"""
```

---

## Data Models

### OrderState
```python
class OrderStatus(str, Enum):
    CREATED = "created"
    PENDING_SUBMISSION = "pending_submission"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"
    EXPIRED = "expired"

class OrderState(BaseModel):
    """Complete order state tracked by OMS"""

    # Identifiers
    client_order_id: str
    exchange_order_id: Optional[str] = None
    exchange: str
    symbol: str

    # Order details
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC

    # State
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal
    average_fill_price: Optional[Decimal] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Metadata
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For order splitting
    fees: Decimal = Decimal("0")
    rejection_reason: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in {
            OrderStatus.CREATED,
            OrderStatus.PENDING_SUBMISSION,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.ERROR,
            OrderStatus.EXPIRED
        }

    @property
    def fill_percentage(self) -> Decimal:
        if self.quantity > 0:
            return (self.filled_quantity / self.quantity) * 100
        return Decimal("0")
```

### Fill
```python
class Fill(BaseModel):
    """Represents an order fill (partial or complete)"""

    fill_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime
    trade_id: Optional[str] = None  # Exchange trade ID
    is_maker: Optional[bool] = None
```

---

## Risk Checks

### Standard Pre-Trade Checks

1. **Capital Check**
```python
async def check_sufficient_capital(
    order: Order,
    portfolio: Portfolio
) -> bool:
    """Ensure sufficient buying power"""
    required = order.quantity * (order.price or current_price)
    return portfolio.buying_power >= required
```

2. **Position Limit Check**
```python
async def check_position_limit(
    order: Order,
    portfolio: Portfolio
) -> bool:
    """Ensure position limit not exceeded"""
    current_pos = portfolio.get_position(order.symbol)
    new_position = current_pos + order.quantity

    return abs(new_position) <= config.max_position_size
```

3. **Order Count Limit Check**
```python
async def check_order_count(
    order: Order,
    portfolio: Portfolio
) -> bool:
    """Ensure max open order limit not exceeded"""
    open_orders = order_manager.get_open_orders()
    return len(open_orders) < config.max_open_orders
```

4. **Price Validity Check**
```python
async def check_price_sanity(
    order: Order,
    portfolio: Portfolio
) -> bool:
    """Ensure price within reasonable range of market"""
    if order.order_type == OrderType.MARKET:
        return True

    current_price = await get_current_price(order.symbol)
    deviation = abs(order.price - current_price) / current_price

    return deviation <= config.max_price_deviation  # e.g., 5%
```

5. **Quantity Validity Check**
```python
async def check_quantity_valid(
    order: Order,
    portfolio: Portfolio
) -> bool:
    """Ensure quantity meets exchange requirements"""
    instrument = instrument_mapper.get_instrument(order.symbol)

    return (
        order.quantity >= instrument.min_order_size and
        (not instrument.max_order_size or
         order.quantity <= instrument.max_order_size)
    )
```

6. **Exposure Limit Check**
```python
async def check_exposure_limit(
    order: Order,
    portfolio: Portfolio
) -> bool:
    """Ensure total exposure within limits"""
    total_exposure = portfolio.calculate_exposure()
    order_exposure = order.quantity * (order.price or current_price)

    return (total_exposure + order_exposure) <= config.max_exposure
```

---

## Rate Limit Coordination

### IRateLimitGuard
```python
class IRateLimitGuard(ABC):
    @abstractmethod
    async def check_rate_limit(
        self,
        exchange: str,
        endpoint: str
    ) -> bool:
        """Check if we can make a request"""

    @abstractmethod
    async def acquire_rate_limit(
        self,
        exchange: str,
        endpoint: str
    ) -> None:
        """Wait for rate limit availability"""

    @abstractmethod
    def get_rate_limit_status(
        self,
        exchange: str
    ) -> dict[str, Any]:
        """Get current rate limit status"""
```

### Implementation
```python
class RateLimitGuard:
    """
    Coordinates rate limits across OMS

    Maintains per-exchange rate limit state
    Queues orders if rate limit hit
    Provides visibility into rate limit usage
    """

    def __init__(self):
        self.limiters: dict[str, IRateLimiter] = {}
        self.queues: dict[str, asyncio.Queue] = {}

    async def acquire_rate_limit(self, exchange: str, endpoint: str):
        limiter = self.limiters.get(exchange)
        if limiter:
            await limiter.acquire(endpoint)
```

---

## Order Router

### Smart Order Routing
```python
class SmartOrderRouter(IOrderRouter):
    """
    Routes orders to best exchange

    Strategies:
    - Single exchange: Route to configured exchange
    - Best price: Route to exchange with best quote
    - Liquidity: Route to exchange with best depth
    - Split: Split large orders across exchanges
    """

    async def route_order(self, order: Order) -> str:
        if self.routing_strategy == "configured":
            return order.metadata.get("exchange", self.default_exchange)

        elif self.routing_strategy == "best_price":
            return await self._find_best_price_exchange(order)

        elif self.routing_strategy == "liquidity":
            return await self._find_best_liquidity_exchange(order)
```

---

## Implementation Plan

### Step 1: Core Data Models (Day 1)
```
trading/oms/
├── __init__.py
├── models.py            # OrderState, Fill, etc.
└── enums.py             # OrderStatus, etc.
```

**Tasks:**
- [ ] Define OrderState model
- [ ] Define Fill model
- [ ] Define OrderResult, CancelResult
- [ ] Define all enums
- [ ] Add helper properties (is_open, is_terminal, etc.)

### Step 2: Risk Checker (Day 2-3)
```
trading/oms/
├── risk/
│   ├── __init__.py
│   ├── base.py          # IRiskChecker
│   ├── checks.py        # Standard check functions
│   └── checker.py       # RiskChecker implementation
```

**Tasks:**
- [ ] Define IRiskChecker interface
- [ ] Implement RiskChecker with plugin architecture
- [ ] Implement standard checks:
  - [ ] Capital check
  - [ ] Position limit check
  - [ ] Order count check
  - [ ] Price validity check
  - [ ] Quantity validity check
  - [ ] Exposure limit check
- [ ] Make checks configurable
- [ ] Add custom check support
- [ ] Test all checks

### Step 3: Rate Limit Guard (Day 3)
```
trading/oms/
└── rate_limit_guard.py
```

**Tasks:**
- [ ] Define IRateLimitGuard interface
- [ ] Implement RateLimitGuard
- [ ] Integrate with exchange rate limiters
- [ ] Order queueing logic
- [ ] Rate limit status monitoring
- [ ] Test rate limiting

### Step 4: Order Router (Day 4)
```
trading/oms/
├── routing/
│   ├── __init__.py
│   ├── base.py          # IOrderRouter
│   ├── simple.py        # SimpleRouter (single exchange)
│   └── smart.py         # SmartOrderRouter
```

**Tasks:**
- [ ] Define IOrderRouter interface
- [ ] Implement SimpleRouter (single exchange)
- [ ] Implement SmartOrderRouter
  - [ ] Best price routing
  - [ ] Best liquidity routing
  - [ ] Split order support
- [ ] Test routing logic

### Step 5: Order Manager (Day 5-7)
```
trading/oms/
├── order_manager.py     # Main OMS
└── state_tracker.py     # Order state tracking
```

**Tasks:**
- [ ] Define IOrderManager interface
- [ ] Implement OrderManager
  - [ ] submit_order with full flow
  - [ ] cancel_order
  - [ ] modify_order
  - [ ] get_order, get_open_orders
  - [ ] sync_orders
- [ ] Implement order state tracking
  - [ ] In-memory state store
  - [ ] State persistence (SQLite/PostgreSQL)
  - [ ] State updates on events
- [ ] Implement event publishing
  - [ ] OrderSubmittedEvent
  - [ ] OrderAcceptedEvent
  - [ ] OrderFilledEvent
  - [ ] OrderCancelledEvent
  - [ ] OrderRejectedEvent
- [ ] Fill handling logic
- [ ] Error handling and recovery

### Step 6: Order Reconciliation (Day 8)
```
trading/oms/
└── reconciliation.py
```

**Tasks:**
- [ ] Sync with exchange on startup
- [ ] Periodic reconciliation
- [ ] Handle orphaned orders
- [ ] Fill reconciliation
- [ ] State mismatch detection

### Step 7: Testing (Day 9-10)
```
tests/oms/
├── test_models.py
├── test_risk_checker.py
├── test_rate_limit_guard.py
├── test_router.py
├── test_order_manager.py
└── test_integration.py
```

**Tasks:**
- [ ] Unit tests for all components
- [ ] Test order lifecycle
- [ ] Test risk checks (pass and fail)
- [ ] Test rate limiting
- [ ] Test order routing
- [ ] Integration tests with mock exchange
- [ ] Test error scenarios
- [ ] Performance tests (1000+ orders/sec)

---

## Database Schema (for persistence)

```sql
CREATE TABLE orders (
    client_order_id VARCHAR(64) PRIMARY KEY,
    exchange_order_id VARCHAR(64),
    exchange VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(4) NOT NULL,
    order_type VARCHAR(16) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    time_in_force VARCHAR(8),
    status VARCHAR(32) NOT NULL,
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    created_at TIMESTAMP NOT NULL,
    submitted_at TIMESTAMP,
    last_updated_at TIMESTAMP NOT NULL,
    strategy_id VARCHAR(64),
    fees DECIMAL(20, 8) DEFAULT 0,
    rejection_reason TEXT,
    metadata JSONB
);

CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_exchange_symbol ON orders(exchange, symbol);
CREATE INDEX idx_orders_strategy ON orders(strategy_id);
CREATE INDEX idx_orders_created_at ON orders(created_at);

CREATE TABLE fills (
    fill_id VARCHAR(64) PRIMARY KEY,
    order_id VARCHAR(64) NOT NULL REFERENCES orders(client_order_id),
    exchange VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8) NOT NULL,
    fee_currency VARCHAR(8) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    trade_id VARCHAR(64),
    is_maker BOOLEAN
);

CREATE INDEX idx_fills_order_id ON fills(order_id);
CREATE INDEX idx_fills_timestamp ON fills(timestamp);
```

---

## Example Usage

### Submitting an Order
```python
from trading.oms import OrderManager
from trading.oms.models import Order

oms = OrderManager(
    event_bus=event_bus,
    risk_checker=risk_checker,
    rate_limit_guard=rate_limit_guard,
    order_router=order_router,
    portfolio=portfolio
)

# Create order
order = Order(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=Decimal("0.001"),
    price=Decimal("50000"),
    strategy_id="my_strategy"
)

# Submit (all checks happen automatically)
result = await oms.submit_order(order)

if result.success:
    print(f"Order submitted: {result.order_id}")
else:
    print(f"Order rejected: {result.error}")
```

### Listening to Order Events
```python
async def handle_order_filled(event: OrderFilledEvent):
    print(f"Order {event.order_id} filled!")
    print(f"Quantity: {event.filled_quantity}")
    print(f"Price: {event.average_fill_price}")

await event_bus.subscribe("order_filled", handle_order_filled)
```

### Custom Risk Check
```python
async def check_max_daily_trades(order: Order, portfolio: Portfolio) -> bool:
    """Custom check: max 100 trades per day"""
    today_trades = await db.count_trades(date=date.today())
    return today_trades < 100

risk_checker.add_check("max_daily_trades", check_max_daily_trades)
```

---

## Performance Targets

- **Order submission**: < 5ms (excluding network)
- **Risk checks**: < 2ms for all checks
- **State update**: < 1ms
- **Throughput**: 1000+ orders/second
- **Database write**: Async, non-blocking

---

## Success Criteria

- [ ] Complete order lifecycle management
- [ ] All standard risk checks implemented
- [ ] Rate limiting integrated
- [ ] Order routing works (simple + smart)
- [ ] State persistence to database
- [ ] Event publishing for all state changes
- [ ] Order reconciliation works
- [ ] Can handle 1000+ orders/sec
- [ ] Comprehensive test coverage (>90%)
- [ ] Documentation with examples

---

**Status**: Ready to implement (depends on Modules 1, 2)
**Priority**: Critical
**Estimated Time**: 10 days
**LOC**: ~2500-3000 lines
