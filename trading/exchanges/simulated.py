"""
Simulated exchange for backtesting and paper trading

Provides realistic order execution simulation with:
- Market data replay from recorded events
- Order matching based on tick/orderbook data
- Configurable slippage models
- Latency simulation
- Position tracking

Operating Modes:
1. Replay: Replay historical market data with accurate timing
2. Live Paper: Connect to live data but simulate execution
"""
import asyncio
from typing import Optional, Callable, Awaitable, Dict, List
from decimal import Decimal
from datetime import datetime
from collections import defaultdict
import uuid
from loguru import logger

from .base import (
    IExchange,
    IMarketDataConnector,
    IExecutionConnector,
    IInstrumentMapper,
    Order,
    OrderResult,
    OrderStatus,
    OrderSide,
    OrderType,
    Position,
    Instrument,
    ConnectionState,
)
from .matching import OrderMatcher, FillModel
from .slippage import ISlippageModel, create_slippage_model
from ..events.base import BaseEvent
from ..events.market_data import TickEvent, TradeEvent, OrderBookEvent, BarEvent
from ..events.order import (
    OrderSubmittedEvent,
    OrderAcceptedEvent,
    OrderPartiallyFilledEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
    OrderRejectedEvent,
)
from ..events.persistence import EventReader


class SimulatedInstrumentMapper(IInstrumentMapper):
    """Simple instrument mapper for simulation"""

    def __init__(self):
        self.instruments: Dict[str, Instrument] = {}

    async def normalize_symbol(self, raw_symbol: str) -> str:
        """Already normalized in simulation"""
        return raw_symbol

    async def denormalize_symbol(self, normalized_symbol: str) -> str:
        """Already denormalized in simulation"""
        return normalized_symbol

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument info"""
        if symbol not in self.instruments:
            # Create default instrument
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                exchange="simulated",
                raw_symbol=symbol,
                base_currency=symbol.split("/")[0] if "/" in symbol else "BTC",
                quote_currency=symbol.split("/")[1] if "/" in symbol else "USDT",
                min_quantity=Decimal("0.001"),
                max_quantity=Decimal("1000000"),
                quantity_step=Decimal("0.001"),
                min_price=Decimal("0.01"),
                max_price=Decimal("1000000"),
                price_step=Decimal("0.01"),
            )
        return self.instruments[symbol]


class SimulatedMarketDataConnector(IMarketDataConnector):
    """
    Market data connector for simulation

    Can either:
    1. Replay historical data from EventReader
    2. Connect to live data source but only for paper trading
    """

    def __init__(
        self,
        event_reader: Optional[EventReader] = None,
        replay_speed: float = 1.0
    ):
        """
        Initialize simulated market data

        Args:
            event_reader: Reader for historical events (for replay mode)
            replay_speed: Speed multiplier for replay (1.0 = real-time)
        """
        self.event_reader = event_reader
        self.replay_speed = replay_speed
        self.state = ConnectionState.DISCONNECTED

        # Market data caches
        self.latest_ticks: Dict[str, TickEvent] = {}
        self.latest_orderbooks: Dict[str, OrderBookEvent] = {}
        self.latest_bars: Dict[str, BarEvent] = {}

        # Subscriptions
        self.tick_handlers: List[Callable[[TickEvent], Awaitable[None]]] = []
        self.trade_handlers: List[Callable[[TradeEvent], Awaitable[None]]] = []
        self.orderbook_handlers: List[Callable[[OrderBookEvent], Awaitable[None]]] = []
        self.bar_handlers: List[Callable[[BarEvent], Awaitable[None]]] = []

        self._replay_task: Optional[asyncio.Task] = None

        logger.info("Initialized SimulatedMarketDataConnector")

    async def connect(self) -> None:
        """Connect (start replay if configured)"""
        self.state = ConnectionState.CONNECTED
        logger.info("Simulated market data connected")

    async def disconnect(self) -> None:
        """Disconnect"""
        if self._replay_task:
            self._replay_task.cancel()
            try:
                await self._replay_task
            except asyncio.CancelledError:
                pass

        self.state = ConnectionState.DISCONNECTED
        logger.info("Simulated market data disconnected")

    async def subscribe_ticks(
        self,
        symbols: list[str],
        handler: Callable[[TickEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to tick updates"""
        self.tick_handlers.append(handler)
        logger.debug(f"Subscribed to ticks: {symbols}")

    async def subscribe_trades(
        self,
        symbols: list[str],
        handler: Callable[[TradeEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to trade updates"""
        self.trade_handlers.append(handler)
        logger.debug(f"Subscribed to trades: {symbols}")

    async def subscribe_order_book(
        self,
        symbols: list[str],
        handler: Callable[[OrderBookEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to order book updates"""
        self.orderbook_handlers.append(handler)
        logger.debug(f"Subscribed to orderbook: {symbols}")

    async def subscribe_bars(
        self,
        symbols: list[str],
        interval: str,
        handler: Callable[[BarEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to bar/candle updates"""
        self.bar_handlers.append(handler)
        logger.debug(f"Subscribed to bars: {symbols} @ {interval}")

    async def replay_events(
        self,
        start: datetime,
        end: datetime,
        symbols: Optional[List[str]] = None
    ) -> None:
        """
        Replay historical events

        Args:
            start: Start timestamp
            end: End timestamp
            symbols: Filter to specific symbols (None = all)
        """
        if not self.event_reader:
            logger.warning("No event reader configured for replay")
            return

        logger.info(f"Starting event replay: {start} to {end}")

        async for event in self.event_reader.read_range(start, end):
            # Filter by symbol if specified
            if symbols and hasattr(event, 'symbol') and event.symbol not in symbols:
                continue

            # Update caches and call handlers
            await self._handle_event(event)

            # Respect replay speed
            if self.replay_speed < float('inf'):
                await asyncio.sleep(0)  # Yield control

        logger.info("Event replay completed")

    async def _handle_event(self, event: BaseEvent) -> None:
        """Handle replayed event"""
        if isinstance(event, TickEvent):
            self.latest_ticks[event.symbol] = event
            for handler in self.tick_handlers:
                await handler(event)

        elif isinstance(event, TradeEvent):
            for handler in self.trade_handlers:
                await handler(event)

        elif isinstance(event, OrderBookEvent):
            self.latest_orderbooks[event.symbol] = event
            for handler in self.orderbook_handlers:
                await handler(event)

        elif isinstance(event, BarEvent):
            self.latest_bars[event.symbol] = event
            for handler in self.bar_handlers:
                await handler(event)

    def get_latest_tick(self, symbol: str) -> Optional[TickEvent]:
        """Get latest tick for symbol"""
        return self.latest_ticks.get(symbol)

    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookEvent]:
        """Get latest order book for symbol"""
        return self.latest_orderbooks.get(symbol)

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state


class SimulatedExecutionConnector(IExecutionConnector):
    """
    Execution connector for simulation

    Simulates order execution with:
    - Realistic fills based on market data
    - Configurable slippage
    - Latency simulation
    - Position tracking
    """

    def __init__(
        self,
        market_data: SimulatedMarketDataConnector,
        matcher: OrderMatcher,
        slippage_model: ISlippageModel,
        initial_balance: Decimal = Decimal("100000")
    ):
        """
        Initialize simulated execution

        Args:
            market_data: Market data connector
            matcher: Order matching engine
            slippage_model: Slippage model
            initial_balance: Initial account balance
        """
        self.market_data = market_data
        self.matcher = matcher
        self.slippage_model = slippage_model
        self.state = ConnectionState.DISCONNECTED

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

        # Account
        self.balance = initial_balance
        self.initial_balance = initial_balance

        # Event handlers
        self.order_update_handlers: List[Callable[[Order], Awaitable[None]]] = []

        logger.info(f"Initialized SimulatedExecutionConnector: balance={initial_balance}")

    async def connect(self) -> None:
        """Connect"""
        self.state = ConnectionState.CONNECTED
        logger.info("Simulated execution connected")

    async def disconnect(self) -> None:
        """Disconnect"""
        self.state = ConnectionState.DISCONNECTED
        logger.info("Simulated execution disconnected")

    async def place_order(self, order: Order) -> OrderResult:
        """
        Place order (simulated)

        Args:
            order: Order to place

        Returns:
            Order result
        """
        # Validate order
        if not order.order_id:
            order.order_id = f"sim_{uuid.uuid4().hex[:16]}"

        # Store order
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = datetime.utcnow()
        self.orders[order.order_id] = order

        logger.info(
            f"Order submitted: {order.order_id} {order.side.value} "
            f"{order.quantity} {order.symbol} @ {order.limit_price or 'MARKET'}"
        )

        # Notify handlers
        await self._notify_order_update(order)

        # Accept order immediately (simulation)
        order.status = OrderStatus.ACCEPTED
        await self._notify_order_update(order)

        # Try to fill order immediately if market data available
        asyncio.create_task(self._try_fill_order(order))

        return OrderResult(
            success=True,
            order_id=order.order_id,
            exchange_order_id=order.order_id,
            message="Order accepted (simulated)"
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel order"""
        if order_id not in self.orders:
            return OrderResult(
                success=False,
                order_id=order_id,
                message="Order not found"
            )

        order = self.orders[order_id]

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return OrderResult(
                success=False,
                order_id=order_id,
                message=f"Cannot cancel order in state: {order.status}"
            )

        order.status = OrderStatus.CANCELLED
        await self._notify_order_update(order)

        logger.info(f"Order cancelled: {order_id}")

        return OrderResult(
            success=True,
            order_id=order_id,
            message="Order cancelled"
        )

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        orders = [
            order for order in self.orders.values()
            if order.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)
        ]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)

    async def _try_fill_order(self, order: Order) -> None:
        """Try to fill order based on current market data"""
        # Get latest market data
        tick = self.market_data.get_latest_tick(order.symbol)
        orderbook = self.market_data.get_latest_orderbook(order.symbol)

        if not tick:
            # No market data yet - wait for tick
            logger.debug(f"No market data for {order.symbol}, order waiting")
            return

        # Check if order can fill
        should_fill, fill_price, fill_qty = self.matcher.check_fill(
            order, tick, orderbook
        )

        if not should_fill or not fill_price or not fill_qty:
            # Order doesn't fill yet
            return

        # Apply slippage
        slipped_price = self.slippage_model.calculate_slippage(
            order, fill_price, tick, orderbook
        )

        # Execute fill
        await self._execute_fill(order, slipped_price, fill_qty)

    async def _execute_fill(
        self,
        order: Order,
        fill_price: Decimal,
        fill_qty: Decimal
    ) -> None:
        """Execute order fill"""

        order.filled_quantity += fill_qty
        fill_value = fill_price * fill_qty

        # Update position
        await self._update_position(order, fill_qty, fill_price)

        # Update balance (for closed positions)
        if order.side == OrderSide.SELL:
            self.balance += fill_value
        else:
            self.balance -= fill_value

        # Check if fully filled
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_time = datetime.utcnow()
            order.average_filled_price = fill_price

            logger.info(
                f"Order filled: {order.order_id} {order.quantity} @ {fill_price}"
            )
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            logger.info(
                f"Order partially filled: {order.order_id} "
                f"{order.filled_quantity}/{order.quantity} @ {fill_price}"
            )

        await self._notify_order_update(order)

    async def _update_position(
        self,
        order: Order,
        fill_qty: Decimal,
        fill_price: Decimal
    ) -> None:
        """Update position from fill"""

        if order.symbol not in self.positions:
            # Open new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                exchange="simulated",
                quantity=Decimal("0"),
                average_entry_price=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                last_update=datetime.utcnow()
            )

        position = self.positions[order.symbol]

        # Calculate new position
        if order.side == OrderSide.BUY:
            # Increase position
            total_cost = (position.quantity * position.average_entry_price) + (fill_qty * fill_price)
            position.quantity += fill_qty
            if position.quantity > 0:
                position.average_entry_price = total_cost / position.quantity
        else:
            # Decrease position
            if position.quantity > 0:
                # Realize P&L
                pnl = (fill_price - position.average_entry_price) * fill_qty
                position.realized_pnl += pnl

            position.quantity -= fill_qty

        position.last_update = datetime.utcnow()

        # Remove position if zero
        if position.quantity == 0:
            del self.positions[order.symbol]

        logger.debug(f"Position updated: {order.symbol} qty={position.quantity}")

    async def _notify_order_update(self, order: Order) -> None:
        """Notify order update handlers"""
        for handler in self.order_update_handlers:
            await handler(order)

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state

    def get_account_balance(self) -> Decimal:
        """Get account balance"""
        return self.balance

    def get_total_equity(self) -> Decimal:
        """Get total equity (balance + unrealized P&L)"""
        total_unrealized = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        return self.balance + total_unrealized


class SimulatedExchange(IExchange):
    """
    Complete simulated exchange for backtesting and paper trading

    Provides realistic simulation with:
    - Historical data replay
    - Realistic order matching
    - Configurable slippage
    - Position tracking
    """

    def __init__(
        self,
        event_reader: Optional[EventReader] = None,
        fill_model: FillModel = "realistic",
        slippage_model: str = "spread",
        initial_balance: Decimal = Decimal("100000"),
        **kwargs
    ):
        """
        Initialize simulated exchange

        Args:
            event_reader: Event reader for replay mode
            fill_model: Order matching model
            slippage_model: Slippage model type
            initial_balance: Initial account balance
            **kwargs: Additional parameters for slippage model
        """
        self.exchange_id = "simulated"

        # Create components
        self.matcher = OrderMatcher(fill_model=fill_model)
        self.slippage = create_slippage_model(slippage_model, **kwargs)

        self.market_data = SimulatedMarketDataConnector(event_reader=event_reader)
        self.execution = SimulatedExecutionConnector(
            market_data=self.market_data,
            matcher=self.matcher,
            slippage_model=self.slippage,
            initial_balance=initial_balance
        )
        self.instrument_mapper = SimulatedInstrumentMapper()

        logger.info(
            f"Initialized SimulatedExchange: "
            f"fill={fill_model}, slippage={slippage_model}"
        )

    async def connect(self) -> None:
        """Connect all components"""
        await self.market_data.connect()
        await self.execution.connect()
        logger.info("SimulatedExchange connected")

    async def disconnect(self) -> None:
        """Disconnect all components"""
        await self.market_data.disconnect()
        await self.execution.disconnect()
        logger.info("SimulatedExchange disconnected")

    def get_market_data_connector(self) -> IMarketDataConnector:
        """Get market data connector"""
        return self.market_data

    def get_execution_connector(self) -> IExecutionConnector:
        """Get execution connector"""
        return self.execution

    def get_instrument_mapper(self) -> IInstrumentMapper:
        """Get instrument mapper"""
        return self.instrument_mapper

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.market_data.get_state()

    async def run_backtest(
        self,
        start: datetime,
        end: datetime,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Run backtest simulation

        Args:
            start: Start timestamp
            end: End timestamp
            symbols: Symbols to backtest

        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest: {start} to {end}")

        # Connect
        await self.connect()

        # Replay events
        await self.market_data.replay_events(start, end, symbols)

        # Collect results
        results = {
            "start": start,
            "end": end,
            "initial_balance": self.execution.initial_balance,
            "final_balance": self.execution.balance,
            "final_equity": self.execution.get_total_equity(),
            "total_orders": len(self.execution.orders),
            "filled_orders": len([
                o for o in self.execution.orders.values()
                if o.status == OrderStatus.FILLED
            ]),
            "open_positions": len(self.execution.positions),
            "realized_pnl": sum(
                pos.realized_pnl for pos in self.execution.positions.values()
            ),
        }

        logger.info(f"Backtest complete: P&L={results['realized_pnl']}")

        return results
