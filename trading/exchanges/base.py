"""
Base interfaces for exchange connectors

Following SOLID principles:
- Single Responsibility: Each interface has one clear purpose
- Open/Closed: Easy to extend with new exchanges
- Liskov Substitution: All exchanges interchangeable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions

All exchanges implement these interfaces regardless of type (CEX/DEX/traditional).
"""
from abc import ABC, abstractmethod
from typing import Optional, Any
from enum import Enum

from trading.events.bus import IEventBus


class ExchangeType(str, Enum):
    """Exchange type classification"""
    CEX = "centralized"  # Binance, OKEx, Coinbase, etc.
    DEX = "decentralized"  # Uniswap, Hyperliquid, etc.
    TRADITIONAL = "traditional"  # CME, NSE, BSE, etc.


class ConnectionState(str, Enum):
    """Connection state"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class IExchange(ABC):
    """
    Base interface for all exchanges

    An exchange provides:
    - Market data connectivity
    - Order execution
    - Instrument mapping/normalization
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name (e.g., 'binance', 'okex')"""
        pass

    @property
    @abstractmethod
    def exchange_type(self) -> ExchangeType:
        """Exchange type"""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to exchange

        This should:
        - Initialize WebSocket connections
        - Authenticate if needed
        - Subscribe to necessary channels
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from exchange

        Clean shutdown of all connections.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected"""
        pass

    @abstractmethod
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state"""
        pass

    @abstractmethod
    def get_market_data_connector(self) -> "IMarketDataConnector":
        """Get market data connector for this exchange"""
        pass

    @abstractmethod
    def get_execution_connector(self) -> "IExecutionConnector":
        """Get execution connector for this exchange"""
        pass

    @abstractmethod
    def get_instrument_mapper(self) -> "IInstrumentMapper":
        """Get instrument mapper for this exchange"""
        pass


class IMarketDataConnector(ABC):
    """
    Interface for market data connectivity

    Responsibilities:
    - Subscribe to market data feeds
    - Parse incoming data
    - Publish events to event bus
    """

    @abstractmethod
    async def subscribe_ticks(self, symbols: list[str]) -> None:
        """
        Subscribe to tick data (best bid/ask/last)

        Args:
            symbols: List of normalized symbols
        """
        pass

    @abstractmethod
    async def subscribe_trades(self, symbols: list[str]) -> None:
        """
        Subscribe to trade feed (actual trades executed)

        Args:
            symbols: List of normalized symbols
        """
        pass

    @abstractmethod
    async def subscribe_order_book(
        self,
        symbols: list[str],
        depth: int = 20
    ) -> None:
        """
        Subscribe to order book updates

        Args:
            symbols: List of normalized symbols
            depth: Number of price levels
        """
        pass

    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: list[str],
        interval: str = "1m"
    ) -> None:
        """
        Subscribe to OHLCV bars

        Args:
            symbols: List of normalized symbols
            interval: Bar interval (1m, 5m, 1h, 1d, etc.)
        """
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """
        Unsubscribe from symbols

        Args:
            symbols: List of normalized symbols to unsubscribe from
        """
        pass

    @abstractmethod
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all symbols"""
        pass


class IExecutionConnector(ABC):
    """
    Interface for order execution

    Responsibilities:
    - Place orders
    - Cancel orders
    - Query order status
    - Publish order events
    """

    @abstractmethod
    async def place_order(self, order: "Order") -> "OrderResult":
        """
        Place an order

        Args:
            order: Order to place

        Returns:
            OrderResult with exchange order ID or error
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
        symbol: str
    ) -> "CancelResult":
        """
        Cancel an order

        Args:
            order_id: Exchange order ID or client order ID
            symbol: Normalized symbol

        Returns:
            CancelResult with status
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders

        Args:
            symbol: If provided, cancel only for this symbol

        Returns:
            Number of orders cancelled
        """
        pass

    @abstractmethod
    async def get_order_status(
        self,
        order_id: str,
        symbol: str
    ) -> "OrderStatus":
        """
        Get order status

        Args:
            order_id: Exchange order ID or client order ID
            symbol: Normalized symbol

        Returns:
            Current order status
        """
        pass

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> list["OrderStatus"]:
        """
        Get all open orders

        Args:
            symbol: If provided, filter by symbol

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> list["Position"]:
        """
        Get current positions (for futures/margin trading)

        Args:
            symbol: If provided, filter by symbol

        Returns:
            List of positions
        """
        pass


class IInstrumentMapper(ABC):
    """
    Interface for instrument mapping and normalization

    Responsibilities:
    - Convert between exchange and normalized symbols
    - Provide instrument metadata
    - Handle different instrument types
    """

    @abstractmethod
    async def load_instruments(self) -> None:
        """
        Load instrument information from exchange

        This should be called once during initialization.
        """
        pass

    @abstractmethod
    def normalize_symbol(self, raw_symbol: str) -> str:
        """
        Convert exchange symbol to normalized format

        Args:
            raw_symbol: Exchange-specific symbol (e.g., 'BTCUSDT')

        Returns:
            Normalized symbol (e.g., 'BTC/USDT')
        """
        pass

    @abstractmethod
    def denormalize_symbol(self, symbol: str) -> str:
        """
        Convert normalized symbol to exchange format

        Args:
            symbol: Normalized symbol (e.g., 'BTC/USDT')

        Returns:
            Exchange-specific symbol (e.g., 'BTCUSDT')
        """
        pass

    @abstractmethod
    def get_instrument(self, symbol: str) -> Optional["Instrument"]:
        """
        Get instrument metadata

        Args:
            symbol: Normalized symbol

        Returns:
            Instrument info or None if not found
        """
        pass

    @abstractmethod
    def list_instruments(
        self,
        instrument_type: Optional["InstrumentType"] = None,
        quote_currency: Optional[str] = None
    ) -> list["Instrument"]:
        """
        List all available instruments

        Args:
            instrument_type: Filter by instrument type
            quote_currency: Filter by quote currency (e.g., 'USDT')

        Returns:
            List of instruments
        """
        pass

    @abstractmethod
    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is valid on this exchange

        Args:
            symbol: Normalized symbol

        Returns:
            True if valid
        """
        pass


class IRateLimiter(ABC):
    """
    Interface for rate limiting

    Prevents exceeding exchange rate limits.
    """

    @abstractmethod
    async def acquire(self, endpoint: str, weight: int = 1) -> None:
        """
        Acquire permission to make request

        Blocks if rate limit would be exceeded.

        Args:
            endpoint: API endpoint identifier
            weight: Request weight (some endpoints cost more)
        """
        pass

    @abstractmethod
    def get_remaining(self, endpoint: str) -> int:
        """
        Get remaining requests for endpoint

        Args:
            endpoint: API endpoint identifier

        Returns:
            Number of remaining requests
        """
        pass

    @abstractmethod
    def reset_limits(self) -> None:
        """Reset all rate limits (for testing)"""
        pass


# Import these from trading.events.order once implemented
# For now, define minimal versions to avoid circular imports
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class Order(BaseModel):
    """Order to be placed"""
    client_order_id: str
    symbol: str  # Normalized
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    strategy_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderResult(BaseModel):
    """Result of order placement"""
    success: bool
    order_id: Optional[str] = None  # Exchange order ID
    client_order_id: str
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CancelResult(BaseModel):
    """Result of order cancellation"""
    success: bool
    order_id: str
    error: Optional[str] = None


class OrderStatus(BaseModel):
    """Current order status"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: str  # 'open', 'filled', 'cancelled', etc.
    quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    price: Optional[Decimal] = None
    average_fill_price: Optional[Decimal] = None
    timestamp: datetime


class Position(BaseModel):
    """Current position"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    leverage: Optional[Decimal] = None


class InstrumentType(str, Enum):
    """Instrument type"""
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTION = "option"


class Instrument(BaseModel):
    """Instrument metadata"""
    symbol: str  # Normalized
    raw_symbol: str  # Exchange-specific
    base_currency: str
    quote_currency: str
    instrument_type: InstrumentType
    tick_size: Decimal  # Minimum price increment
    min_order_size: Decimal
    max_order_size: Optional[Decimal] = None
    price_precision: int
    quantity_precision: int
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
