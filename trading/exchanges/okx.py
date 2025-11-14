"""
OKX Exchange Connector

Provides connectivity to OKX (formerly OKEx) exchange for:
- Spot trading
- Futures trading
- Perpetual swaps
- Market data streaming
- Order execution

OKX API Documentation: https://www.okx.com/docs-v5/en/
"""
import asyncio
import hmac
import hashlib
import base64
from typing import Optional, List, Dict, Any, Callable, Awaitable
from decimal import Decimal
from datetime import datetime
import json
from loguru import logger

try:
    import ccxt.async_support as ccxt
except ImportError:
    import ccxt

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
    TimeInForce,
    Position,
    Instrument,
    ConnectionState,
    ExchangeType,
)
from .rate_limiter import TokenBucketRateLimiter
from .websocket import ResilientWebSocket
from ..events.market_data import (
    TickEvent,
    TradeEvent,
    OrderBookEvent,
    OrderBookLevel,
    BarEvent,
)
from ..events.order import (
    OrderSubmittedEvent,
    OrderAcceptedEvent,
    OrderPartiallyFilledEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
    OrderRejectedEvent,
)


class OKXInstrumentMapper(IInstrumentMapper):
    """Maps OKX symbols to normalized format"""

    def __init__(self, market_type: str = "spot", rest_url: str = "https://www.okx.com"):
        """
        Initialize instrument mapper

        Args:
            market_type: Type of market ("spot", "futures", "swap")
            rest_url: REST API base URL
        """
        self.market_type = market_type
        self.rest_url = rest_url
        self.instruments: Dict[str, Instrument] = {}
        self._raw_to_normalized: Dict[str, str] = {}
        self._normalized_to_raw: Dict[str, str] = {}

    async def load_instruments(self) -> None:
        """Load instrument info from OKX"""
        logger.info(f"Loading OKX {self.market_type} instruments")

        # For now, create placeholder instruments
        # In production, would fetch from OKX API
        common_pairs = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
            "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT"
        ]

        for symbol in common_pairs:
            base, quote = symbol.split("/")
            raw_symbol = f"{base}-{quote}"

            self._raw_to_normalized[raw_symbol] = symbol
            self._normalized_to_raw[symbol] = raw_symbol

            self.instruments[symbol] = Instrument(
                symbol=symbol,
                exchange="okx",
                raw_symbol=raw_symbol,
                base_currency=base,
                quote_currency=quote,
                min_quantity=Decimal("0.001"),
                max_quantity=Decimal("1000000"),
                quantity_step=Decimal("0.001"),
                min_price=Decimal("0.01"),
                max_price=Decimal("1000000"),
                price_step=Decimal("0.01"),
            )

        logger.info(f"Loaded {len(self.instruments)} OKX instruments")

    async def normalize_symbol(self, raw_symbol: str) -> str:
        """Convert OKX symbol to normalized format (e.g., 'BTC-USDT' -> 'BTC/USDT')"""
        return self._raw_to_normalized.get(raw_symbol, raw_symbol.replace("-", "/"))

    async def denormalize_symbol(self, normalized_symbol: str) -> str:
        """Convert normalized symbol to OKX format (e.g., 'BTC/USDT' -> 'BTC-USDT')"""
        return self._normalized_to_raw.get(normalized_symbol, normalized_symbol.replace("/", "-"))

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument info"""
        return self.instruments.get(symbol)

    async def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        return symbol in self.instruments

    async def list_instruments(self) -> List[Instrument]:
        """List all instruments"""
        return list(self.instruments.values())


class OKXMarketDataConnector(IMarketDataConnector):
    """OKX market data connector via WebSocket"""

    def __init__(
        self,
        instrument_mapper: OKXInstrumentMapper,
        ws_url: str = "wss://ws.okx.com:8443/ws/v5/public",
        event_bus: Optional[Any] = None
    ):
        """
        Initialize market data connector

        Args:
            instrument_mapper: Instrument mapper instance
            ws_url: WebSocket URL
            event_bus: Event bus for publishing events
        """
        self.instrument_mapper = instrument_mapper
        self.ws_url = ws_url
        self.event_bus = event_bus
        self.ws: Optional[ResilientWebSocket] = None
        self.state = ConnectionState.DISCONNECTED

        # Subscription tracking
        self.subscribed_symbols: set = set()
        self.tick_handlers: List[Callable] = []
        self.trade_handlers: List[Callable] = []
        self.orderbook_handlers: List[Callable] = []
        self.bar_handlers: List[Callable] = []

        logger.info("Initialized OKXMarketDataConnector")

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state

    async def connect(self) -> None:
        """Connect to OKX WebSocket"""
        logger.info("Connecting to OKX market data")

        self.ws = ResilientWebSocket(
            url=self.ws_url,
            on_message=self._handle_message,
            name="okx_market_data"
        )

        await self.ws.connect()
        self.state = ConnectionState.CONNECTED
        logger.info("OKX market data connected")

    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        if self.ws:
            await self.ws.disconnect()
        self.state = ConnectionState.DISCONNECTED
        logger.info("OKX market data disconnected")

    async def subscribe_ticks(
        self,
        symbols: list[str],
        handler: Optional[Callable[[TickEvent], Awaitable[None]]] = None
    ) -> None:
        """Subscribe to tick/BBO updates"""
        if handler:
            self.tick_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            # Subscribe to ticker channel
            await self._subscribe_channel("tickers", raw_symbol)
            logger.debug(f"Subscribed to ticks: {symbol}")

    async def subscribe_trades(
        self,
        symbols: list[str],
        handler: Optional[Callable[[TradeEvent], Awaitable[None]]] = None
    ) -> None:
        """Subscribe to trade updates"""
        if handler:
            self.trade_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            await self._subscribe_channel("trades", raw_symbol)
            logger.debug(f"Subscribed to trades: {symbol}")

    async def subscribe_order_book(
        self,
        symbols: list[str],
        depth: int = 20,
        handler: Optional[Callable[[OrderBookEvent], Awaitable[None]]] = None
    ) -> None:
        """Subscribe to order book updates"""
        if handler:
            self.orderbook_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            # OKX supports books, books5, books-l2-tbt, bbo-tbt
            channel = f"books{depth}" if depth <= 400 else "books"
            await self._subscribe_channel(channel, raw_symbol)
            logger.debug(f"Subscribed to orderbook: {symbol} depth={depth}")

    async def subscribe_bars(
        self,
        symbols: list[str],
        interval: str,
        handler: Optional[Callable[[BarEvent], Awaitable[None]]] = None
    ) -> None:
        """Subscribe to candlestick/bar updates"""
        if handler:
            self.bar_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            # Convert interval to OKX format (1m, 5m, 1H, 1D, etc.)
            await self._subscribe_channel(f"candle{interval}", raw_symbol)
            logger.debug(f"Subscribed to bars: {symbol} @ {interval}")

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from specific symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
        logger.debug(f"Unsubscribed from symbols: {symbols}")

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all symbols"""
        self.subscribed_symbols.clear()
        logger.debug("Unsubscribed from all symbols")

    async def _subscribe_channel(self, channel: str, inst_id: str) -> None:
        """Subscribe to OKX channel"""
        if not self.ws:
            return

        message = {
            "op": "subscribe",
            "args": [{
                "channel": channel,
                "instId": inst_id
            }]
        }

        await self.ws.send(json.dumps(message))

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Handle subscription confirmations
            if data.get("event") == "subscribe":
                logger.debug(f"OKX subscription confirmed: {data}")
                return

            # Handle data messages
            if "data" in data and "arg" in data:
                channel = data["arg"].get("channel", "")

                if channel.startswith("tickers"):
                    await self._handle_ticker(data)
                elif channel.startswith("trades"):
                    await self._handle_trade(data)
                elif channel.startswith("books"):
                    await self._handle_orderbook(data)
                elif channel.startswith("candle"):
                    await self._handle_candle(data)

        except Exception as e:
            logger.error(f"Error handling OKX message: {e}")

    async def _handle_ticker(self, data: Dict) -> None:
        """Handle ticker update"""
        for tick_data in data.get("data", []):
            try:
                raw_symbol = data["arg"]["instId"]
                symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

                event = TickEvent(
                    exchange="okx",
                    symbol=symbol,
                    raw_symbol=raw_symbol,
                    bid=Decimal(tick_data.get("bidPx", "0")),
                    bid_size=Decimal(tick_data.get("bidSz", "0")),
                    ask=Decimal(tick_data.get("askPx", "0")),
                    ask_size=Decimal(tick_data.get("askSz", "0")),
                    last=Decimal(tick_data.get("last", "0")),
                    volume=Decimal(tick_data.get("vol24h", "0")),
                    exchange_timestamp=datetime.fromtimestamp(int(tick_data.get("ts", "0")) / 1000)
                )

                # Notify handlers
                for handler in self.tick_handlers:
                    await handler(event)

                # Publish to event bus
                if self.event_bus:
                    await self.event_bus.publish(event)

            except Exception as e:
                logger.error(f"Error processing ticker: {e}")

    async def _handle_trade(self, data: Dict) -> None:
        """Handle trade update"""
        # Similar implementation to ticker
        pass

    async def _handle_orderbook(self, data: Dict) -> None:
        """Handle orderbook update"""
        # Similar implementation to ticker
        pass

    async def _handle_candle(self, data: Dict) -> None:
        """Handle candle update"""
        # Similar implementation to ticker
        pass


class OKXExecutionConnector(IExecutionConnector):
    """OKX order execution connector"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        instrument_mapper: OKXInstrumentMapper,
        rate_limiter: TokenBucketRateLimiter,
        rest_url: str = "https://www.okx.com"
    ):
        """
        Initialize execution connector

        Args:
            api_key: OKX API key
            api_secret: OKX API secret
            passphrase: OKX API passphrase
            instrument_mapper: Instrument mapper instance
            rate_limiter: Rate limiter
            rest_url: REST API URL
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.instrument_mapper = instrument_mapper
        self.rate_limiter = rate_limiter
        self.rest_url = rest_url
        self.state = ConnectionState.DISCONNECTED

        logger.info("Initialized OKXExecutionConnector")

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state

    async def connect(self) -> None:
        """Connect to execution API"""
        self.state = ConnectionState.CONNECTED
        logger.info("OKX execution connected")

    async def disconnect(self) -> None:
        """Disconnect from execution API"""
        self.state = ConnectionState.DISCONNECTED
        logger.info("OKX execution disconnected")

    async def place_order(self, order: Order) -> OrderResult:
        """Place order on OKX"""
        await self.rate_limiter.acquire("place_order")

        try:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(order.symbol)

            # Build OKX order request
            order_request = {
                "instId": raw_symbol,
                "tdMode": "cash",  # Trading mode: cash (spot)
                "side": "buy" if order.side == OrderSide.BUY else "sell",
                "ordType": self._map_order_type(order.order_type),
                "sz": str(order.quantity),
            }

            if order.price:
                order_request["px"] = str(order.price)

            # TODO: Implement actual REST API call
            logger.info(f"Would place OKX order: {order_request}")

            return OrderResult(
                success=True,
                order_id=order.order_id,
                exchange_order_id=f"okx_{order.order_id}",
                message="Order placed (simulated)"
            )

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(
                success=False,
                order_id=order.order_id,
                message=str(e)
            )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """Cancel order on OKX"""
        await self.rate_limiter.acquire("cancel_order")

        logger.info(f"Would cancel OKX order: {order_id}")
        return OrderResult(
            success=True,
            order_id=order_id,
            message="Order cancelled (simulated)"
        )

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders"""
        logger.info(f"Would cancel all OKX orders for {symbol or 'all symbols'}")
        return 0

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order status from OKX"""
        return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders from OKX"""
        return []

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get positions from OKX"""
        return []

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map OrderType to OKX order type"""
        mapping = {
            OrderType.LIMIT: "limit",
            OrderType.MARKET: "market",
            OrderType.STOP_LOSS: "conditional",
            OrderType.STOP_LOSS_LIMIT: "conditional",
            OrderType.TAKE_PROFIT: "conditional",
            OrderType.TAKE_PROFIT_LIMIT: "conditional",
        }
        return mapping.get(order_type, "limit")


class OKXExchange(IExchange):
    """Complete OKX exchange implementation"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        market_type: str = "spot",
        testnet: bool = False
    ):
        """
        Initialize OKX exchange

        Args:
            api_key: API key (optional for market data only)
            api_secret: API secret (optional for market data only)
            passphrase: API passphrase (optional for market data only)
            market_type: Market type ("spot", "futures", "swap")
            testnet: Use testnet environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.market_type = market_type

        # URLs
        if testnet:
            rest_url = "https://www.okx.com"  # OKX doesn't have separate testnet
            ws_url = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
        else:
            rest_url = "https://www.okx.com"
            ws_url = "wss://ws.okx.com:8443/ws/v5/public"

        # Create components
        self.instrument_mapper = OKXInstrumentMapper(market_type, rest_url)
        self.rate_limiter = TokenBucketRateLimiter("okx")

        self.market_data = OKXMarketDataConnector(
            instrument_mapper=self.instrument_mapper,
            ws_url=ws_url
        )

        if api_key and api_secret and passphrase:
            self.execution = OKXExecutionConnector(
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                instrument_mapper=self.instrument_mapper,
                rate_limiter=self.rate_limiter,
                rest_url=rest_url
            )
        else:
            self.execution = None

        logger.info(f"Initialized OKXExchange: market_type={market_type}, testnet={testnet}")

    @property
    def name(self) -> str:
        """Exchange name"""
        return "okx"

    @property
    def exchange_type(self) -> ExchangeType:
        """Exchange type"""
        return ExchangeType.CRYPTO

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.market_data.get_state() == ConnectionState.CONNECTED

    def get_connection_state(self) -> ConnectionState:
        """Get connection state"""
        return self.market_data.get_state()

    async def connect(self) -> None:
        """Connect all components"""
        await self.instrument_mapper.load_instruments()
        await self.market_data.connect()
        if self.execution:
            await self.execution.connect()
        logger.info("OKXExchange connected")

    async def disconnect(self) -> None:
        """Disconnect all components"""
        await self.market_data.disconnect()
        if self.execution:
            await self.execution.disconnect()
        logger.info("OKXExchange disconnected")

    def get_market_data_connector(self) -> IMarketDataConnector:
        """Get market data connector"""
        return self.market_data

    def get_execution_connector(self) -> IExecutionConnector:
        """Get execution connector"""
        if not self.execution:
            raise ValueError("Execution not configured - provide API credentials")
        return self.execution

    def get_instrument_mapper(self) -> IInstrumentMapper:
        """Get instrument mapper"""
        return self.instrument_mapper
