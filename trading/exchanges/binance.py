"""
Binance Exchange Connector

Connects to Binance spot and futures markets with:
- WebSocket market data streaming
- REST API order execution
- Symbol normalization
- Rate limiting
- Auto-reconnection

Supports:
- Spot markets
- USD-M Futures (USDT perpetuals)
- COIN-M Futures (coin perpetuals)

Endpoints:
- Spot: wss://stream.binance.com:9443/ws
- Futures: wss://fstream.binance.com/ws
"""
import asyncio
import hmac
import hashlib
import time
from typing import Optional, Callable, Awaitable, Dict, List
from decimal import Decimal
from datetime import datetime
import json
from urllib.parse import urlencode
from loguru import logger

try:
    import aiohttp
except ImportError:
    logger.warning("aiohttp not installed - Binance connector will not work")
    aiohttp = None

from .base import (
    IExchange,
    IMarketDataConnector,
    IExecutionConnector,
    IInstrumentMapper,
    ConnectionState,
    Order,
    OrderResult,
    OrderStatus,
    OrderSide,
    OrderType,
    TimeInForce,
    Position,
    Instrument,
)
from .websocket import ResilientWebSocket
from .rate_limiter import TokenBucketRateLimiter
from ..events.market_data import (
    TickEvent,
    TradeEvent,
    OrderBookEvent,
    OrderBookLevel,
    BarEvent,
)


class BinanceInstrumentMapper(IInstrumentMapper):
    """
    Maps between normalized and Binance symbols

    Normalized: BTC/USDT, ETH/USDT
    Binance: BTCUSDT, ETHUSDT
    """

    def __init__(
        self,
        market_type: str = "spot",
        rest_url: str = "https://api.binance.com"
    ):
        """
        Initialize instrument mapper

        Args:
            market_type: 'spot', 'future', or 'delivery'
            rest_url: Binance REST API URL
        """
        self.market_type = market_type
        self.rest_url = rest_url
        self.instruments: Dict[str, Instrument] = {}
        self._symbol_map: Dict[str, str] = {}  # normalized -> raw
        self._reverse_map: Dict[str, str] = {}  # raw -> normalized

        logger.info(f"Initialized BinanceInstrumentMapper: {market_type}")

    async def load_instruments(self) -> None:
        """Load instrument information from Binance API"""
        if not aiohttp:
            logger.error("aiohttp not installed")
            return

        endpoint = "/api/v3/exchangeInfo" if self.market_type == "spot" else "/fapi/v1/exchangeInfo"
        url = f"{self.rest_url}{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to load instruments: {response.status}")
                        return

                    data = await response.json()
                    symbols = data.get("symbols", [])

                    for symbol_info in symbols:
                        if symbol_info.get("status") != "TRADING":
                            continue

                        raw_symbol = symbol_info["symbol"]
                        base = symbol_info["baseAsset"]
                        quote = symbol_info["quoteAsset"]
                        normalized = f"{base}/{quote}"

                        # Parse filters
                        filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}

                        lot_size = filters.get("LOT_SIZE", {})
                        price_filter = filters.get("PRICE_FILTER", {})

                        instrument = Instrument(
                            symbol=normalized,
                            exchange="binance",
                            raw_symbol=raw_symbol,
                            base_currency=base,
                            quote_currency=quote,
                            min_quantity=Decimal(lot_size.get("minQty", "0.001")),
                            max_quantity=Decimal(lot_size.get("maxQty", "1000000")),
                            quantity_step=Decimal(lot_size.get("stepSize", "0.001")),
                            min_price=Decimal(price_filter.get("minPrice", "0.01")),
                            max_price=Decimal(price_filter.get("maxPrice", "1000000")),
                            price_step=Decimal(price_filter.get("tickSize", "0.01")),
                        )

                        self.instruments[normalized] = instrument
                        self._symbol_map[normalized] = raw_symbol
                        self._reverse_map[raw_symbol] = normalized

                    logger.info(f"Loaded {len(self.instruments)} Binance instruments")

        except Exception as e:
            logger.error(f"Error loading instruments: {e}")

    async def normalize_symbol(self, raw_symbol: str) -> str:
        """Convert Binance symbol to normalized format"""
        if raw_symbol in self._reverse_map:
            return self._reverse_map[raw_symbol]

        # Try to parse common patterns
        # BTCUSDT -> BTC/USDT
        for quote in ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"]:
            if raw_symbol.endswith(quote):
                base = raw_symbol[:-len(quote)]
                return f"{base}/{quote}"

        return raw_symbol

    async def denormalize_symbol(self, normalized_symbol: str) -> str:
        """Convert normalized symbol to Binance format"""
        if normalized_symbol in self._symbol_map:
            return self._symbol_map[normalized_symbol]

        # BTC/USDT -> BTCUSDT
        return normalized_symbol.replace("/", "")

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument information"""
        return self.instruments.get(symbol)


class BinanceMarketDataConnector(IMarketDataConnector):
    """
    Binance market data connector via WebSocket

    Streams:
    - <symbol>@ticker: 24hr ticker statistics
    - <symbol>@trade: Individual trades
    - <symbol>@depth: Order book updates
    - <symbol>@kline_<interval>: Candlestick data
    """

    def __init__(
        self,
        instrument_mapper: BinanceInstrumentMapper,
        ws_url: str = "wss://stream.binance.com:9443/ws"
    ):
        """
        Initialize market data connector

        Args:
            instrument_mapper: Instrument mapper for symbol conversion
            ws_url: WebSocket URL
        """
        self.instrument_mapper = instrument_mapper
        self.ws_url = ws_url
        self.ws: Optional[ResilientWebSocket] = None
        self.state = ConnectionState.DISCONNECTED

        # Subscriptions
        self.subscribed_streams: set[str] = set()

        # Handlers
        self.tick_handlers: List[Callable[[TickEvent], Awaitable[None]]] = []
        self.trade_handlers: List[Callable[[TradeEvent], Awaitable[None]]] = []
        self.orderbook_handlers: List[Callable[[OrderBookEvent], Awaitable[None]]] = []
        self.bar_handlers: List[Callable[[BarEvent], Awaitable[None]]] = []

        logger.info("Initialized BinanceMarketDataConnector")

    async def connect(self) -> None:
        """Connect to Binance WebSocket"""
        self.state = ConnectionState.CONNECTING

        self.ws = ResilientWebSocket(
            url=self.ws_url,
            on_message=self._handle_message,
            on_state_change=self._handle_state_change,
        )

        await self.ws.connect()

        logger.info("Connected to Binance WebSocket")

    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        if self.ws:
            await self.ws.close()
        self.state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Binance WebSocket")

    async def _handle_message(self, message: dict) -> None:
        """Handle WebSocket message"""
        try:
            # Binance stream messages have 'e' field for event type
            event_type = message.get("e")

            if event_type == "24hrTicker":
                await self._handle_ticker(message)
            elif event_type == "trade":
                await self._handle_trade(message)
            elif event_type == "depthUpdate":
                await self._handle_depth(message)
            elif event_type == "kline":
                await self._handle_kline(message)
            else:
                logger.debug(f"Unknown event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def _handle_ticker(self, data: dict) -> None:
        """Handle ticker update"""
        raw_symbol = data.get("s")
        normalized_symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

        tick = TickEvent(
            exchange="binance",
            symbol=normalized_symbol,
            raw_symbol=raw_symbol,
            bid=Decimal(data.get("b", "0")),
            ask=Decimal(data.get("a", "0")),
            last=Decimal(data.get("c", "0")),
            volume=Decimal(data.get("v", "0")),
            exchange_timestamp=datetime.fromtimestamp(data.get("E", 0) / 1000),
        )

        for handler in self.tick_handlers:
            await handler(tick)

    async def _handle_trade(self, data: dict) -> None:
        """Handle trade update"""
        raw_symbol = data.get("s")
        normalized_symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

        trade = TradeEvent(
            exchange="binance",
            symbol=normalized_symbol,
            raw_symbol=raw_symbol,
            trade_id=str(data.get("t")),
            price=Decimal(data.get("p")),
            quantity=Decimal(data.get("q")),
            side="buy" if data.get("m") else "sell",  # m = buyer is maker
            exchange_timestamp=datetime.fromtimestamp(data.get("T", 0) / 1000),
        )

        for handler in self.trade_handlers:
            await handler(trade)

    async def _handle_depth(self, data: dict) -> None:
        """Handle order book update"""
        raw_symbol = data.get("s")
        normalized_symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

        # Parse bids and asks
        bids = [
            OrderBookLevel(price=Decimal(p), quantity=Decimal(q))
            for p, q in data.get("b", [])
        ]
        asks = [
            OrderBookLevel(price=Decimal(p), quantity=Decimal(q))
            for p, q in data.get("a", [])
        ]

        orderbook = OrderBookEvent(
            exchange="binance",
            symbol=normalized_symbol,
            raw_symbol=raw_symbol,
            bids=bids,
            asks=asks,
            exchange_timestamp=datetime.utcnow(),
        )

        for handler in self.orderbook_handlers:
            await handler(orderbook)

    async def _handle_kline(self, data: dict) -> None:
        """Handle kline/candlestick update"""
        raw_symbol = data.get("s")
        normalized_symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

        kline = data.get("k", {})

        bar = BarEvent(
            exchange="binance",
            symbol=normalized_symbol,
            raw_symbol=raw_symbol,
            interval=kline.get("i"),
            open_time=datetime.fromtimestamp(kline.get("t", 0) / 1000),
            close_time=datetime.fromtimestamp(kline.get("T", 0) / 1000),
            open=Decimal(kline.get("o")),
            high=Decimal(kline.get("h")),
            low=Decimal(kline.get("l")),
            close=Decimal(kline.get("c")),
            volume=Decimal(kline.get("v")),
            is_closed=kline.get("x", False),
        )

        for handler in self.bar_handlers:
            await handler(bar)

    async def _handle_state_change(self, new_state: ConnectionState) -> None:
        """Handle WebSocket state change"""
        self.state = new_state

    async def subscribe_ticks(
        self,
        symbols: list[str],
        handler: Callable[[TickEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to ticker updates"""
        self.tick_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            stream = f"{raw_symbol.lower()}@ticker"

            if stream not in self.subscribed_streams:
                await self._subscribe_stream(stream)

        logger.info(f"Subscribed to ticks: {symbols}")

    async def subscribe_trades(
        self,
        symbols: list[str],
        handler: Callable[[TradeEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to trade updates"""
        self.trade_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            stream = f"{raw_symbol.lower()}@trade"

            if stream not in self.subscribed_streams:
                await self._subscribe_stream(stream)

        logger.info(f"Subscribed to trades: {symbols}")

    async def subscribe_order_book(
        self,
        symbols: list[str],
        handler: Callable[[OrderBookEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to order book updates"""
        self.orderbook_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            stream = f"{raw_symbol.lower()}@depth"

            if stream not in self.subscribed_streams:
                await self._subscribe_stream(stream)

        logger.info(f"Subscribed to orderbook: {symbols}")

    async def subscribe_bars(
        self,
        symbols: list[str],
        interval: str,
        handler: Callable[[BarEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to candlestick/bar updates"""
        self.bar_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            stream = f"{raw_symbol.lower()}@kline_{interval}"

            if stream not in self.subscribed_streams:
                await self._subscribe_stream(stream)

        logger.info(f"Subscribed to bars: {symbols} @ {interval}")

    async def _subscribe_stream(self, stream: str) -> None:
        """Subscribe to a Binance stream"""
        if not self.ws:
            logger.warning("WebSocket not connected")
            return

        message = {
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": int(time.time() * 1000)
        }

        await self.ws.send(message)
        self.subscribed_streams.add(stream)
        logger.debug(f"Subscribed to stream: {stream}")

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state


class BinanceExecutionConnector(IExecutionConnector):
    """
    Binance execution connector via REST API

    Endpoints:
    - POST /api/v3/order: Place order
    - DELETE /api/v3/order: Cancel order
    - GET /api/v3/order: Query order
    - GET /api/v3/openOrders: Query open orders
    - GET /api/v3/account: Query account
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        instrument_mapper: BinanceInstrumentMapper,
        rate_limiter: TokenBucketRateLimiter,
        rest_url: str = "https://api.binance.com"
    ):
        """
        Initialize execution connector

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            instrument_mapper: Instrument mapper
            rate_limiter: Rate limiter
            rest_url: REST API URL
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.instrument_mapper = instrument_mapper
        self.rate_limiter = rate_limiter
        self.rest_url = rest_url
        self.state = ConnectionState.DISCONNECTED

        # Order tracking
        self.orders: Dict[str, Order] = {}

        logger.info("Initialized BinanceExecutionConnector")

    async def connect(self) -> None:
        """Connect (validate API credentials)"""
        # Test API connection
        try:
            result = await self._request("GET", "/api/v3/account")
            if result:
                self.state = ConnectionState.CONNECTED
                logger.info("Binance execution connected")
            else:
                self.state = ConnectionState.ERROR
                logger.error("Failed to connect to Binance API")
        except Exception as e:
            self.state = ConnectionState.ERROR
            logger.error(f"Failed to connect: {e}")

    async def disconnect(self) -> None:
        """Disconnect"""
        self.state = ConnectionState.DISCONNECTED
        logger.info("Binance execution disconnected")

    async def place_order(self, order: Order) -> OrderResult:
        """Place order on Binance"""
        try:
            # Convert to Binance format
            raw_symbol = await self.instrument_mapper.denormalize_symbol(order.symbol)

            params = {
                "symbol": raw_symbol,
                "side": order.side.value.upper(),
                "type": self._convert_order_type(order.order_type),
                "quantity": str(order.quantity),
                "timestamp": int(time.time() * 1000)
            }

            if order.order_type == OrderType.LIMIT:
                params["price"] = str(order.limit_price)
                params["timeInForce"] = self._convert_time_in_force(order.time_in_force)

            # Place order
            response = await self._request("POST", "/api/v3/order", params)

            if response:
                exchange_order_id = str(response.get("orderId"))
                order.order_id = exchange_order_id
                order.status = OrderStatus.ACCEPTED
                self.orders[exchange_order_id] = order

                logger.info(f"Order placed: {exchange_order_id}")

                return OrderResult(
                    success=True,
                    order_id=exchange_order_id,
                    exchange_order_id=exchange_order_id,
                    message="Order placed successfully"
                )
            else:
                return OrderResult(
                    success=False,
                    order_id="",
                    message="Failed to place order"
                )

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(
                success=False,
                order_id="",
                message=str(e)
            )

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel order on Binance"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    message="Order not found"
                )

            raw_symbol = await self.instrument_mapper.denormalize_symbol(order.symbol)

            params = {
                "symbol": raw_symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000)
            }

            response = await self._request("DELETE", "/api/v3/order", params)

            if response:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    message="Order cancelled successfully"
                )
            else:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    message="Failed to cancel order"
                )

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=str(e)
            )

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Query order status"""
        return self.orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        orders = [
            o for o in self.orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)
        ]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    async def get_positions(self) -> List[Position]:
        """Get account positions"""
        # For spot, positions are just balances
        # For futures, query position endpoint
        # TODO: Implement position query
        return []

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        # TODO: Implement
        return None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None
    ) -> Optional[dict]:
        """Make authenticated request to Binance API"""
        if not aiohttp:
            logger.error("aiohttp not installed")
            return None

        # Apply rate limiting
        await self.rate_limiter.acquire(endpoint)

        url = f"{self.rest_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}

        if params is None:
            params = {}

        # Sign request
        if params:
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature

        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            text = await response.text()
                            logger.error(f"API error {response.status}: {text}")
                            return None

                elif method == "POST":
                    async with session.post(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            text = await response.text()
                            logger.error(f"API error {response.status}: {text}")
                            return None

                elif method == "DELETE":
                    async with session.delete(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            text = await response.text()
                            logger.error(f"API error {response.status}: {text}")
                            return None

        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert to Binance order type"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS",
            OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
        }
        return mapping.get(order_type, "MARKET")

    def _convert_time_in_force(self, tif: TimeInForce) -> str:
        """Convert to Binance time in force"""
        mapping = {
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
        }
        return mapping.get(tif, "GTC")

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state


class BinanceExchange(IExchange):
    """
    Complete Binance exchange connector

    Supports spot and futures markets with:
    - Real-time market data streaming
    - Order execution with rate limiting
    - Symbol normalization
    - Connection resilience
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        market_type: str = "spot",
        testnet: bool = False
    ):
        """
        Initialize Binance exchange

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            market_type: 'spot' or 'future'
            testnet: Use testnet endpoints
        """
        self.exchange_id = "binance"
        self.market_type = market_type

        # URLs
        if testnet:
            rest_url = "https://testnet.binance.vision"
            ws_url = "wss://testnet.binance.vision/ws"
        else:
            if market_type == "spot":
                rest_url = "https://api.binance.com"
                ws_url = "wss://stream.binance.com:9443/ws"
            else:  # futures
                rest_url = "https://fapi.binance.com"
                ws_url = "wss://fstream.binance.com/ws"

        # Create components
        self.instrument_mapper = BinanceInstrumentMapper(market_type, rest_url)
        self.rate_limiter = TokenBucketRateLimiter("binance")

        self.market_data = BinanceMarketDataConnector(
            instrument_mapper=self.instrument_mapper,
            ws_url=ws_url
        )

        self.execution = BinanceExecutionConnector(
            api_key=api_key,
            api_secret=api_secret,
            instrument_mapper=self.instrument_mapper,
            rate_limiter=self.rate_limiter,
            rest_url=rest_url
        )

        logger.info(f"Initialized BinanceExchange: {market_type}")

    async def connect(self) -> None:
        """Connect all components"""
        # Load instruments first
        await self.instrument_mapper.load_instruments()

        # Connect market data and execution
        await self.market_data.connect()
        await self.execution.connect()

        logger.info("BinanceExchange connected")

    async def disconnect(self) -> None:
        """Disconnect all components"""
        await self.market_data.disconnect()
        await self.execution.disconnect()

        logger.info("BinanceExchange disconnected")

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
