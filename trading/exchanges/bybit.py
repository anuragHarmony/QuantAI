"""
Bybit Exchange Connector (V5 API)

Provides connectivity to Bybit exchange for:
- Spot trading
- USDT/USDC perpetual contracts
- Inverse perpetual contracts
- Options trading
- Market data streaming
- Order execution

Bybit V5 API Documentation: https://bybit-exchange.github.io/docs/v5/intro
"""
import asyncio
import hmac
import hashlib
import base64
import time
from typing import Optional, List, Dict, Any, Callable, Awaitable
from decimal import Decimal
from datetime import datetime
import json
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


class BybitInstrumentMapper(IInstrumentMapper):
    """Maps Bybit symbols to normalized format"""

    def __init__(self, market_type: str = "spot", rest_url: str = "https://api.bybit.com"):
        """
        Initialize instrument mapper

        Args:
            market_type: Type of market ("spot", "linear", "inverse", "option")
            rest_url: REST API base URL
        """
        self.market_type = market_type
        self.rest_url = rest_url
        self.instruments: Dict[str, Instrument] = {}
        self._raw_to_normalized: Dict[str, str] = {}
        self._normalized_to_raw: Dict[str, str] = {}

    async def load_instruments(self) -> None:
        """Load instrument info from Bybit"""
        logger.info(f"Loading Bybit {self.market_type} instruments")

        # For now, create placeholder instruments
        # In production, would fetch from Bybit /v5/market/instruments-info API
        common_pairs = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
            "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT",
            "MATIC/USDT", "DOT/USDT", "LINK/USDT", "UNI/USDT"
        ]

        for symbol in common_pairs:
            base, quote = symbol.split("/")

            # Bybit format: BTCUSDT for spot, BTCUSDT for linear perpetual
            if self.market_type == "spot":
                raw_symbol = f"{base}{quote}"
            elif self.market_type == "linear":
                raw_symbol = f"{base}{quote}"  # Linear perpetuals use same format
            elif self.market_type == "inverse":
                raw_symbol = f"{base}{quote}"
            else:
                raw_symbol = f"{base}{quote}"

            self._raw_to_normalized[raw_symbol] = symbol
            self._normalized_to_raw[symbol] = raw_symbol

            self.instruments[symbol] = Instrument(
                symbol=symbol,
                exchange="bybit",
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

        logger.info(f"Loaded {len(self.instruments)} Bybit instruments")

    async def normalize_symbol(self, raw_symbol: str) -> str:
        """Convert Bybit symbol to normalized format (e.g., 'BTCUSDT' -> 'BTC/USDT')"""
        if raw_symbol in self._raw_to_normalized:
            return self._raw_to_normalized[raw_symbol]

        # Try to parse if not in cache
        # Bybit uses formats like BTCUSDT, ETHUSDT
        for quote in ['USDT', 'USDC', 'USD', 'BTC', 'ETH']:
            if raw_symbol.endswith(quote):
                base = raw_symbol[:-len(quote)]
                normalized = f"{base}/{quote}"
                self._raw_to_normalized[raw_symbol] = normalized
                self._normalized_to_raw[normalized] = raw_symbol
                return normalized

        return raw_symbol

    async def denormalize_symbol(self, normalized_symbol: str) -> str:
        """Convert normalized symbol to Bybit format (e.g., 'BTC/USDT' -> 'BTCUSDT')"""
        return self._normalized_to_raw.get(normalized_symbol, normalized_symbol.replace("/", ""))

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument info"""
        return self.instruments.get(symbol)

    async def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        return symbol in self.instruments

    async def list_instruments(self) -> List[Instrument]:
        """List all instruments"""
        return list(self.instruments.values())


class BybitMarketDataConnector(IMarketDataConnector):
    """Bybit market data connector via WebSocket V5"""

    def __init__(
        self,
        instrument_mapper: BybitInstrumentMapper,
        ws_url: str,
        event_bus: Optional[Any] = None
    ):
        """
        Initialize market data connector

        Args:
            instrument_mapper: Instrument mapper instance
            ws_url: WebSocket URL (differs by market type)
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

        logger.info(f"Initialized BybitMarketDataConnector: {ws_url}")

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state

    async def connect(self) -> None:
        """Connect to Bybit WebSocket"""
        logger.info("Connecting to Bybit market data")

        self.ws = ResilientWebSocket(
            url=self.ws_url,
            on_message=self._handle_message,
            name="bybit_market_data"
        )

        await self.ws.connect()
        self.state = ConnectionState.CONNECTED
        logger.info("Bybit market data connected")

    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        if self.ws:
            await self.ws.disconnect()
        self.state = ConnectionState.DISCONNECTED
        logger.info("Bybit market data disconnected")

    async def subscribe_ticks(
        self,
        symbols: list[str],
        handler: Optional[Callable[[TickEvent], Awaitable[None]]] = None
    ) -> None:
        """Subscribe to ticker/BBO updates"""
        if handler:
            self.tick_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            # Subscribe to tickers channel
            await self._subscribe_channel(f"tickers.{raw_symbol}")
            logger.debug(f"Subscribed to tickers: {symbol}")

    async def subscribe_trades(
        self,
        symbols: list[str],
        handler: Optional[Callable[[TradeEvent], Awaitable[None]]] = None
    ) -> None:
        """Subscribe to public trade updates"""
        if handler:
            self.trade_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            await self._subscribe_channel(f"publicTrade.{raw_symbol}")
            logger.debug(f"Subscribed to trades: {symbol}")

    async def subscribe_order_book(
        self,
        symbols: list[str],
        depth: int = 50,
        handler: Optional[Callable[[OrderBookEvent], Awaitable[None]]] = None
    ) -> None:
        """
        Subscribe to order book updates

        Bybit supports depths: 1, 50, 200, 500 for spot
        For linear/inverse: 1, 50, 200, 500
        """
        if handler:
            self.orderbook_handlers.append(handler)

        # Map depth to Bybit's supported depths
        if depth <= 1:
            depth = 1
        elif depth <= 50:
            depth = 50
        elif depth <= 200:
            depth = 200
        else:
            depth = 500

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            # Subscribe to orderbook.{depth}.{symbol}
            await self._subscribe_channel(f"orderbook.{depth}.{raw_symbol}")
            logger.debug(f"Subscribed to orderbook: {symbol} depth={depth}")

    async def subscribe_bars(
        self,
        symbols: list[str],
        interval: str,
        handler: Optional[Callable[[BarEvent], Awaitable[None]]] = None
    ) -> None:
        """
        Subscribe to kline/candlestick updates

        Bybit intervals: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        """
        if handler:
            self.bar_handlers.append(handler)

        for symbol in symbols:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(symbol)
            self.subscribed_symbols.add(symbol)

            # Subscribe to kline.{interval}.{symbol}
            await self._subscribe_channel(f"kline.{interval}.{raw_symbol}")
            logger.debug(f"Subscribed to kline: {symbol} @ {interval}")

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from specific symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
        logger.debug(f"Unsubscribed from symbols: {symbols}")

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all symbols"""
        self.subscribed_symbols.clear()
        logger.debug("Unsubscribed from all symbols")

    async def _subscribe_channel(self, topic: str) -> None:
        """Subscribe to Bybit V5 channel"""
        if not self.ws:
            return

        message = {
            "op": "subscribe",
            "args": [topic]
        }

        await self.ws.send(json.dumps(message))

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Handle subscription confirmations
            if data.get("op") == "subscribe":
                if data.get("success"):
                    logger.debug(f"Bybit subscription confirmed: {data.get('ret_msg', '')}")
                else:
                    logger.error(f"Bybit subscription failed: {data}")
                return

            # Handle pings
            if data.get("op") == "ping":
                await self.ws.send(json.dumps({"op": "pong"}))
                return

            # Handle data messages
            if "topic" in data and "data" in data:
                topic = data["topic"]

                if topic.startswith("tickers."):
                    await self._handle_ticker(data)
                elif topic.startswith("publicTrade."):
                    await self._handle_trade(data)
                elif topic.startswith("orderbook."):
                    await self._handle_orderbook(data)
                elif topic.startswith("kline."):
                    await self._handle_kline(data)

        except Exception as e:
            logger.error(f"Error handling Bybit message: {e}")

    async def _handle_ticker(self, data: Dict) -> None:
        """Handle ticker update"""
        try:
            tick_data = data.get("data", {})
            topic = data.get("topic", "")

            # Extract symbol from topic: "tickers.BTCUSDT"
            raw_symbol = topic.split(".")[-1]
            symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

            event = TickEvent(
                exchange="bybit",
                symbol=symbol,
                raw_symbol=raw_symbol,
                bid=Decimal(tick_data.get("bid1Price", "0")),
                bid_size=Decimal(tick_data.get("bid1Size", "0")),
                ask=Decimal(tick_data.get("ask1Price", "0")),
                ask_size=Decimal(tick_data.get("ask1Size", "0")),
                last=Decimal(tick_data.get("lastPrice", "0")),
                volume=Decimal(tick_data.get("volume24h", "0")),
                exchange_timestamp=datetime.fromtimestamp(int(data.get("ts", "0")) / 1000)
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
        # Bybit sends array of trades
        for trade_data in data.get("data", []):
            try:
                topic = data.get("topic", "")
                raw_symbol = topic.split(".")[-1]
                symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

                event = TradeEvent(
                    exchange="bybit",
                    symbol=symbol,
                    raw_symbol=raw_symbol,
                    trade_id=trade_data.get("i", ""),
                    price=Decimal(trade_data.get("p", "0")),
                    quantity=Decimal(trade_data.get("v", "0")),
                    side=OrderSide.BUY if trade_data.get("S") == "Buy" else OrderSide.SELL,
                    exchange_timestamp=datetime.fromtimestamp(int(trade_data.get("T", "0")) / 1000)
                )

                for handler in self.trade_handlers:
                    await handler(event)

                if self.event_bus:
                    await self.event_bus.publish(event)

            except Exception as e:
                logger.error(f"Error processing trade: {e}")

    async def _handle_orderbook(self, data: Dict) -> None:
        """Handle orderbook update"""
        try:
            book_data = data.get("data", {})
            topic = data.get("topic", "")
            raw_symbol = topic.split(".")[-1]
            symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

            # Parse bids and asks
            bids = [
                OrderBookLevel(price=Decimal(b[0]), quantity=Decimal(b[1]))
                for b in book_data.get("b", [])
            ]
            asks = [
                OrderBookLevel(price=Decimal(a[0]), quantity=Decimal(a[1]))
                for a in book_data.get("a", [])
            ]

            event = OrderBookEvent(
                exchange="bybit",
                symbol=symbol,
                raw_symbol=raw_symbol,
                bids=bids,
                asks=asks,
                exchange_timestamp=datetime.fromtimestamp(int(data.get("ts", "0")) / 1000),
                is_snapshot=data.get("type") == "snapshot"
            )

            for handler in self.orderbook_handlers:
                await handler(event)

            if self.event_bus:
                await self.event_bus.publish(event)

        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")

    async def _handle_kline(self, data: Dict) -> None:
        """Handle kline/candlestick update"""
        for kline_data in data.get("data", []):
            try:
                topic = data.get("topic", "")
                parts = topic.split(".")
                raw_symbol = parts[-1]
                interval = parts[1]
                symbol = await self.instrument_mapper.normalize_symbol(raw_symbol)

                event = BarEvent(
                    exchange="bybit",
                    symbol=symbol,
                    raw_symbol=raw_symbol,
                    interval=interval,
                    open_time=datetime.fromtimestamp(int(kline_data.get("start", "0")) / 1000),
                    close_time=datetime.fromtimestamp(int(kline_data.get("end", "0")) / 1000),
                    open=Decimal(kline_data.get("open", "0")),
                    high=Decimal(kline_data.get("high", "0")),
                    low=Decimal(kline_data.get("low", "0")),
                    close=Decimal(kline_data.get("close", "0")),
                    volume=Decimal(kline_data.get("volume", "0")),
                    is_closed=kline_data.get("confirm", False),
                    exchange_timestamp=datetime.fromtimestamp(int(data.get("ts", "0")) / 1000)
                )

                for handler in self.bar_handlers:
                    await handler(event)

                if self.event_bus:
                    await self.event_bus.publish(event)

            except Exception as e:
                logger.error(f"Error processing kline: {e}")


class BybitExecutionConnector(IExecutionConnector):
    """Bybit order execution connector using V5 API"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        instrument_mapper: BybitInstrumentMapper,
        rate_limiter: TokenBucketRateLimiter,
        rest_url: str = "https://api.bybit.com",
        market_type: str = "spot"
    ):
        """
        Initialize execution connector

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            instrument_mapper: Instrument mapper instance
            rate_limiter: Rate limiter
            rest_url: REST API URL
            market_type: Market type (spot, linear, inverse, option)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.instrument_mapper = instrument_mapper
        self.rate_limiter = rate_limiter
        self.rest_url = rest_url
        self.market_type = market_type
        self.state = ConnectionState.DISCONNECTED

        logger.info(f"Initialized BybitExecutionConnector: {market_type}")

    def get_state(self) -> ConnectionState:
        """Get connection state"""
        return self.state

    async def connect(self) -> None:
        """Connect to execution API"""
        self.state = ConnectionState.CONNECTED
        logger.info("Bybit execution connected")

    async def disconnect(self) -> None:
        """Disconnect from execution API"""
        self.state = ConnectionState.DISCONNECTED
        logger.info("Bybit execution disconnected")

    def _generate_signature(self, timestamp: int, params: str) -> str:
        """Generate HMAC SHA256 signature for Bybit V5 API"""
        param_str = f"{timestamp}{self.api_key}{params}"
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def place_order(self, order: Order) -> OrderResult:
        """Place order on Bybit"""
        await self.rate_limiter.acquire("place_order")

        try:
            raw_symbol = await self.instrument_mapper.denormalize_symbol(order.symbol)

            # Map category based on market type
            category_map = {
                "spot": "spot",
                "linear": "linear",
                "inverse": "inverse",
                "option": "option"
            }
            category = category_map.get(self.market_type, "spot")

            # Build Bybit V5 order request
            order_request = {
                "category": category,
                "symbol": raw_symbol,
                "side": "Buy" if order.side == OrderSide.BUY else "Sell",
                "orderType": self._map_order_type(order.order_type),
                "qty": str(order.quantity),
            }

            if order.price:
                order_request["price"] = str(order.price)

            if order.time_in_force:
                order_request["timeInForce"] = self._map_time_in_force(order.time_in_force)

            # TODO: Implement actual REST API call to /v5/order/create
            logger.info(f"Would place Bybit order: {order_request}")

            return OrderResult(
                success=True,
                order_id=order.order_id,
                exchange_order_id=f"bybit_{order.order_id}",
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
        """Cancel order on Bybit"""
        await self.rate_limiter.acquire("cancel_order")

        logger.info(f"Would cancel Bybit order: {order_id}")
        return OrderResult(
            success=True,
            order_id=order_id,
            message="Order cancelled (simulated)"
        )

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders"""
        logger.info(f"Would cancel all Bybit orders for {symbol or 'all symbols'}")
        return 0

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order status from Bybit"""
        return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders from Bybit"""
        return []

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get positions from Bybit"""
        return []

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map OrderType to Bybit order type"""
        mapping = {
            OrderType.LIMIT: "Limit",
            OrderType.MARKET: "Market",
        }
        return mapping.get(order_type, "Limit")

    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map TimeInForce to Bybit format"""
        mapping = {
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
        }
        return mapping.get(tif, "GTC")


class BybitExchange(IExchange):
    """Complete Bybit exchange implementation using V5 API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        market_type: str = "spot",
        testnet: bool = False
    ):
        """
        Initialize Bybit exchange

        Args:
            api_key: API key (optional for market data only)
            api_secret: API secret (optional for market data only)
            market_type: Market type ("spot", "linear", "inverse", "option")
            testnet: Use testnet environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.market_type = market_type

        # URLs based on environment and market type
        if testnet:
            rest_url = "https://api-testnet.bybit.com"
            ws_url_map = {
                "spot": "wss://stream-testnet.bybit.com/v5/public/spot",
                "linear": "wss://stream-testnet.bybit.com/v5/public/linear",
                "inverse": "wss://stream-testnet.bybit.com/v5/public/inverse",
                "option": "wss://stream-testnet.bybit.com/v5/public/option",
            }
        else:
            rest_url = "https://api.bybit.com"
            ws_url_map = {
                "spot": "wss://stream.bybit.com/v5/public/spot",
                "linear": "wss://stream.bybit.com/v5/public/linear",
                "inverse": "wss://stream.bybit.com/v5/public/inverse",
                "option": "wss://stream.bybit.com/v5/public/option",
            }

        ws_url = ws_url_map.get(market_type, ws_url_map["spot"])

        # Create components
        self.instrument_mapper = BybitInstrumentMapper(market_type, rest_url)
        self.rate_limiter = TokenBucketRateLimiter("bybit")

        self.market_data = BybitMarketDataConnector(
            instrument_mapper=self.instrument_mapper,
            ws_url=ws_url
        )

        if api_key and api_secret:
            self.execution = BybitExecutionConnector(
                api_key=api_key,
                api_secret=api_secret,
                instrument_mapper=self.instrument_mapper,
                rate_limiter=self.rate_limiter,
                rest_url=rest_url,
                market_type=market_type
            )
        else:
            self.execution = None

        logger.info(f"Initialized BybitExchange: market_type={market_type}, testnet={testnet}")

    @property
    def name(self) -> str:
        """Exchange name"""
        return "bybit"

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
        logger.info("BybitExchange connected")

    async def disconnect(self) -> None:
        """Disconnect all components"""
        await self.market_data.disconnect()
        if self.execution:
            await self.execution.disconnect()
        logger.info("BybitExchange disconnected")

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
