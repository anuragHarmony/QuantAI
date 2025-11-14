"""
Exchange Connectors

Unified interface to multiple exchanges with:
- Multi-exchange support (crypto + traditional)
- Market data streaming (ticks, trades, orderbook, bars)
- Order execution with smart routing
- Rate limiting and connection resilience
- Simulation mode for backtesting

Exchanges:
- SimulatedExchange: For backtesting and paper trading
- BinanceExchange: Binance spot and futures
- (More to be added: OKEx, Bybit, Hyperliquid, CME, NSE/BSE)

Usage (Live):
    exchange = BinanceExchange(api_key=key, api_secret=secret)
    await exchange.connect()

    md = exchange.get_market_data_connector()
    await md.subscribe_ticks(["BTC/USDT"], tick_handler)

Usage (Simulation):
    reader = EventReader("./data/events")
    exchange = SimulatedExchange(event_reader=reader)

    results = await exchange.run_backtest(start, end, ["BTC/USDT"])
"""

# Base interfaces
from .base import (
    IExchange,
    IMarketDataConnector,
    IExecutionConnector,
    IInstrumentMapper,
    IRateLimiter,
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

# Rate limiting
from .rate_limiter import TokenBucket, TokenBucketRateLimiter

# WebSocket
from .websocket import ResilientWebSocket, WebSocketManager

# Simulation
from .matching import OrderMatcher, FillModel
from .slippage import (
    ISlippageModel,
    NoSlippage,
    FixedSlippage,
    VolumeSlippage,
    SpreadSlippage,
    HybridSlippage,
    create_slippage_model,
)
from .simulated import (
    SimulatedExchange,
    SimulatedMarketDataConnector,
    SimulatedExecutionConnector,
    SimulatedInstrumentMapper,
)

# Live Exchanges
from .binance import (
    BinanceExchange,
    BinanceMarketDataConnector,
    BinanceExecutionConnector,
    BinanceInstrumentMapper,
)

__all__ = [
    # Base
    "IExchange",
    "IMarketDataConnector",
    "IExecutionConnector",
    "IInstrumentMapper",
    "IRateLimiter",
    "ConnectionState",
    "Order",
    "OrderResult",
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "Position",
    "Instrument",
    # Rate Limiting
    "TokenBucket",
    "TokenBucketRateLimiter",
    # WebSocket
    "ResilientWebSocket",
    "WebSocketManager",
    # Simulation
    "OrderMatcher",
    "FillModel",
    "ISlippageModel",
    "NoSlippage",
    "FixedSlippage",
    "VolumeSlippage",
    "SpreadSlippage",
    "HybridSlippage",
    "create_slippage_model",
    "SimulatedExchange",
    "SimulatedMarketDataConnector",
    "SimulatedExecutionConnector",
    "SimulatedInstrumentMapper",
    # Live Exchanges
    "BinanceExchange",
    "BinanceMarketDataConnector",
    "BinanceExecutionConnector",
    "BinanceInstrumentMapper",
]
