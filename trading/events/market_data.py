"""
Market data events

Events representing market data from exchanges:
- Tick: Best bid/ask/last updates
- Trade: Actual trades executed on exchange
- OrderBook: Full order book snapshots and updates
- Bar: Aggregated OHLCV bars
- Quote: Best bid/ask (BBO)
"""
from datetime import datetime
from typing import Optional, Literal
from decimal import Decimal

from pydantic import BaseModel, Field

from .base import BaseEvent


class MarketDataEvent(BaseEvent):
    """Base class for all market data events"""
    exchange: str = Field(description="Exchange name")
    symbol: str = Field(description="Normalized symbol (e.g., 'BTC/USDT')")
    raw_symbol: str = Field(description="Exchange-specific symbol (e.g., 'BTCUSDT')")


class TickEvent(MarketDataEvent):
    """
    Tick event - best bid/ask/last price update

    Represents a single price update from the exchange.
    """
    event_type: Literal["tick"] = "tick"

    # Prices
    bid: Optional[Decimal] = Field(default=None, description="Best bid price")
    ask: Optional[Decimal] = Field(default=None, description="Best ask price")
    last: Optional[Decimal] = Field(default=None, description="Last trade price")

    # Volumes
    bid_volume: Optional[Decimal] = Field(default=None, description="Volume at best bid")
    ask_volume: Optional[Decimal] = Field(default=None, description="Volume at best ask")
    volume: Optional[Decimal] = Field(default=None, description="24h volume")

    # Timing
    exchange_timestamp: datetime = Field(description="Exchange-reported timestamp")
    local_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When we received this"
    )

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / Decimal("2")
        return None


class TradeEvent(MarketDataEvent):
    """
    Trade event - actual trade executed on exchange

    Represents a trade that happened on the exchange order book.
    """
    event_type: Literal["trade"] = "trade"

    # Trade details
    trade_id: str = Field(description="Exchange trade ID")
    price: Decimal = Field(description="Trade price")
    quantity: Decimal = Field(description="Trade quantity")
    side: Literal["buy", "sell"] = Field(description="Aggressor side")

    # Timing
    exchange_timestamp: datetime = Field(description="Exchange-reported timestamp")
    local_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When we received this"
    )

    # Additional info
    is_buyer_maker: Optional[bool] = Field(
        default=None,
        description="Was buyer the maker (passive side)"
    )


class OrderBookLevel(BaseModel):
    """Single level in order book"""
    price: Decimal
    quantity: Decimal
    num_orders: Optional[int] = None

    class Config:
        frozen = True


class OrderBookEvent(MarketDataEvent):
    """
    Order book event - full snapshot or update

    Represents the limit order book at a point in time.
    Can be full snapshot or incremental update.
    """
    event_type: Literal["order_book"] = "order_book"

    # Order book data
    bids: list[OrderBookLevel] = Field(
        default_factory=list,
        description="Bid levels (sorted high to low)"
    )
    asks: list[OrderBookLevel] = Field(
        default_factory=list,
        description="Ask levels (sorted low to high)"
    )

    # Update type
    is_snapshot: bool = Field(default=True, description="Is this a full snapshot?")
    sequence_number: Optional[int] = Field(
        default=None,
        description="Sequence number for ordering updates"
    )

    # Timing
    exchange_timestamp: datetime = Field(description="Exchange-reported timestamp")
    local_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When we received this"
    )

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / Decimal("2")
        return None

    def get_depth(self, side: Literal["bid", "ask"], levels: int = 10) -> Decimal:
        """
        Get total quantity available at top N levels

        Args:
            side: 'bid' or 'ask'
            levels: Number of levels to sum

        Returns:
            Total quantity
        """
        book = self.bids if side == "bid" else self.asks
        return sum(level.quantity for level in book[:levels])


class BarEvent(MarketDataEvent):
    """
    Bar (candlestick) event - aggregated OHLCV data

    Represents aggregated price data over a time period.
    """
    event_type: Literal["bar"] = "bar"

    # OHLCV
    open: Decimal = Field(description="Open price")
    high: Decimal = Field(description="High price")
    low: Decimal = Field(description="Low price")
    close: Decimal = Field(description="Close price")
    volume: Decimal = Field(description="Volume traded")

    # Bar timing
    bar_start: datetime = Field(description="Bar period start")
    bar_end: datetime = Field(description="Bar period end")
    interval: str = Field(description="Bar interval (e.g., '1m', '1h', '1d')")

    # Additional metrics
    num_trades: Optional[int] = Field(default=None, description="Number of trades in bar")
    quote_volume: Optional[Decimal] = Field(
        default=None,
        description="Volume in quote currency"
    )
    taker_buy_volume: Optional[Decimal] = Field(
        default=None,
        description="Volume from taker buy orders"
    )

    @property
    def is_bullish(self) -> bool:
        """Is this a bullish bar?"""
        return self.close > self.open

    @property
    def body_size(self) -> Decimal:
        """Size of candle body"""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> Decimal:
        """Size of candle range (high - low)"""
        return self.high - self.low


class QuoteEvent(MarketDataEvent):
    """
    Quote event - best bid/ask (BBO)

    Simpler than full order book, just best prices.
    """
    event_type: Literal["quote"] = "quote"

    # Best bid/ask
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal

    # Timing
    exchange_timestamp: datetime = Field(description="Exchange-reported timestamp")
    local_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When we received this"
    )

    @property
    def spread(self) -> Decimal:
        """Calculate spread"""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price"""
        return (self.bid_price + self.ask_price) / Decimal("2")

    @property
    def spread_bps(self) -> Decimal:
        """Spread in basis points"""
        return (self.spread / self.mid_price) * Decimal("10000")
