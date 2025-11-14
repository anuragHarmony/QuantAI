"""
Slippage models for realistic simulation

Models how much execution price differs from theoretical price due to:
- Market impact (large orders move the market)
- Bid-ask spread
- Market volatility
- Liquidity constraints

Models:
- NoSlippage: Perfect execution (unrealistic but useful for testing)
- FixedSlippage: Constant percentage slippage
- VolumeSlippage: Slippage based on order size relative to volume
- SpreadSlippage: Slippage based on bid-ask spread
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional
from loguru import logger

from .base import Order, OrderSide
from ..events.market_data import TickEvent, OrderBookEvent


class ISlippageModel(ABC):
    """Abstract interface for slippage models"""

    @abstractmethod
    def calculate_slippage(
        self,
        order: Order,
        theoretical_price: Decimal,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent] = None
    ) -> Decimal:
        """
        Calculate slippage amount

        Args:
            order: The order being filled
            theoretical_price: Ideal fill price
            tick: Current market tick
            orderbook: Order book if available

        Returns:
            Actual fill price after slippage
        """
        pass


class NoSlippage(ISlippageModel):
    """
    No slippage - perfect execution

    Useful for:
    - Testing strategy logic
    - High-frequency strategies with tight spreads
    - Limit orders that are guaranteed to fill at limit
    """

    def __init__(self):
        logger.info("Initialized NoSlippage model")

    def calculate_slippage(
        self,
        order: Order,
        theoretical_price: Decimal,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent] = None
    ) -> Decimal:
        """No slippage - return theoretical price"""
        return theoretical_price


class FixedSlippage(ISlippageModel):
    """
    Fixed percentage slippage

    Simple model that adds constant slippage to all orders.
    Unrealistic but useful for conservative backtests.

    Example: 0.1% slippage means buy at 1.001x and sell at 0.999x
    """

    def __init__(self, slippage_bps: int = 10):
        """
        Initialize fixed slippage

        Args:
            slippage_bps: Slippage in basis points (10 bps = 0.1%)
        """
        self.slippage_bps = slippage_bps
        self.slippage_factor = Decimal(slippage_bps) / Decimal("10000")

        logger.info(f"Initialized FixedSlippage: {slippage_bps} bps")

    def calculate_slippage(
        self,
        order: Order,
        theoretical_price: Decimal,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent] = None
    ) -> Decimal:
        """Apply fixed slippage"""
        slippage_amount = theoretical_price * self.slippage_factor

        if order.side == OrderSide.BUY:
            # Buys cost more
            return theoretical_price + slippage_amount
        else:
            # Sells receive less
            return theoretical_price - slippage_amount


class VolumeSlippage(ISlippageModel):
    """
    Slippage based on order size relative to market volume

    Larger orders relative to average volume cause more slippage.
    More realistic for modeling market impact.

    Formula:
        slippage = base_slippage * (order_size / avg_volume) ^ impact_exponent
    """

    def __init__(
        self,
        base_slippage_bps: int = 5,
        impact_exponent: float = 0.5,
        avg_volume: Optional[Decimal] = None
    ):
        """
        Initialize volume-based slippage

        Args:
            base_slippage_bps: Base slippage for average-sized orders
            impact_exponent: How much size affects slippage (0.5 = square root)
            avg_volume: Average trading volume (if None, estimated from data)
        """
        self.base_slippage_bps = base_slippage_bps
        self.impact_exponent = impact_exponent
        self.avg_volume = avg_volume

        logger.info(
            f"Initialized VolumeSlippage: base={base_slippage_bps} bps, "
            f"exponent={impact_exponent}"
        )

    def calculate_slippage(
        self,
        order: Order,
        theoretical_price: Decimal,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent] = None
    ) -> Decimal:
        """Calculate slippage based on order size"""

        # Estimate volume if not provided
        volume = self.avg_volume
        if volume is None:
            # Use tick volume or orderbook depth as proxy
            if hasattr(tick, 'volume') and tick.volume:
                volume = tick.volume
            elif orderbook:
                # Sum top 5 levels of liquidity
                if order.side == OrderSide.BUY and orderbook.asks:
                    volume = sum(level.quantity for level in orderbook.asks[:5])
                elif order.side == OrderSide.SELL and orderbook.bids:
                    volume = sum(level.quantity for level in orderbook.bids[:5])
                else:
                    volume = order.quantity * Decimal("100")  # Fallback
            else:
                volume = order.quantity * Decimal("100")  # Fallback

        # Calculate size ratio
        size_ratio = float(order.quantity / volume) if volume > 0 else 0.01

        # Calculate slippage with impact exponent
        slippage_multiplier = size_ratio ** self.impact_exponent
        slippage_bps = self.base_slippage_bps * slippage_multiplier
        slippage_factor = Decimal(str(slippage_bps)) / Decimal("10000")

        slippage_amount = theoretical_price * slippage_factor

        if order.side == OrderSide.BUY:
            return theoretical_price + slippage_amount
        else:
            return theoretical_price - slippage_amount


class SpreadSlippage(ISlippageModel):
    """
    Slippage based on bid-ask spread

    Most realistic model - slippage depends on actual market conditions.
    Market orders cross the spread; limit orders may get partial fills.

    Features:
    - Accounts for actual spread width
    - Models partial fills when spread is wide
    - Adjusts for market volatility
    """

    def __init__(
        self,
        spread_capture_ratio: float = 0.5,
        volatility_multiplier: float = 1.0
    ):
        """
        Initialize spread-based slippage

        Args:
            spread_capture_ratio: How much of spread to capture (0.5 = half spread)
            volatility_multiplier: Multiplier for volatile markets (>1 = more slippage)
        """
        self.spread_capture_ratio = Decimal(str(spread_capture_ratio))
        self.volatility_multiplier = volatility_multiplier

        logger.info(
            f"Initialized SpreadSlippage: capture={spread_capture_ratio}, "
            f"vol_mult={volatility_multiplier}"
        )

    def calculate_slippage(
        self,
        order: Order,
        theoretical_price: Decimal,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent] = None
    ) -> Decimal:
        """Calculate slippage based on spread"""

        # Get bid-ask spread
        if tick.bid and tick.ask:
            spread = tick.ask - tick.bid
            mid_price = (tick.bid + tick.ask) / Decimal("2")
        else:
            # No spread data, use small default
            spread = theoretical_price * Decimal("0.0001")  # 1 bps
            mid_price = theoretical_price

        # Calculate spread percentage
        spread_pct = spread / mid_price if mid_price > 0 else Decimal("0.0001")

        # Apply volatility multiplier (could be calculated from recent price moves)
        adjusted_spread = spread * Decimal(str(self.volatility_multiplier))

        # Calculate slippage based on spread capture
        slippage_amount = adjusted_spread * self.spread_capture_ratio

        if order.side == OrderSide.BUY:
            # Buy at mid + portion of spread
            return mid_price + slippage_amount
        else:
            # Sell at mid - portion of spread
            return mid_price - slippage_amount


class HybridSlippage(ISlippageModel):
    """
    Combines multiple slippage models for maximum realism

    Uses:
    - Spread slippage for base cost
    - Volume slippage for market impact
    - Fixed slippage as minimum floor
    """

    def __init__(
        self,
        spread_model: Optional[SpreadSlippage] = None,
        volume_model: Optional[VolumeSlippage] = None,
        min_slippage_bps: int = 1
    ):
        """
        Initialize hybrid slippage model

        Args:
            spread_model: Spread slippage model
            volume_model: Volume slippage model
            min_slippage_bps: Minimum slippage floor
        """
        self.spread_model = spread_model or SpreadSlippage()
        self.volume_model = volume_model or VolumeSlippage()
        self.min_slippage_bps = min_slippage_bps

        logger.info("Initialized HybridSlippage model")

    def calculate_slippage(
        self,
        order: Order,
        theoretical_price: Decimal,
        tick: TickEvent,
        orderbook: Optional[OrderBookEvent] = None
    ) -> Decimal:
        """Combine multiple slippage sources"""

        # Calculate each component
        spread_price = self.spread_model.calculate_slippage(
            order, theoretical_price, tick, orderbook
        )
        volume_price = self.volume_model.calculate_slippage(
            order, theoretical_price, tick, orderbook
        )

        # Take the worse price (more slippage)
        if order.side == OrderSide.BUY:
            slipped_price = max(spread_price, volume_price)
        else:
            slipped_price = min(spread_price, volume_price)

        # Apply minimum slippage
        min_slippage = theoretical_price * Decimal(self.min_slippage_bps) / Decimal("10000")
        if order.side == OrderSide.BUY:
            slipped_price = max(slipped_price, theoretical_price + min_slippage)
        else:
            slipped_price = min(slipped_price, theoretical_price - min_slippage)

        return slipped_price


# Factory function
def create_slippage_model(
    model_type: str = "spread",
    **kwargs
) -> ISlippageModel:
    """
    Create slippage model

    Args:
        model_type: Type of model ('none', 'fixed', 'volume', 'spread', 'hybrid')
        **kwargs: Model-specific parameters

    Returns:
        Slippage model instance
    """
    models = {
        "none": NoSlippage,
        "fixed": FixedSlippage,
        "volume": VolumeSlippage,
        "spread": SpreadSlippage,
        "hybrid": HybridSlippage,
    }

    if model_type not in models:
        raise ValueError(f"Unknown slippage model: {model_type}")

    return models[model_type](**kwargs)
