"""
Position Tracker

Tracks individual position state with:
- Entry/exit tracking
- Average entry price calculation
- Position size tracking
- Cost basis tracking

Handles:
- Opening positions
- Adding to positions
- Reducing positions
- Closing positions
"""
from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field
from loguru import logger

from ..exchanges.base import Position, OrderSide


@dataclass
class PositionState:
    """
    Internal position state with full tracking

    Extends Position with additional tracking fields.
    """
    symbol: str
    exchange: str

    # Position basics
    quantity: Decimal = Decimal("0")
    average_entry_price: Decimal = Decimal("0")

    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Costs
    total_cost_basis: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")

    # Timestamps
    first_entry: Optional[datetime] = None
    last_update: Optional[datetime] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def to_position(self) -> Position:
        """Convert to Position object"""
        return Position(
            symbol=self.symbol,
            exchange=self.exchange,
            quantity=self.quantity,
            average_entry_price=self.average_entry_price,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            last_update=self.last_update or datetime.utcnow()
        )


class PositionTracker:
    """
    Tracks positions across all symbols

    Updates positions based on order fills and maintains
    accurate P&L calculations.
    """

    def __init__(self):
        """Initialize position tracker"""
        self.positions: Dict[str, PositionState] = {}  # symbol -> PositionState

        logger.info("Initialized PositionTracker")

    def update_from_fill(
        self,
        symbol: str,
        exchange: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal = Decimal("0")
    ) -> PositionState:
        """
        Update position from order fill

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            side: Buy or sell
            quantity: Fill quantity
            price: Fill price
            fees: Trading fees

        Returns:
            Updated position state
        """

        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = PositionState(
                symbol=symbol,
                exchange=exchange,
                first_entry=datetime.utcnow()
            )

        position = self.positions[symbol]
        previous_qty = position.quantity

        logger.debug(
            f"Updating position: {symbol} | {side.value} {quantity} @ {price} | "
            f"Previous qty: {previous_qty}"
        )

        if side == OrderSide.BUY:
            # Adding to long or reducing short
            if previous_qty >= 0:
                # Adding to long
                self._add_to_long(position, quantity, price, fees)
            else:
                # Reducing short
                self._reduce_short(position, quantity, price, fees)

        else:  # SELL
            # Reducing long or adding to short
            if previous_qty > 0:
                # Reducing long
                self._reduce_long(position, quantity, price, fees)
            else:
                # Adding to short
                self._add_to_short(position, quantity, price, fees)

        position.last_update = datetime.utcnow()

        # Remove position if quantity is zero
        if position.quantity == 0:
            logger.info(f"Position closed: {symbol} | Realized P&L: {position.realized_pnl}")
            del self.positions[symbol]
        else:
            logger.debug(
                f"Position updated: {symbol} | "
                f"Qty: {position.quantity} @ {position.average_entry_price}"
            )

        return position

    def _add_to_long(
        self,
        position: PositionState,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal
    ) -> None:
        """Add to long position"""

        # Calculate new average entry price
        total_cost = (position.quantity * position.average_entry_price) + \
                     (quantity * price) + fees

        position.quantity += quantity
        position.average_entry_price = total_cost / position.quantity
        position.total_cost_basis += (quantity * price) + fees
        position.total_fees += fees

    def _reduce_long(
        self,
        position: PositionState,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal
    ) -> None:
        """Reduce long position and realize P&L"""

        # Calculate realized P&L
        pnl = (price - position.average_entry_price) * quantity - fees
        position.realized_pnl += pnl

        position.quantity -= quantity
        position.total_fees += fees

        # If quantity goes negative, we're now short
        if position.quantity < 0:
            # Crossed to short - reset entry price
            position.average_entry_price = price
            position.total_cost_basis = abs(position.quantity) * price

    def _add_to_short(
        self,
        position: PositionState,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal
    ) -> None:
        """Add to short position"""

        # For shorts, track negative quantity
        total_cost = (abs(position.quantity) * position.average_entry_price) + \
                     (quantity * price) + fees

        position.quantity -= quantity  # Quantity goes negative
        position.average_entry_price = total_cost / abs(position.quantity)
        position.total_cost_basis += (quantity * price) + fees
        position.total_fees += fees

    def _reduce_short(
        self,
        position: PositionState,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal
    ) -> None:
        """Reduce short position and realize P&L"""

        # Calculate realized P&L (shorts profit when price drops)
        pnl = (position.average_entry_price - price) * quantity - fees
        position.realized_pnl += pnl

        position.quantity += quantity  # Reduce negative quantity
        position.total_fees += fees

        # If quantity goes positive, we're now long
        if position.quantity > 0:
            # Crossed to long - reset entry price
            position.average_entry_price = price
            position.total_cost_basis = position.quantity * price

    def update_market_prices(self, symbol: str, current_price: Decimal) -> None:
        """
        Update unrealized P&L based on current market price

        Args:
            symbol: Symbol to update
            current_price: Current market price
        """

        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate unrealized P&L
        if position.quantity > 0:
            # Long position
            position.unrealized_pnl = (current_price - position.average_entry_price) * position.quantity
        elif position.quantity < 0:
            # Short position
            position.unrealized_pnl = (position.average_entry_price - current_price) * abs(position.quantity)
        else:
            position.unrealized_pnl = Decimal("0")

        position.last_update = datetime.utcnow()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        if symbol in self.positions:
            return self.positions[symbol].to_position()
        return None

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return {
            symbol: state.to_position()
            for symbol, state in self.positions.items()
        }

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol"""
        return symbol in self.positions

    def get_position_size(self, symbol: str) -> Decimal:
        """Get position size (signed)"""
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return Decimal("0")

    def get_total_realized_pnl(self) -> Decimal:
        """Get total realized P&L across all positions"""
        return sum(pos.realized_pnl for pos in self.positions.values())

    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_total_fees(self) -> Decimal:
        """Get total fees paid"""
        return sum(pos.total_fees for pos in self.positions.values())

    def get_statistics(self) -> Dict:
        """Get position statistics"""
        return {
            "num_positions": len(self.positions),
            "long_positions": len([p for p in self.positions.values() if p.quantity > 0]),
            "short_positions": len([p for p in self.positions.values() if p.quantity < 0]),
            "total_realized_pnl": self.get_total_realized_pnl(),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "total_fees": self.get_total_fees(),
        }
