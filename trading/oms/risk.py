"""
Risk checks for Order Management System

Pre-trade risk checks to prevent:
- Excessive position sizes
- Over-concentration
- Capital breaches
- Daily loss limit violations

Each check implements RiskCheck interface and can block orders.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, date
from loguru import logger

from ..exchanges.base import Order, OrderSide, Position


@dataclass
class RiskViolation:
    """Represents a risk check violation"""
    check_name: str
    severity: str  # 'error' or 'warning'
    message: str
    current_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None


class RiskCheck(ABC):
    """Abstract base for risk checks"""

    @abstractmethod
    async def check(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> Optional[RiskViolation]:
        """
        Check if order passes risk requirements

        Args:
            order: Order to check
            current_positions: Current positions by symbol
            account_state: Current account state (balance, equity, etc)

        Returns:
            RiskViolation if check fails, None if passes
        """
        pass


class PositionLimitCheck(RiskCheck):
    """
    Check if order would exceed position limits

    Prevents taking positions larger than configured limits.
    Limits can be per-symbol or global.
    """

    def __init__(
        self,
        max_position_size: Optional[Decimal] = None,
        per_symbol_limits: Optional[Dict[str, Decimal]] = None
    ):
        """
        Initialize position limit check

        Args:
            max_position_size: Global max position size
            per_symbol_limits: Per-symbol position limits
        """
        self.max_position_size = max_position_size
        self.per_symbol_limits = per_symbol_limits or {}

        logger.info(
            f"Initialized PositionLimitCheck: global={max_position_size}, "
            f"per_symbol={len(self.per_symbol_limits)}"
        )

    async def check(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> Optional[RiskViolation]:
        """Check position limits"""

        # Get current position
        current_pos = current_positions.get(order.symbol)
        current_qty = current_pos.quantity if current_pos else Decimal("0")

        # Calculate new position after order
        if order.side == OrderSide.BUY:
            new_qty = current_qty + order.quantity
        else:
            new_qty = current_qty - order.quantity

        new_qty_abs = abs(new_qty)

        # Check symbol-specific limit
        if order.symbol in self.per_symbol_limits:
            limit = self.per_symbol_limits[order.symbol]
            if new_qty_abs > limit:
                return RiskViolation(
                    check_name="PositionLimit",
                    severity="error",
                    message=f"Order would exceed position limit for {order.symbol}",
                    current_value=new_qty_abs,
                    limit_value=limit
                )

        # Check global limit
        if self.max_position_size and new_qty_abs > self.max_position_size:
            return RiskViolation(
                check_name="PositionLimit",
                severity="error",
                message=f"Order would exceed global position limit",
                current_value=new_qty_abs,
                limit_value=self.max_position_size
            )

        return None


class OrderSizeCheck(RiskCheck):
    """
    Check if order size is within acceptable range

    Prevents orders that are too small (fees > profit potential)
    or too large (operational risk).
    """

    def __init__(
        self,
        min_order_size: Optional[Decimal] = None,
        max_order_size: Optional[Decimal] = None,
        max_order_value: Optional[Decimal] = None
    ):
        """
        Initialize order size check

        Args:
            min_order_size: Minimum order size (in base currency)
            max_order_size: Maximum order size (in base currency)
            max_order_value: Maximum order value (in quote currency)
        """
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        self.max_order_value = max_order_value

        logger.info(
            f"Initialized OrderSizeCheck: "
            f"min={min_order_size}, max={max_order_size}, max_value={max_order_value}"
        )

    async def check(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> Optional[RiskViolation]:
        """Check order size"""

        # Check minimum size
        if self.min_order_size and order.quantity < self.min_order_size:
            return RiskViolation(
                check_name="OrderSize",
                severity="error",
                message="Order size below minimum",
                current_value=order.quantity,
                limit_value=self.min_order_size
            )

        # Check maximum size
        if self.max_order_size and order.quantity > self.max_order_size:
            return RiskViolation(
                check_name="OrderSize",
                severity="error",
                message="Order size exceeds maximum",
                current_value=order.quantity,
                limit_value=self.max_order_size
            )

        # Check maximum value (requires price)
        if self.max_order_value and order.limit_price:
            order_value = order.quantity * order.limit_price
            if order_value > self.max_order_value:
                return RiskViolation(
                    check_name="OrderValue",
                    severity="error",
                    message="Order value exceeds maximum",
                    current_value=order_value,
                    limit_value=self.max_order_value
                )

        return None


class MaxLossCheck(RiskCheck):
    """
    Check if order could result in excessive loss

    Based on stop loss or worst-case scenario.
    """

    def __init__(self, max_loss_per_trade: Decimal):
        """
        Initialize max loss check

        Args:
            max_loss_per_trade: Maximum acceptable loss per trade
        """
        self.max_loss_per_trade = max_loss_per_trade

        logger.info(f"Initialized MaxLossCheck: max_loss={max_loss_per_trade}")

    async def check(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> Optional[RiskViolation]:
        """Check maximum loss"""

        # Calculate potential loss
        # (simplified - in practice, use stop loss distance)
        if order.limit_price and order.stop_price:
            if order.side == OrderSide.BUY:
                # Long: loss if price drops to stop
                potential_loss = (order.limit_price - order.stop_price) * order.quantity
            else:
                # Short: loss if price rises to stop
                potential_loss = (order.stop_price - order.limit_price) * order.quantity

            if potential_loss > self.max_loss_per_trade:
                return RiskViolation(
                    check_name="MaxLoss",
                    severity="error",
                    message="Potential loss exceeds maximum per trade",
                    current_value=potential_loss,
                    limit_value=self.max_loss_per_trade
                )

        return None


class ConcentrationCheck(RiskCheck):
    """
    Check if order would cause over-concentration

    Ensures portfolio is diversified across symbols.
    """

    def __init__(self, max_concentration_pct: Decimal = Decimal("0.3")):
        """
        Initialize concentration check

        Args:
            max_concentration_pct: Max % of portfolio in single position (0.3 = 30%)
        """
        self.max_concentration_pct = max_concentration_pct

        logger.info(
            f"Initialized ConcentrationCheck: max={max_concentration_pct * 100}%"
        )

    async def check(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> Optional[RiskViolation]:
        """Check concentration"""

        total_equity = account_state.get("equity", Decimal("0"))
        if total_equity == 0:
            return None

        # Get current position
        current_pos = current_positions.get(order.symbol)
        current_qty = current_pos.quantity if current_pos else Decimal("0")

        # Calculate new position
        if order.side == OrderSide.BUY:
            new_qty = current_qty + order.quantity
        else:
            new_qty = current_qty - order.quantity

        # Calculate position value (need price)
        if order.limit_price:
            position_value = abs(new_qty * order.limit_price)
        elif current_pos:
            position_value = abs(new_qty * current_pos.average_entry_price)
        else:
            # Can't calculate without price
            return None

        # Check concentration
        concentration = position_value / total_equity

        if concentration > self.max_concentration_pct:
            return RiskViolation(
                check_name="Concentration",
                severity="warning",  # Warning, not error
                message=f"Position concentration for {order.symbol} exceeds limit",
                current_value=concentration * Decimal("100"),  # As percentage
                limit_value=self.max_concentration_pct * Decimal("100")
            )

        return None


class DailyLossLimitCheck(RiskCheck):
    """
    Check if daily loss limit has been reached

    Stops trading for the day if losses exceed threshold.
    """

    def __init__(self, max_daily_loss: Decimal):
        """
        Initialize daily loss limit check

        Args:
            max_daily_loss: Maximum acceptable loss per day
        """
        self.max_daily_loss = max_daily_loss
        self.daily_pnl: Dict[date, Decimal] = {}

        logger.info(f"Initialized DailyLossLimitCheck: max_loss={max_daily_loss}")

    async def check(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> Optional[RiskViolation]:
        """Check daily loss limit"""

        today = date.today()
        today_pnl = self.daily_pnl.get(today, Decimal("0"))

        # Check if already at loss limit
        if today_pnl < -self.max_daily_loss:
            return RiskViolation(
                check_name="DailyLossLimit",
                severity="error",
                message="Daily loss limit reached - no new orders allowed",
                current_value=abs(today_pnl),
                limit_value=self.max_daily_loss
            )

        return None

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update daily P&L"""
        today = date.today()
        self.daily_pnl[today] = self.daily_pnl.get(today, Decimal("0")) + pnl


class CapitalCheck(RiskCheck):
    """
    Check if sufficient capital/margin available

    Prevents orders that would exceed available capital.
    """

    def __init__(self, min_margin_ratio: Decimal = Decimal("0.2")):
        """
        Initialize capital check

        Args:
            min_margin_ratio: Minimum margin ratio to maintain (0.2 = 20%)
        """
        self.min_margin_ratio = min_margin_ratio

        logger.info(f"Initialized CapitalCheck: min_margin={min_margin_ratio * 100}%")

    async def check(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> Optional[RiskViolation]:
        """Check capital availability"""

        balance = account_state.get("balance", Decimal("0"))
        equity = account_state.get("equity", Decimal("0"))
        used_margin = account_state.get("used_margin", Decimal("0"))

        # Calculate required margin for order
        if order.limit_price:
            order_value = order.quantity * order.limit_price
        else:
            # Estimate based on current market (simplified)
            order_value = order.quantity * Decimal("50000")  # Placeholder

        # Assume 1x leverage (full margin)
        required_margin = order_value

        # Check if sufficient balance
        available = balance - used_margin
        if required_margin > available:
            return RiskViolation(
                check_name="Capital",
                severity="error",
                message="Insufficient capital for order",
                current_value=available,
                limit_value=required_margin
            )

        # Check if margin ratio would be maintained
        new_used_margin = used_margin + required_margin
        new_margin_ratio = (equity - new_used_margin) / equity if equity > 0 else Decimal("0")

        if new_margin_ratio < self.min_margin_ratio:
            return RiskViolation(
                check_name="MarginRatio",
                severity="error",
                message="Order would breach minimum margin ratio",
                current_value=new_margin_ratio * Decimal("100"),
                limit_value=self.min_margin_ratio * Decimal("100")
            )

        return None


class RiskChecker:
    """
    Aggregates multiple risk checks

    Runs all configured checks and returns violations.
    """

    def __init__(self, checks: Optional[List[RiskCheck]] = None):
        """
        Initialize risk checker

        Args:
            checks: List of risk checks to run
        """
        self.checks = checks or []

        logger.info(f"Initialized RiskChecker with {len(self.checks)} checks")

    def add_check(self, check: RiskCheck) -> None:
        """Add a risk check"""
        self.checks.append(check)

    async def check_order(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_state: dict
    ) -> List[RiskViolation]:
        """
        Run all risk checks on order

        Args:
            order: Order to check
            current_positions: Current positions
            account_state: Account state

        Returns:
            List of violations (empty if all checks pass)
        """
        violations = []

        for check in self.checks:
            try:
                violation = await check.check(order, current_positions, account_state)
                if violation:
                    violations.append(violation)
                    logger.warning(
                        f"Risk check failed: {violation.check_name} - {violation.message}"
                    )
            except Exception as e:
                logger.error(f"Error in risk check {check.__class__.__name__}: {e}")
                # Don't block on check errors, but log them
                violations.append(
                    RiskViolation(
                        check_name=check.__class__.__name__,
                        severity="error",
                        message=f"Risk check error: {str(e)}"
                    )
                )

        if not violations:
            logger.debug(f"Order passed all {len(self.checks)} risk checks")

        return violations

    def has_errors(self, violations: List[RiskViolation]) -> bool:
        """Check if any violations are errors (vs warnings)"""
        return any(v.severity == "error" for v in violations)


def create_default_risk_checker(
    max_position_size: Decimal = Decimal("10"),
    max_order_size: Decimal = Decimal("5"),
    max_daily_loss: Decimal = Decimal("1000"),
    max_concentration_pct: Decimal = Decimal("0.3"),
    min_margin_ratio: Decimal = Decimal("0.2")
) -> RiskChecker:
    """
    Create risk checker with standard checks

    Args:
        max_position_size: Max position size
        max_order_size: Max order size
        max_daily_loss: Max daily loss
        max_concentration_pct: Max concentration (0.3 = 30%)
        min_margin_ratio: Min margin ratio

    Returns:
        Configured RiskChecker
    """
    checks = [
        PositionLimitCheck(max_position_size=max_position_size),
        OrderSizeCheck(max_order_size=max_order_size),
        ConcentrationCheck(max_concentration_pct=max_concentration_pct),
        DailyLossLimitCheck(max_daily_loss=max_daily_loss),
        CapitalCheck(min_margin_ratio=min_margin_ratio),
    ]

    return RiskChecker(checks=checks)
