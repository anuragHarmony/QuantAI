"""
Order Management System (OMS)

Central component for managing order lifecycle with:
- Pre-trade risk checks
- Order routing to exchanges
- Order state tracking
- Event-driven architecture

Components:
- OrderManager: Core OMS orchestrator
- RiskChecker: Pre-trade risk validation
- OrderRouter: Smart order routing

Usage:
    # Create OMS
    oms = OrderManager(event_bus, portfolio, exchanges)
    await oms.start()

    # Submit order (will check risk and route)
    result = await oms.submit_order(order)

    # OMS handles all state transitions via events
"""

from .risk import (
    RiskCheck,
    RiskViolation,
    PositionLimitCheck,
    OrderSizeCheck,
    MaxLossCheck,
    ConcentrationCheck,
    DailyLossLimitCheck,
    CapitalCheck,
    RiskChecker,
    create_default_risk_checker,
)

from .router import OrderRouter, RoutingDecision

from .manager import OrderManager

__all__ = [
    # Risk
    "RiskCheck",
    "RiskViolation",
    "PositionLimitCheck",
    "OrderSizeCheck",
    "MaxLossCheck",
    "ConcentrationCheck",
    "DailyLossLimitCheck",
    "CapitalCheck",
    "RiskChecker",
    "create_default_risk_checker",
    # Routing
    "OrderRouter",
    "RoutingDecision",
    # Manager
    "OrderManager",
]
