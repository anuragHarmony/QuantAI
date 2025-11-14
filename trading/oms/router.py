"""
Order Router

Smart order routing to select best exchange for execution.

Routing strategies:
- Direct: Route to specified exchange
- Best Price: Route to exchange with best price
- Best Liquidity: Route to exchange with most liquidity
- Cost: Route to exchange with lowest fees
- Split: Split order across multiple exchanges

For Phase 2B, we implement simple direct routing.
Advanced routing can be added later.
"""
from typing import Optional, List, Dict
from dataclasses import dataclass
from decimal import Decimal
from loguru import logger

from ..exchanges.base import IExchange, Order


@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    exchange: IExchange
    exchange_name: str
    reason: str
    confidence: float = 1.0  # 0-1, how confident in this routing


class OrderRouter:
    """
    Routes orders to appropriate exchange

    Currently implements simple direct routing.
    Future enhancements:
    - Best execution routing
    - Smart order routing (SOR)
    - Order splitting
    """

    def __init__(self, exchanges: Dict[str, IExchange]):
        """
        Initialize order router

        Args:
            exchanges: Dictionary of exchange_name -> IExchange
        """
        self.exchanges = exchanges

        logger.info(f"Initialized OrderRouter with {len(exchanges)} exchanges")

    async def route_order(
        self,
        order: Order,
        strategy: str = "direct"
    ) -> Optional[RoutingDecision]:
        """
        Route order to best exchange

        Args:
            order: Order to route
            strategy: Routing strategy ('direct', 'best_price', 'best_liquidity')

        Returns:
            RoutingDecision or None if no suitable exchange
        """

        if strategy == "direct":
            return await self._route_direct(order)
        elif strategy == "best_price":
            return await self._route_best_price(order)
        elif strategy == "best_liquidity":
            return await self._route_best_liquidity(order)
        else:
            logger.warning(f"Unknown routing strategy: {strategy}, using direct")
            return await self._route_direct(order)

    async def _route_direct(self, order: Order) -> Optional[RoutingDecision]:
        """
        Direct routing - use exchange specified in order metadata

        Falls back to first available exchange if not specified.
        """

        # Check if exchange specified in order metadata
        preferred_exchange = order.metadata.get("exchange")

        if preferred_exchange and preferred_exchange in self.exchanges:
            exchange = self.exchanges[preferred_exchange]

            logger.debug(f"Routing order to {preferred_exchange} (direct)")

            return RoutingDecision(
                exchange=exchange,
                exchange_name=preferred_exchange,
                reason="Direct routing - exchange specified",
                confidence=1.0
            )

        # Fall back to first available exchange
        if self.exchanges:
            exchange_name = list(self.exchanges.keys())[0]
            exchange = self.exchanges[exchange_name]

            logger.debug(f"Routing order to {exchange_name} (default)")

            return RoutingDecision(
                exchange=exchange,
                exchange_name=exchange_name,
                reason="Default routing - first available exchange",
                confidence=0.5
            )

        logger.error("No exchanges available for routing")
        return None

    async def _route_best_price(self, order: Order) -> Optional[RoutingDecision]:
        """
        Route to exchange with best price

        TODO: Query all exchanges and select best bid/ask
        """

        # For now, fall back to direct routing
        # In full implementation:
        # 1. Query latest tick from all exchanges
        # 2. Compare bid/ask prices
        # 3. Select exchange with best price
        # 4. Consider fees in calculation

        logger.warning("Best price routing not yet implemented, using direct")
        return await self._route_direct(order)

    async def _route_best_liquidity(self, order: Order) -> Optional[RoutingDecision]:
        """
        Route to exchange with best liquidity

        TODO: Query order books and select deepest market
        """

        # For now, fall back to direct routing
        # In full implementation:
        # 1. Query order book depth from all exchanges
        # 2. Calculate available liquidity at various price levels
        # 3. Select exchange that can fill order with least slippage

        logger.warning("Best liquidity routing not yet implemented, using direct")
        return await self._route_direct(order)

    def add_exchange(self, name: str, exchange: IExchange) -> None:
        """Add exchange to routing table"""
        self.exchanges[name] = exchange
        logger.info(f"Added exchange to router: {name}")

    def remove_exchange(self, name: str) -> None:
        """Remove exchange from routing table"""
        if name in self.exchanges:
            del self.exchanges[name]
            logger.info(f"Removed exchange from router: {name}")

    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchange names"""
        return list(self.exchanges.keys())
