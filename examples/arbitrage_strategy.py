"""
Cross-Exchange Arbitrage Strategy Example

Implements a cross-exchange arbitrage strategy that exploits price differences
between multiple exchanges for the same asset:

Strategy Logic:
1. Monitor the same symbol on multiple exchanges simultaneously
2. Calculate price spread between exchanges
3. Execute when spread exceeds transaction costs + minimum profit threshold
4. Buy on cheaper exchange, sell on expensive exchange
5. Manage inventory to avoid imbalanced positions
6. Account for fees, slippage, and transfer times

Arbitrage Opportunity Criteria:
- Spread > (Fee_Buy + Fee_Sell + Slippage + Min_Profit)
- Sufficient liquidity on both exchanges
- Inventory limits not exceeded
- No open arbitrage in same direction

Risk Management:
- Maximum position per exchange
- Inventory balance limits
- Spread size limits
- Execution timeout handling

This is a simplified arbitrage example. Real-world arbitrage requires:
- Ultra-low latency connections
- Deep liquidity analysis
- Transfer time consideration
- Regulatory compliance
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, field

from trading.strategy.base import BaseStrategy
from trading.strategy.config import StrategyConfig
from trading.events import TickEvent, FillEvent
from trading.backtest import BacktestEngine, BacktestConfig


@dataclass
class ExchangePrice:
    """Price information for a symbol on an exchange"""
    exchange: str
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: datetime

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price"""
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread_bps(self) -> Decimal:
        """Calculate bid-ask spread in basis points"""
        if self.mid_price == 0:
            return Decimal("0")
        return ((self.ask - self.bid) / self.mid_price) * Decimal("10000")


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    gross_spread_bps: Decimal
    net_spread_bps: Decimal
    expected_profit_pct: Decimal
    timestamp: datetime
    executed: bool = False


class ArbitrageStrategy(BaseStrategy):
    """
    Cross-Exchange Arbitrage Strategy

    Exploits price differences for the same asset across different exchanges.
    Requires real-time price feeds from multiple exchanges.
    """

    def __init__(self, config, event_bus, portfolio_manager, order_manager):
        super().__init__(config, event_bus, portfolio_manager, order_manager)

        # Strategy parameters
        self.min_profit_bps = Decimal(str(config.parameters.get("min_profit_bps", 10)))  # 10 bps = 0.1%
        self.max_position_size_pct = Decimal(str(config.parameters.get("max_position_size_pct", 0.20)))  # 20%
        self.max_inventory_imbalance_pct = Decimal(str(config.parameters.get("max_inventory_imbalance_pct", 0.10)))  # 10%

        # Transaction costs
        self.fee_bps = Decimal(str(config.parameters.get("fee_bps", 10)))  # 10 bps per side
        self.slippage_bps = Decimal(str(config.parameters.get("slippage_bps", 5)))  # 5 bps slippage

        # Execution parameters
        self.max_stale_price_seconds = config.parameters.get("max_stale_price_seconds", 5)
        self.require_simultaneous_execution = config.parameters.get("require_simultaneous_execution", True)

        # Price tracking per exchange
        self.exchange_prices: Dict[Tuple[str, str], ExchangePrice] = {}  # (exchange, symbol) -> ExchangePrice

        # Position tracking per exchange
        self.exchange_positions: Dict[Tuple[str, str], Decimal] = {}  # (exchange, symbol) -> quantity

        # Arbitrage tracking
        self.opportunities: List[ArbitrageOpportunity] = []
        self.active_arbitrages: Dict[str, ArbitrageOpportunity] = {}  # symbol -> opportunity

        # Statistics
        self.opportunities_detected = 0
        self.opportunities_executed = 0
        self.opportunities_missed = 0
        self.total_arbitrage_profit = Decimal("0")

        logger.info(
            f"Initialized Arbitrage Strategy: "
            f"Min profit: {self.min_profit_bps} bps, "
            f"Max position: {self.max_position_size_pct:.1%}, "
            f"Fees: {self.fee_bps} bps, "
            f"Slippage: {self.slippage_bps} bps"
        )

    async def on_start(self):
        """Called when strategy starts"""
        logger.info("=" * 70)
        logger.info("Cross-Exchange Arbitrage Strategy Started")
        logger.info(f"Subscribed symbols: {self.config.subscriptions.get('symbols', [])}")
        logger.info(f"Subscribed exchanges: {self.config.subscriptions.get('exchanges', [])}")
        logger.info(f"Minimum profit threshold: {self.min_profit_bps} bps")
        logger.info("=" * 70)

    async def on_stop(self):
        """Called when strategy stops"""
        logger.info("=" * 70)
        logger.info("Cross-Exchange Arbitrage Strategy Stopped")
        logger.info(f"Opportunities detected: {self.opportunities_detected}")
        logger.info(f"Opportunities executed: {self.opportunities_executed}")
        logger.info(f"Opportunities missed: {self.opportunities_missed}")
        logger.info(f"Total arbitrage profit: ${self.total_arbitrage_profit:.2f}")
        if self.opportunities_detected > 0:
            success_rate = (self.opportunities_executed / self.opportunities_detected) * 100
            logger.info(f"Execution success rate: {success_rate:.1f}%")
        logger.info("=" * 70)

    async def on_tick(self, tick: TickEvent):
        """Process tick event and detect arbitrage opportunities"""

        # Update price data
        key = (tick.exchange, tick.symbol)
        self.exchange_prices[key] = ExchangePrice(
            exchange=tick.exchange,
            symbol=tick.symbol,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            timestamp=tick.exchange_timestamp or datetime.now()
        )

        # Check for arbitrage opportunities
        await self._scan_for_opportunities(tick.symbol)

    async def on_fill(self, fill: FillEvent):
        """Handle order fill events"""
        key = (fill.exchange, fill.symbol)

        # Update position tracking
        if fill.side == "buy":
            current_pos = self.exchange_positions.get(key, Decimal("0"))
            self.exchange_positions[key] = current_pos + fill.quantity

            logger.info(
                f"📍 Arbitrage BUY filled: {fill.symbol} @ {fill.exchange} | "
                f"Price: {fill.price:.2f} | Qty: {fill.quantity:.5f}"
            )

        elif fill.side == "sell":
            current_pos = self.exchange_positions.get(key, Decimal("0"))
            self.exchange_positions[key] = current_pos - fill.quantity

            logger.info(
                f"📍 Arbitrage SELL filled: {fill.symbol} @ {fill.exchange} | "
                f"Price: {fill.price:.2f} | Qty: {fill.quantity:.5f}"
            )

    async def _scan_for_opportunities(self, symbol: str):
        """Scan for arbitrage opportunities for a symbol"""

        # Get all exchanges with prices for this symbol
        exchanges_with_prices = [
            (exchange, price)
            for (exchange, sym), price in self.exchange_prices.items()
            if sym == symbol
        ]

        if len(exchanges_with_prices) < 2:
            return  # Need at least 2 exchanges

        # Check all exchange pairs
        for i, (exchange1, price1) in enumerate(exchanges_with_prices):
            for exchange2, price2 in exchanges_with_prices[i+1:]:

                # Check if prices are fresh
                if not self._are_prices_fresh(price1, price2):
                    continue

                # Calculate opportunity in both directions
                await self._check_arbitrage_opportunity(symbol, exchange1, price1, exchange2, price2)
                await self._check_arbitrage_opportunity(symbol, exchange2, price2, exchange1, price1)

    def _are_prices_fresh(self, price1: ExchangePrice, price2: ExchangePrice) -> bool:
        """Check if both prices are recent enough"""
        now = datetime.now()
        max_age = timedelta(seconds=self.max_stale_price_seconds)

        age1 = now - price1.timestamp
        age2 = now - price2.timestamp

        return age1 <= max_age and age2 <= max_age

    async def _check_arbitrage_opportunity(
        self,
        symbol: str,
        buy_exchange: str,
        buy_price: ExchangePrice,
        sell_exchange: str,
        sell_price: ExchangePrice
    ):
        """Check for arbitrage opportunity: buy on exchange1, sell on exchange2"""

        # Get execution prices (buy at ask, sell at bid)
        execution_buy_price = buy_price.ask
        execution_sell_price = sell_price.bid

        # Calculate gross spread
        if execution_buy_price == 0:
            return

        gross_spread_bps = ((execution_sell_price - execution_buy_price) / execution_buy_price) * Decimal("10000")

        # Calculate costs
        total_cost_bps = (self.fee_bps * 2) + (self.slippage_bps * 2)  # Both sides

        # Calculate net spread
        net_spread_bps = gross_spread_bps - total_cost_bps

        # Check if profitable
        if net_spread_bps < self.min_profit_bps:
            return

        # Check if we already have an active arbitrage for this symbol
        if symbol in self.active_arbitrages:
            return

        # Check position limits
        if not await self._check_position_limits(symbol, buy_exchange, sell_exchange):
            return

        # Create opportunity
        opportunity = ArbitrageOpportunity(
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=execution_buy_price,
            sell_price=execution_sell_price,
            gross_spread_bps=gross_spread_bps,
            net_spread_bps=net_spread_bps,
            expected_profit_pct=net_spread_bps / Decimal("100"),
            timestamp=datetime.now()
        )

        self.opportunities_detected += 1
        self.opportunities.append(opportunity)

        logger.info(
            f"💎 ARBITRAGE OPPORTUNITY: {symbol} | "
            f"Buy @ {buy_exchange}: {execution_buy_price:.2f} | "
            f"Sell @ {sell_exchange}: {execution_sell_price:.2f} | "
            f"Gross: {gross_spread_bps:.1f} bps | "
            f"Net: {net_spread_bps:.1f} bps | "
            f"Expected profit: {opportunity.expected_profit_pct:.3f}%"
        )

        # Execute arbitrage
        await self._execute_arbitrage(opportunity)

    async def _check_position_limits(self, symbol: str, buy_exchange: str, sell_exchange: str) -> bool:
        """Check if position limits allow this arbitrage"""

        # Get current equity
        stats = self.portfolio_manager.get_statistics()
        current_equity = stats["current_equity"]
        max_position_value = current_equity * self.max_position_size_pct

        # Get current positions on both exchanges
        buy_key = (buy_exchange, symbol)
        sell_key = (sell_exchange, symbol)

        buy_position = self.exchange_positions.get(buy_key, Decimal("0"))
        sell_position = self.exchange_positions.get(sell_key, Decimal("0"))

        # Check individual exchange limits
        buy_price_info = self.exchange_prices.get(buy_key)
        if buy_price_info:
            current_buy_value = buy_position * buy_price_info.mid_price
            if current_buy_value >= max_position_value:
                logger.debug(f"Position limit reached on {buy_exchange}")
                return False

        # Check inventory imbalance
        total_position = buy_position + sell_position
        max_imbalance = current_equity * self.max_inventory_imbalance_pct

        if abs(total_position) * buy_price_info.mid_price > max_imbalance if buy_price_info else False:
            logger.debug(f"Inventory imbalance too high for {symbol}")
            return False

        return True

    async def _execute_arbitrage(self, opportunity: ArbitrageOpportunity):
        """Execute the arbitrage opportunity"""

        symbol = opportunity.symbol

        # Mark as active
        self.active_arbitrages[symbol] = opportunity

        # Calculate position size
        stats = self.portfolio_manager.get_statistics()
        position_value = stats["current_equity"] * self.max_position_size_pct
        quantity = position_value / opportunity.buy_price
        quantity = quantity.quantize(Decimal("0.00001"))

        if quantity <= 0:
            logger.warning(f"Calculated quantity is zero or negative for {symbol}")
            self.opportunities_missed += 1
            return

        logger.info(
            f"⚡ EXECUTING ARBITRAGE: {symbol} | "
            f"Qty: {quantity:.5f} | "
            f"Buy @ {opportunity.buy_exchange} | "
            f"Sell @ {opportunity.sell_exchange}"
        )

        try:
            if self.require_simultaneous_execution:
                # Execute both legs simultaneously
                await asyncio.gather(
                    self._place_arbitrage_leg(
                        symbol=symbol,
                        side="buy",
                        quantity=quantity,
                        exchange=opportunity.buy_exchange
                    ),
                    self._place_arbitrage_leg(
                        symbol=symbol,
                        side="sell",
                        quantity=quantity,
                        exchange=opportunity.sell_exchange
                    )
                )
            else:
                # Execute sequentially (buy first, then sell)
                await self._place_arbitrage_leg(
                    symbol=symbol,
                    side="buy",
                    quantity=quantity,
                    exchange=opportunity.buy_exchange
                )

                await self._place_arbitrage_leg(
                    symbol=symbol,
                    side="sell",
                    quantity=quantity,
                    exchange=opportunity.sell_exchange
                )

            # Mark as executed
            opportunity.executed = True
            self.opportunities_executed += 1

            # Estimate profit
            estimated_profit = (opportunity.sell_price - opportunity.buy_price) * quantity
            self.total_arbitrage_profit += estimated_profit

            logger.info(
                f"✅ Arbitrage executed successfully | "
                f"Estimated profit: ${estimated_profit:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to execute arbitrage for {symbol}: {e}")
            self.opportunities_missed += 1

        finally:
            # Remove from active
            self.active_arbitrages.pop(symbol, None)

    async def _place_arbitrage_leg(self, symbol: str, side: str, quantity: Decimal, exchange: str):
        """Place one leg of the arbitrage trade"""

        await self.place_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            exchange=exchange
        )


async def run_arbitrage_backtest():
    """Run backtest with arbitrage strategy"""
    logger.info("=" * 70)
    logger.info("CROSS-EXCHANGE ARBITRAGE STRATEGY BACKTEST")
    logger.info("=" * 70)

    # Strategy configuration
    strategy_config = StrategyConfig(
        name="Cross_Exchange_Arbitrage",
        strategy_id="arbitrage_v1",
        enabled=True,
        mode="simulation",
        subscriptions={
            "symbols": ["BTC/USDT", "ETH/USDT"],
            # In real scenario, would subscribe to multiple exchanges
            # For simulation, we'd need to simulate price differences
            "exchanges": ["simulated"],
            "data_types": ["tick"]
        },
        parameters={
            # Profitability threshold
            "min_profit_bps": 10,  # 0.1% minimum profit

            # Position sizing
            "max_position_size_pct": 0.20,  # 20% max position
            "max_inventory_imbalance_pct": 0.10,  # 10% max imbalance

            # Transaction costs
            "fee_bps": 10,  # 0.1% fees per side
            "slippage_bps": 5,  # 0.05% slippage per side

            # Execution
            "max_stale_price_seconds": 5,
            "require_simultaneous_execution": True,
        }
    )

    # Backtest configuration
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),  # 1 month
        initial_capital=Decimal("100000"),
        fill_model="aggressive",  # Arbitrage needs fast fills
        slippage_model="fixed",
        slippage_bps=Decimal("5"),
        maker_fee=Decimal("0.0010"),  # 10 bps
        taker_fee=Decimal("0.0010"),  # 10 bps
        data_frequency="1m",
        enable_risk_checks=True
    )

    # Note: This backtest simulation won't show real arbitrage opportunities
    # as it only uses one simulated exchange. In production, you'd need:
    # 1. Multiple live exchange connections
    # 2. Real-time price feeds from all exchanges
    # 3. Ultra-low latency infrastructure

    logger.warning(
        "⚠️  NOTE: This is a demonstration backtest. "
        "Real arbitrage requires multiple live exchange connections "
        "and real-time price feeds with price discrepancies."
    )

    # Run backtest
    engine = BacktestEngine(
        strategy_class=ArbitrageStrategy,
        strategy_config=strategy_config,
        config=backtest_config
    )

    result = await engine.run()

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Final P&L: ${result.total_pnl:,.2f}")
    logger.info(f"📈 Return: {result.total_return_pct:.2f}%")
    logger.info(f"🎯 Win Rate: {result.win_rate:.2f}%")
    logger.info(f"📉 Max Drawdown: {result.max_drawdown_pct:.2f}%")
    logger.info("=" * 70)

    return result


async def main():
    """Main entry point"""
    await run_arbitrage_backtest()


if __name__ == "__main__":
    asyncio.run(main())
