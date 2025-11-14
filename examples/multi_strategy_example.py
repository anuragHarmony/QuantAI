"""
Example: Multiple Strategies Running Concurrently

Demonstrates:
1. Loading strategies from config files
2. Running multiple strategies simultaneously
3. Each strategy with independent state
4. Position seeding from files
5. Real-time monitoring of all strategies

This shows the complete production setup.
"""
import asyncio
from decimal import Decimal
from loguru import logger

from trading.events import InMemoryEventBus, TickEvent, OrderFilledEvent
from trading.events.order import OrderSide, OrderType
from trading.exchanges import SimulatedExchange
from trading.oms import OrderManager, OrderRouter, create_default_risk_checker
from trading.portfolio import PortfolioManager
from trading.strategy import BaseStrategy, StrategyManager, StrategyConfig


# Define example strategies

class BTCMovingAverageStrategy(BaseStrategy):
    """
    Simple BTC moving average crossover strategy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Strategy state
        self.fast_period = self.config.parameters.get("fast_period", 10)
        self.slow_period = self.config.parameters.get("slow_period", 30)
        self.position_size = Decimal(str(self.config.parameters.get("position_size", 0.1)))

        self.prices = []
        self.in_position = False

        logger.info(
            f"Initialized BTC MA Strategy: "
            f"fast={self.fast_period}, slow={self.slow_period}"
        )

    async def on_tick(self, tick: TickEvent) -> None:
        """Handle tick - calculate MAs and trade"""

        if not tick.last or tick.symbol != "BTC/USDT":
            return

        # Update price history
        self.prices.append(float(tick.last))

        # Need enough history
        if len(self.prices) < self.slow_period:
            return

        # Calculate MAs
        fast_ma = sum(self.prices[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period:]) / self.slow_period

        # Previous MAs for crossover detection
        prev_fast_ma = sum(self.prices[-self.fast_period-1:-1]) / self.fast_period
        prev_slow_ma = sum(self.prices[-self.slow_period-1:-1]) / self.slow_period

        # Check for position
        position = await self.get_position("BTC/USDT")
        self.in_position = position is not None and position.quantity > 0

        # Buy signal: fast crosses above slow
        if not self.in_position and fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
            logger.info(f"[{self.strategy_id}] BUY SIGNAL: Fast MA crossed above Slow MA")
            await self.submit_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=self.position_size,
                order_type=OrderType.MARKET
            )

        # Sell signal: fast crosses below slow
        elif self.in_position and fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
            logger.info(f"[{self.strategy_id}] SELL SIGNAL: Fast MA crossed below Slow MA")
            await self.submit_order(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=self.position_size,
                order_type=OrderType.MARKET
            )


class ETHMomentumStrategy(BaseStrategy):
    """
    Simple ETH momentum strategy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lookback = self.config.parameters.get("lookback_period", 20)
        self.threshold = Decimal(str(self.config.parameters.get("momentum_threshold", 0.02)))
        self.position_size = Decimal(str(self.config.parameters.get("position_size", 1.0)))

        self.prices = []

        logger.info(
            f"Initialized ETH Momentum Strategy: "
            f"lookback={self.lookback}, threshold={self.threshold}"
        )

    async def on_tick(self, tick: TickEvent) -> None:
        """Handle tick - calculate momentum and trade"""

        if not tick.last or tick.symbol != "ETH/USDT":
            return

        self.prices.append(float(tick.last))

        if len(self.prices) < self.lookback:
            return

        # Calculate momentum
        old_price = Decimal(str(self.prices[-self.lookback]))
        current_price = tick.last
        momentum = (current_price - old_price) / old_price

        position = await self.get_position("ETH/USDT")
        has_position = position is not None and position.quantity > 0

        # Strong positive momentum - buy
        if not has_position and momentum > self.threshold:
            logger.info(
                f"[{self.strategy_id}] BUY SIGNAL: "
                f"Momentum={momentum*100:.2f}% > {self.threshold*100:.2f}%"
            )
            await self.submit_order(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                quantity=self.position_size,
                order_type=OrderType.MARKET
            )

        # Negative momentum - sell
        elif has_position and momentum < -self.threshold:
            logger.info(
                f"[{self.strategy_id}] SELL SIGNAL: "
                f"Momentum={momentum*100:.2f}% < {-self.threshold*100:.2f}%"
            )
            await self.submit_order(
                symbol="ETH/USDT",
                side=OrderSide.SELL,
                quantity=self.position_size,
                order_type=OrderType.MARKET
            )


async def multi_strategy_example():
    """Run multiple strategies concurrently"""

    logger.info("=== Multi-Strategy Example ===\n")

    # 1. Create infrastructure
    event_bus = InMemoryEventBus()

    # Portfolio manager
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=Decimal("100000"),
        update_interval=2.0
    )
    await portfolio.start()

    # Simulated exchange
    exchange = SimulatedExchange(
        fill_model="aggressive",
        slippage_model="none",
        initial_balance=Decimal("100000")
    )
    await exchange.connect()

    # OMS
    risk_checker = create_default_risk_checker(
        max_position_size=Decimal("10"),
        max_order_size=Decimal("5"),
        max_daily_loss=Decimal("5000")
    )
    order_router = OrderRouter(exchanges={"simulated": exchange})
    oms = OrderManager(
        event_bus=event_bus,
        risk_checker=risk_checker,
        order_router=order_router,
        portfolio_manager=portfolio
    )
    await oms.start()

    # 2. Create Strategy Manager
    strategy_manager = StrategyManager(
        event_bus=event_bus,
        portfolio_manager=portfolio,
        order_manager=oms
    )

    # Register strategy classes
    strategy_manager.register_strategy_class("BTC_MA_Cross", BTCMovingAverageStrategy)
    strategy_manager.register_strategy_class("ETH_Momentum", ETHMomentumStrategy)

    logger.info("Infrastructure initialized\n")

    # 3. Load strategies from config files
    logger.info("--- Loading Strategies ---\n")

    # Load BTC strategy
    try:
        btc_id = await strategy_manager.load_strategy(
            "configs/strategies/example_btc_strategy.yaml",
            strategy_class=BTCMovingAverageStrategy
        )
        logger.info(f"Loaded: {btc_id}")
    except Exception as e:
        logger.error(f"Failed to load BTC strategy: {e}")

    # Load ETH strategy
    try:
        eth_id = await strategy_manager.load_strategy(
            "configs/strategies/example_eth_strategy.yaml",
            strategy_class=ETHMomentumStrategy
        )
        logger.info(f"Loaded: {eth_id}")
    except Exception as e:
        logger.error(f"Failed to load ETH strategy: {e}")

    # List loaded strategies
    logger.info("\n--- Loaded Strategies ---")
    for strategy_info in strategy_manager.list_strategies():
        logger.info(
            f"{strategy_info['name']} (ID: {strategy_info['strategy_id']}) | "
            f"Symbols: {strategy_info['symbols']} | "
            f"Enabled: {strategy_info['enabled']}"
        )

    # 4. Start all strategies
    logger.info("\n--- Starting Strategies ---\n")
    started = await strategy_manager.start_all()
    logger.info(f"Started {started} strategies\n")

    # 5. Simulate market data
    logger.info("--- Simulating Market Data ---\n")

    market_data = exchange.get_market_data_connector()

    # Simulate 100 ticks for both BTC and ETH
    for i in range(100):
        # BTC price (trending)
        if i < 40:
            btc_price = Decimal("50000") + Decimal(i * 10)
        elif i < 70:
            btc_price = Decimal("50400") - Decimal((i - 40) * 5)
        else:
            btc_price = Decimal("50250") + Decimal((i - 70) * 8)

        btc_tick = TickEvent(
            exchange="simulated",
            symbol="BTC/USDT",
            raw_symbol="BTC/USDT",
            bid=btc_price - Decimal("1"),
            ask=btc_price + Decimal("1"),
            last=btc_price,
            exchange_timestamp=datetime.utcnow()
        )

        # ETH price (volatile)
        eth_base = Decimal("3000")
        eth_variance = Decimal(((i % 20) - 10) * 5)
        eth_price = eth_base + eth_variance

        eth_tick = TickEvent(
            exchange="simulated",
            symbol="ETH/USDT",
            raw_symbol="ETH/USDT",
            bid=eth_price - Decimal("1"),
            ask=eth_price + Decimal("1"),
            last=eth_price,
            exchange_timestamp=datetime.utcnow()
        )

        # Send ticks
        await market_data._handle_event(btc_tick)
        await market_data._handle_event(eth_tick)

        await asyncio.sleep(0.05)  # Small delay

    logger.info("\n--- Market Data Complete ---\n")

    # 6. Get results
    await asyncio.sleep(1)  # Let final events process

    logger.info("--- Strategy Statistics ---\n")
    sm_stats = strategy_manager.get_statistics()
    for key, value in sm_stats.items():
        logger.info(f"{key}: {value}")

    logger.info("\n--- Portfolio Statistics ---\n")
    pf_stats = portfolio.get_statistics()
    for key, value in pf_stats.items():
        if isinstance(value, Decimal):
            logger.info(f"{key}: ${value:,.2f}" if "pct" not in key else f"{key}: {value:.2f}%")
        else:
            logger.info(f"{key}: {value}")

    logger.info("\n--- OMS Statistics ---\n")
    oms_stats = oms.get_statistics()
    for key, value in oms_stats.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2%}")
        else:
            logger.info(f"{key}: {value}")

    # 7. List final positions
    logger.info("\n--- Final Positions ---")
    positions = await portfolio.get_all_positions()
    if positions:
        for symbol, pos in positions.items():
            logger.info(
                f"{symbol}: {pos.quantity} @ ${pos.average_entry_price:,.2f} | "
                f"P&L: ${pos.unrealized_pnl + pos.realized_pnl:,.2f}"
            )
    else:
        logger.info("No open positions")

    # 8. Stop strategies
    logger.info("\n--- Stopping Strategies ---\n")
    await strategy_manager.stop_all()

    # 9. Cleanup
    await oms.stop()
    await portfolio.stop()
    await exchange.disconnect()
    await event_bus.close()

    logger.info("\nExample complete!")


from datetime import datetime


async def main():
    """Main entry point"""
    await multi_strategy_example()


if __name__ == "__main__":
    asyncio.run(main())
