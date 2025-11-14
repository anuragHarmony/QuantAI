"""
Example: Using Universe and Tickers in Strategies

Demonstrates the powerful universe system:
1. Define instrument groups (tickers)
2. Filter instruments dynamically
3. Iterate over ticker groups
4. Access instrument metadata (tags, weights)
5. Run strategies across multiple instrument groups

This is the RECOMMENDED way to manage instruments!
"""
import asyncio
from decimal import Decimal
from datetime import datetime
from loguru import logger

from trading.events import InMemoryEventBus, TickEvent
from trading.events.order import OrderSide, OrderType
from trading.exchanges import SimulatedExchange
from trading.oms import OrderManager, OrderRouter, create_default_risk_checker
from trading.portfolio import PortfolioManager
from trading.strategy import BaseStrategy, StrategyManager, StrategyConfig


class UniverseAwareStrategy(BaseStrategy):
    """
    Example strategy that uses Universe and Tickers

    This strategy demonstrates how to:
    - Access tickers (groups of instruments)
    - Filter by tags
    - Use instrument weights
    - Trade across groups
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Strategy state
        self.prices = {}  # symbol -> [prices]
        self.last_trade_time = {}  # symbol -> datetime

        logger.info(f"Initialized {self.strategy_id}")

        # Log universe information
        if self.universe:
            logger.info(f"  Universe: {len(self.universe)} instruments")
            logger.info(f"  Tickers: {self.universe.get_ticker_ids()}")

            # Show instruments by ticker
            for ticker_id in self.universe.get_ticker_ids():
                ticker = self.universe.get_ticker(ticker_id)
                logger.info(f"    {ticker_id}: {ticker.get_symbols()}")

    async def on_start(self) -> None:
        """Called when strategy starts"""
        logger.info(f"Strategy {self.strategy_id} starting...")

        if self.universe:
            # Example 1: Get specific ticker
            majors = self.get_ticker("crypto_majors")
            if majors:
                logger.info(f"\nTrading majors: {majors.get_symbols()}")

            # Example 2: Get instruments by tag
            high_liq = self.get_instruments_by_tag("high_liquidity")
            if high_liq:
                logger.info(f"High liquidity instruments: {[str(i) for i in high_liq]}")

            # Example 3: Filter universe
            btc_products = self.filter_universe(symbol_pattern="BTC/*")
            logger.info(f"BTC products: {[str(i) for i in btc_products]}")

    async def on_tick(self, tick: TickEvent) -> None:
        """Handle tick - demonstrate universe usage"""

        if not tick.last:
            return

        symbol = tick.symbol

        # Track prices
        if symbol not in self.prices:
            self.prices[symbol] = []
        self.prices[symbol].append(float(tick.last))

        # Keep only recent prices
        if len(self.prices[symbol]) > 50:
            self.prices[symbol] = self.prices[symbol][-50:]

        # Need enough history
        if len(self.prices[symbol]) < 20:
            return

        # Example trading logic using universe

        # 1. Check if this instrument is in a specific ticker
        majors_ticker = self.get_ticker("crypto_majors")
        if majors_ticker and symbol in majors_ticker:
            await self._trade_major(symbol, tick)

        # 2. Check if instrument has specific tag
        instruments = self.get_all_instruments()
        instrument = next((i for i in instruments if i.symbol == symbol), None)

        if instrument and instrument.has_tag("major"):
            # This is a major instrument - use more aggressive sizing
            pass

    async def _trade_major(self, symbol: str, tick: TickEvent) -> None:
        """Trade major cryptocurrencies"""

        # Simple momentum strategy
        prices = self.prices[symbol]

        if len(prices) < 20:
            return

        # Calculate momentum
        old_price = Decimal(str(prices[-20]))
        current_price = tick.last
        momentum = (current_price - old_price) / old_price

        # Get instrument to check weight
        instruments = self.get_all_instruments()
        instrument = next((i for i in instruments if i.symbol == symbol), None)

        # Use weight for position sizing if available
        if instrument and instrument.weight:
            position_size = instrument.weight * Decimal("1.0")  # Scaled by total allocation
        else:
            position_size = Decimal("0.1")

        # Check existing position
        position = await self.get_position(symbol)
        has_position = position is not None and position.quantity > 0

        # Buy signal
        if not has_position and momentum > Decimal("0.02"):
            logger.info(
                f"[{self.strategy_id}] BUY SIGNAL: {symbol} | "
                f"Momentum: {momentum*100:.2f}%"
            )
            await self.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=position_size,
                order_type=OrderType.MARKET
            )

        # Sell signal
        elif has_position and momentum < Decimal("-0.01"):
            logger.info(
                f"[{self.strategy_id}] SELL SIGNAL: {symbol} | "
                f"Momentum: {momentum*100:.2f}%"
            )
            await self.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.MARKET
            )


class TickerRotationStrategy(BaseStrategy):
    """
    Strategy that rotates between ticker groups

    Demonstrates:
    - Trading different tickers at different times
    - Rebalancing across groups
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_ticker = "crypto_majors"
        self.rotation_interval = 100  # Ticks between rotation
        self.tick_count = 0

    async def on_tick(self, tick: TickEvent) -> None:
        """Rotate between tickers"""

        self.tick_count += 1

        # Rotate ticker every N ticks
        if self.tick_count % self.rotation_interval == 0:
            await self._rotate_ticker()

        # Trade current ticker only
        current_ticker_obj = self.get_ticker(self.current_ticker)
        if current_ticker_obj and tick.symbol in current_ticker_obj:
            # Trade this instrument
            pass

    async def _rotate_ticker(self):
        """Rotate to next ticker"""
        if not self.universe:
            return

        tickers = self.universe.get_ticker_ids()
        if not tickers:
            return

        current_idx = tickers.index(self.current_ticker) if self.current_ticker in tickers else 0
        next_idx = (current_idx + 1) % len(tickers)
        self.current_ticker = tickers[next_idx]

        logger.info(f"[{self.strategy_id}] Rotated to ticker: {self.current_ticker}")


async def universe_example():
    """Demonstrate universe system"""

    logger.info("=== Universe System Example ===\n")

    # 1. Create infrastructure
    event_bus = InMemoryEventBus()
    portfolio = PortfolioManager(event_bus, Decimal("100000"))
    exchange = SimulatedExchange(fill_model="aggressive", slippage_model="none")

    await portfolio.start()
    await exchange.connect()

    # OMS
    risk_checker = create_default_risk_checker()
    router = OrderRouter(exchanges={"simulated": exchange})
    oms = OrderManager(event_bus, risk_checker, router, portfolio)
    await oms.start()

    # 2. Load strategy with universe configuration
    logger.info("--- Loading Strategy with Universe ---\n")

    strategy_manager = StrategyManager(event_bus, portfolio, oms)
    strategy_manager.register_strategy_class("Multi_Asset_Portfolio", UniverseAwareStrategy)

    try:
        strategy_id = await strategy_manager.load_strategy(
            "configs/strategies/universe_example.yaml",
            strategy_class=UniverseAwareStrategy
        )
        logger.info(f"Loaded strategy: {strategy_id}\n")
    except Exception as e:
        logger.error(f"Failed to load strategy: {e}")
        return

    # 3. Get the strategy and explore universe
    strategy = strategy_manager.get_strategy(strategy_id)

    if strategy and strategy.universe:
        universe = strategy.universe

        logger.info("--- Universe Information ---")
        logger.info(f"Name: {universe.name}")
        logger.info(f"Total Instruments: {len(universe)}")
        logger.info(f"Tickers: {universe.get_ticker_ids()}\n")

        # Show each ticker
        for ticker_id in universe.get_ticker_ids():
            ticker = universe.get_ticker(ticker_id)
            logger.info(f"\nTicker: {ticker_id}")
            logger.info(f"  Instruments: {ticker.count()}")
            for instrument in ticker:
                tags_str = f" [{', '.join(instrument.tags)}]" if instrument.tags else ""
                weight_str = f" weight={instrument.weight}" if instrument.weight else ""
                logger.info(f"    - {instrument.symbol}{tags_str}{weight_str}")

        # Demonstrate filtering
        logger.info("\n--- Filtering Examples ---")

        # Get all major instruments
        majors = universe.get_by_tag("major")
        logger.info(f"\nMajor instruments: {[str(i) for i in majors]}")

        # Get all spot instruments
        from trading.universe import InstrumentType
        spot = universe.get_by_type(InstrumentType.SPOT)
        logger.info(f"Spot instruments: {[str(i) for i in spot]}")

        # Filter with pattern
        btc_instruments = universe.filter(symbol_pattern="BTC/*")
        logger.info(f"BTC products: {[str(i) for i in btc_instruments]}")

    # 4. Start strategy
    logger.info("\n--- Starting Strategy ---\n")
    await strategy_manager.start_all()

    # 5. Simulate market data
    logger.info("--- Simulating Market Data ---\n")

    market_data = exchange.get_market_data_connector()

    # Get symbols from universe
    if strategy.universe:
        symbols_to_simulate = strategy.universe.get_symbols()[:3]  # First 3 for demo
    else:
        symbols_to_simulate = ["BTC/USDT", "ETH/USDT"]

    logger.info(f"Simulating data for: {symbols_to_simulate}\n")

    for i in range(50):
        for symbol in symbols_to_simulate:
            # Generate price
            base_price = Decimal("50000") if "BTC" in symbol else Decimal("3000")
            price_change = Decimal(i * 10) if i < 25 else Decimal((50-i) * 10)
            price = base_price + price_change

            tick = TickEvent(
                exchange="simulated",
                symbol=symbol,
                raw_symbol=symbol,
                bid=price - Decimal("1"),
                ask=price + Decimal("1"),
                last=price,
                exchange_timestamp=datetime.utcnow()
            )

            await market_data._handle_event(tick)

        await asyncio.sleep(0.05)

    # 6. Results
    logger.info("\n--- Results ---\n")

    stats = portfolio.get_statistics()
    logger.info("Portfolio Statistics:")
    for key, value in stats.items():
        if isinstance(value, Decimal):
            logger.info(f"  {key}: ${value:,.2f}" if "pct" not in key else f"  {key}: {value:.2f}%")
        else:
            logger.info(f"  {key}: {value}")

    # 7. Cleanup
    await strategy_manager.stop_all()
    await oms.stop()
    await portfolio.stop()
    await exchange.disconnect()
    await event_bus.close()

    logger.info("\nExample complete!")


async def main():
    """Main entry point"""
    await universe_example()


if __name__ == "__main__":
    asyncio.run(main())
