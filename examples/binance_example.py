"""
Example: Live Trading with Binance

Demonstrates how to:
1. Connect to Binance
2. Subscribe to real-time market data
3. Place and manage orders
4. Track positions

IMPORTANT: You need valid Binance API credentials to run this.
For testing, use testnet=True to connect to Binance testnet.
"""
import asyncio
import os
from decimal import Decimal
from loguru import logger

from trading.events import TickEvent, TradeEvent, OrderBookEvent
from trading.exchanges import (
    BinanceExchange,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)


async def handle_tick(tick: TickEvent) -> None:
    """Handle tick updates"""
    logger.info(
        f"TICK: {tick.symbol} | "
        f"Bid: {tick.bid} | Ask: {tick.ask} | Last: {tick.last}"
    )


async def handle_trade(trade: TradeEvent) -> None:
    """Handle trade updates"""
    logger.info(
        f"TRADE: {trade.symbol} | "
        f"{trade.side.upper()} {trade.quantity} @ {trade.price}"
    )


async def handle_orderbook(orderbook: OrderBookEvent) -> None:
    """Handle order book updates"""
    if orderbook.bids and orderbook.asks:
        best_bid = orderbook.bids[0]
        best_ask = orderbook.asks[0]

        logger.info(
            f"BOOK: {orderbook.symbol} | "
            f"Bid: {best_bid.price} ({best_bid.quantity}) | "
            f"Ask: {best_ask.price} ({best_ask.quantity})"
        )


async def market_data_example():
    """Example: Subscribe to market data"""

    logger.info("=== Market Data Example ===")

    # Get API credentials from environment
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        logger.warning(
            "No API credentials found. Set BINANCE_API_KEY and BINANCE_API_SECRET "
            "environment variables. Using testnet=True for demo."
        )
        api_key = "test"
        api_secret = "test"

    # Create exchange (use testnet for testing)
    exchange = BinanceExchange(
        api_key=api_key,
        api_secret=api_secret,
        market_type="spot",
        testnet=True  # Set to False for production
    )

    try:
        # Connect
        logger.info("Connecting to Binance...")
        await exchange.connect()

        # Get market data connector
        md = exchange.get_market_data_connector()

        # Subscribe to different data types
        symbols = ["BTC/USDT", "ETH/USDT"]

        logger.info(f"Subscribing to market data for {symbols}")

        await md.subscribe_ticks(symbols, handle_tick)
        await md.subscribe_trades(symbols, handle_trade)
        await md.subscribe_order_book(symbols, handle_orderbook)

        # Let it run for a while
        logger.info("Streaming market data... (Ctrl+C to stop)")
        await asyncio.sleep(60)  # Run for 60 seconds

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        await exchange.disconnect()
        logger.info("Disconnected")


async def order_execution_example():
    """Example: Place and manage orders"""

    logger.info("=== Order Execution Example ===")

    # Get API credentials
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        logger.error(
            "API credentials required for order execution. "
            "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
        )
        return

    # Create exchange
    exchange = BinanceExchange(
        api_key=api_key,
        api_secret=api_secret,
        market_type="spot",
        testnet=True  # IMPORTANT: Use testnet for testing!
    )

    try:
        # Connect
        await exchange.connect()

        # Get execution connector
        execution = exchange.get_execution_connector()

        # Example 1: Place market order
        logger.info("\n--- Placing Market Order ---")

        market_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),  # Small test amount
            time_in_force=TimeInForce.GTC,
        )

        result = await execution.place_order(market_order)

        if result.success:
            logger.info(f"Market order placed: {result.order_id}")

            # Check order status
            await asyncio.sleep(1)
            status = await execution.get_order_status(result.order_id)
            if status:
                logger.info(f"Order status: {status.status.value}")
        else:
            logger.error(f"Market order failed: {result.message}")

        # Example 2: Place limit order
        logger.info("\n--- Placing Limit Order ---")

        # Get current price first (from market data)
        md = exchange.get_market_data_connector()

        # Wait for tick
        current_price = Decimal("50000")  # Placeholder

        limit_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            limit_price=current_price * Decimal("0.95"),  # 5% below current
            time_in_force=TimeInForce.GTC,
        )

        result = await execution.place_order(limit_order)

        if result.success:
            logger.info(
                f"Limit order placed: {result.order_id} @ {limit_order.limit_price}"
            )

            # Wait a bit
            await asyncio.sleep(2)

            # Cancel the order
            logger.info(f"Cancelling order {result.order_id}")
            cancel_result = await execution.cancel_order(result.order_id)

            if cancel_result.success:
                logger.info("Order cancelled successfully")
            else:
                logger.error(f"Cancel failed: {cancel_result.message}")
        else:
            logger.error(f"Limit order failed: {result.message}")

        # Example 3: Query open orders
        logger.info("\n--- Querying Open Orders ---")

        open_orders = await execution.get_open_orders()
        logger.info(f"Open orders: {len(open_orders)}")

        for order in open_orders:
            logger.info(
                f"  {order.order_id}: {order.side.value} {order.quantity} "
                f"{order.symbol} @ {order.limit_price or 'MARKET'}"
            )

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

    finally:
        # Cleanup
        await exchange.disconnect()
        logger.info("Disconnected")


async def combined_example():
    """Example: Market data + order execution together"""

    logger.info("=== Combined Trading Example ===")

    # Simple price monitoring with order placement
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        logger.error("API credentials required")
        return

    exchange = BinanceExchange(
        api_key=api_key,
        api_secret=api_secret,
        market_type="spot",
        testnet=True
    )

    # Track prices
    prices = {}

    async def track_price(tick: TickEvent) -> None:
        """Track latest prices"""
        prices[tick.symbol] = tick.last

        # Simple alert example
        if tick.symbol == "BTC/USDT" and tick.last:
            if tick.last > Decimal("60000"):
                logger.warning(f"BTC price above $60,000: {tick.last}")

    try:
        await exchange.connect()

        # Subscribe to price updates
        md = exchange.get_market_data_connector()
        await md.subscribe_ticks(["BTC/USDT", "ETH/USDT"], track_price)

        # Let it run
        logger.info("Monitoring prices... (Ctrl+C to stop)")
        await asyncio.sleep(30)

        logger.info(f"\nFinal prices: {prices}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        await exchange.disconnect()


async def main():
    """Main entry point"""

    logger.info("Binance Exchange Examples")
    logger.info("=" * 60)

    # Choose which example to run:

    # 1. Market data streaming (safe, read-only)
    await market_data_example()

    # 2. Order execution (requires valid API keys, uses real/testnet money!)
    # await order_execution_example()

    # 3. Combined example
    # await combined_example()


if __name__ == "__main__":
    # Set up logging
    logger.add("binance_example.log", rotation="1 day")

    asyncio.run(main())
