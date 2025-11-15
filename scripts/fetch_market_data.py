"""
Fetch Real Market Data from Binance

Downloads historical OHLCV data for top 10 coins and saves it locally.
This data will be used by the AI agent to discover profitable strategies.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from research.data.binance_provider import BinanceDataProvider


async def fetch_and_save_data():
    """Fetch historical data for top 10 coins"""

    # Top 10 coins by market cap
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "ADA/USDT",
        "XRP/USDT",
        "DOGE/USDT",
        "MATIC/USDT",
        "DOT/USDT",
        "AVAX/USDT",
    ]

    # Timeframes to fetch
    timeframes = ["1h", "4h"]

    # Date range (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    logger.info("=" * 80)
    logger.info("FETCHING REAL MARKET DATA FROM BINANCE")
    logger.info("=" * 80)
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total datasets: {len(symbols) * len(timeframes)}")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path("/home/user/QuantAI/data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data provider
    provider = BinanceDataProvider()

    successful = 0
    failed = 0

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logger.info(f"\nFetching {symbol} {timeframe}...")

                # Fetch data
                market_data = await provider.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )

                # Save to CSV
                symbol_clean = symbol.replace("/", "")
                filename = f"{symbol_clean}_{timeframe}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
                filepath = output_dir / filename

                market_data.data.to_csv(filepath)

                logger.success(
                    f"✓ Saved {len(market_data.data)} candles to {filename}"
                )
                successful += 1

                # Brief pause to avoid rate limits
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"✗ Failed to fetch {symbol} {timeframe}: {e}")
                failed += 1

    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # List all downloaded files
    logger.info("\nDownloaded files:")
    csv_files = list(output_dir.glob("*.csv"))
    for i, file in enumerate(csv_files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"  {i}. {file.name} ({size_mb:.2f} MB)")

    logger.success(f"\n✓ Downloaded {len(csv_files)} datasets!")

    return csv_files


async def main():
    """Main entry point"""
    csv_files = await fetch_and_save_data()

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("The market data has been downloaded successfully!")
    logger.info("\nTo run the AI agent on this data:")
    logger.info("  1. The data is stored in: /home/user/QuantAI/data/market_data/")
    logger.info("  2. Run the AI agent with: python scripts/run_agent_on_data.py")
    logger.info("  3. The agent will discover strategies on this real market data")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
