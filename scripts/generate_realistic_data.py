"""
Generate Realistic Market Data

Creates synthetic but realistic market data for testing the AI agent.
Data includes realistic price movements, volatility patterns, and market regimes.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


def generate_realistic_ohlcv(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1h",
    initial_price: float = 40000.0
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data with:
    - Trending periods
    - Ranging periods
    - Volatility clustering
    - Realistic spreads and wicks
    """

    # Calculate number of candles
    if timeframe == "1h":
        delta = timedelta(hours=1)
    elif timeframe == "4h":
        delta = timedelta(hours=4)
    else:
        delta = timedelta(hours=1)

    dates = pd.date_range(start_date, end_date, freq=delta)
    n_candles = len(dates)

    logger.info(f"  Generating {n_candles} candles for {symbol} {timeframe}")

    # Generate price with multiple components
    np.random.seed(hash(symbol) % (2**32))  # Consistent seed per symbol

    # 1. Base trend (cycles between up/down/sideways)
    trend_length = n_candles // 10  # Trend changes every ~10% of data
    n_trends = max(1, n_candles // trend_length)

    trend = np.zeros(n_candles)
    for i in range(n_trends):
        start_idx = i * trend_length
        end_idx = min((i + 1) * trend_length, n_candles)

        # Random trend direction (-1, 0, 1)
        direction = np.random.choice([-1, 0, 1], p=[0.25, 0.3, 0.45])  # Slight bull bias
        strength = np.random.uniform(0.0001, 0.0003) # Daily drift

        trend[start_idx:end_idx] = direction * strength

    # 2. Volatility clustering (GARCH-like)
    volatility = np.zeros(n_candles)
    volatility[0] = 0.01  # 1% initial vol

    for i in range(1, n_candles):
        # GARCH(1,1) inspired
        volatility[i] = (0.05 * 0.01 +
                        0.9 * volatility[i-1] +
                        0.05 * abs(np.random.randn()) * 0.01)

    volatility = np.clip(volatility, 0.005, 0.05)  # 0.5% to 5% vol

    # 3. Random walk with drift and vol clustering
    returns = trend + volatility * np.random.randn(n_candles)

    # Generate close prices
    price = initial_price
    close_prices = []

    for ret in returns:
        price = price * (1 + ret)
        close_prices.append(price)

    close_prices = np.array(close_prices)

    # 4. Generate OHLC from close
    # Open is previous close (shifted)
    open_prices = np.concatenate([[initial_price], close_prices[:-1]])

    # High/Low with realistic wicks
    wick_size = volatility * np.abs(np.random.randn(n_candles)) * 0.5
    high_prices = np.maximum(open_prices, close_prices) * (1 + wick_size)
    low_prices = np.minimum(open_prices, close_prices) * (1 - wick_size)

    # Ensure OHLC validity
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # 5. Generate volume (correlated with volatility)
    base_volume = 1000
    volume = base_volume * (1 + 2 * volatility) * (1 + 0.5 * np.abs(np.random.randn(n_candles)))

    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    return df


def main():
    """Generate data for top 10 coins"""

    logger.info("=" * 80)
    logger.info("GENERATING REALISTIC MARKET DATA")
    logger.info("=" * 80)

    # Symbol configurations (symbol, initial_price)
    symbols_config = [
        ("BTC/USDT", 40000.0),
        ("ETH/USDT", 2500.0),
        ("BNB/USDT", 350.0),
        ("SOL/USDT", 80.0),
        ("ADA/USDT", 0.50),
        ("XRP/USDT", 0.60),
        ("DOGE/USDT", 0.10),
        ("MATIC/USDT", 0.80),
        ("DOT/USDT", 6.0),
        ("AVAX/USDT", 30.0),
    ]

    timeframes = ["1h", "4h"]

    # Date range (6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    logger.info(f"Symbols: {len(symbols_config)}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total datasets: {len(symbols_config) * len(timeframes)}")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path("/home/user/QuantAI/data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for symbol, initial_price in symbols_config:
        logger.info(f"\nGenerating {symbol}...")

        for timeframe in timeframes:
            # Generate data
            df = generate_realistic_ohlcv(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                initial_price=initial_price
            )

            # Save to CSV
            symbol_clean = symbol.replace("/", "")
            filename = f"{symbol_clean}_{timeframe}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
            filepath = output_dir / filename

            df.to_csv(filepath)

            logger.success(
                f"  ✓ Saved {len(df)} candles to {filename} "
                f"(${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f})"
            )

            count += 1

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Generated: {count} datasets")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Show statistics
    logger.info("\nDataset Statistics:")
    csv_files = list(output_dir.glob("*.csv"))

    for file in csv_files[:5]:  # Show first 5
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        returns = df['close'].pct_change().dropna()

        logger.info(f"\n  {file.name}:")
        logger.info(f"    Candles: {len(df)}")
        logger.info(f"    Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        logger.info(f"    Avg return: {returns.mean()*100:.4f}%")
        logger.info(f"    Volatility: {returns.std()*100:.2f}%")

    logger.success(f"\n✓ Generated {len(csv_files)} realistic market datasets!")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("The realistic market data has been generated!")
    logger.info("\nTo run the AI agent on this data:")
    logger.info("  python scripts/run_agent_on_data.py")
    logger.info("\nThe agent will:")
    logger.info("  1. Load the market data")
    logger.info("  2. Analyze patterns and regimes")
    logger.info("  3. Generate strategy hypotheses")
    logger.info("  4. Run real backtests")
    logger.info("  5. Learn and iterate")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
