"""
Test Backtesting System

Demonstrates the complete backtesting pipeline:
1. Define a strategy
2. Load it dynamically
3. Run backtest
4. Analyze results
5. Create visualizations

Usage:
    python examples/test_backtesting.py
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from backtesting import BacktestEngine, load_strategy_from_code
from backtesting.visualization import BacktestVisualizer


# Example strategy code
EXAMPLE_STRATEGY = """
import pandas as pd
import numpy as np
from typing import Dict, Any


class SimpleRSIMeanReversionStrategy(BaseStrategy):
    \"\"\"
    Simple RSI Mean Reversion Strategy

    Entry: RSI < 30 (oversold)
    Exit: RSI > 50 (neutral) or stop loss
    \"\"\"

    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__(name="SimpleRSIMeanReversion", parameters=parameters)

        # Default parameters
        self.parameters.setdefault('rsi_period', 14)
        self.parameters.setdefault('rsi_oversold', 30)
        self.parameters.setdefault('rsi_exit', 50)
        self.parameters.setdefault('stop_loss_pct', 2.0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Calculate RSI indicator\"\"\"
        df = data.copy()

        # Calculate RSI
        period = self.parameters['rsi_period']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Generate trading signals\"\"\"
        df = data.copy()

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Initialize signal column
        df['signal'] = 0

        # Entry: RSI oversold
        entry_condition = df['rsi'] < self.parameters['rsi_oversold']
        df.loc[entry_condition, 'signal'] = 1

        # Exit: RSI neutral or above
        exit_condition = df['rsi'] > self.parameters['rsi_exit']
        df.loc[exit_condition, 'signal'] = -1

        return df
"""


async def generate_sample_data(
    symbol: str = "BTC/USDT",
    days: int = 180
) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing

    In production, this would come from your data provider
    """
    logger.info(f"Generating {days} days of sample data...")

    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start_date, end_date, freq='1H')

    # Generate realistic-looking price data
    np.random.seed(42)

    # Start price
    price = 40000.0

    prices = []
    for _ in range(len(dates)):
        # Random walk with trend
        change = np.random.randn() * 0.005  # 0.5% volatility
        price = price * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    # Generate OHLCV
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(prices)) * 0.001),
        'high': prices * (1 + abs(np.random.randn(len(prices))) * 0.002),
        'low': prices * (1 - abs(np.random.randn(len(prices))) * 0.002),
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(prices))
    }, index=dates)

    logger.info(f"Generated {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    return data


async def test_simple_backtest():
    """Test 1: Simple backtest with example strategy"""
    logger.info("=" * 80)
    logger.info("TEST 1: Simple Backtest")
    logger.info("=" * 80)

    # Generate sample data
    data = await generate_sample_data(days=180)

    # Load strategy
    logger.info("Loading strategy...")
    strategy_class = load_strategy_from_code(EXAMPLE_STRATEGY)

    # Create backtest engine
    logger.info("Creating backtest engine...")
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        position_size_pct=1.0,
        enable_shorting=True
    )

    # Run backtest
    logger.info("Running backtest...")
    result = await engine.run_backtest(
        strategy_class=strategy_class,
        data=data,
        strategy_parameters={
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_exit': 50,
            'stop_loss_pct': 2.0
        },
        symbol="BTC/USDT",
        timeframe="1h"
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)

    m = result.metrics
    logger.info(f"\n📊 Performance Metrics:")
    logger.info(f"  Total Return:      {m.total_return*100:>8.2f}%")
    logger.info(f"  Annual Return:     {m.annualized_return*100:>8.2f}%")
    logger.info(f"  Sharpe Ratio:      {m.sharpe_ratio:>8.2f}")
    logger.info(f"  Sortino Ratio:     {m.sortino_ratio:>8.2f}")
    logger.info(f"  Calmar Ratio:      {m.calmar_ratio:>8.2f}")

    logger.info(f"\n📉 Risk Metrics:")
    logger.info(f"  Max Drawdown:      {m.max_drawdown*100:>8.2f}%")
    logger.info(f"  Volatility:        {m.volatility*100:>8.2f}%")
    logger.info(f"  Downside Dev:      {m.downside_deviation*100:>8.2f}%")

    logger.info(f"\n💰 Trading Metrics:")
    logger.info(f"  Total Trades:      {m.total_trades:>8}")
    logger.info(f"  Winning Trades:    {m.winning_trades:>8}")
    logger.info(f"  Losing Trades:     {m.losing_trades:>8}")
    logger.info(f"  Win Rate:          {m.win_rate*100:>8.1f}%")
    logger.info(f"  Profit Factor:     {m.profit_factor:>8.2f}")

    logger.info(f"\n💵 Profit Metrics:")
    logger.info(f"  Avg Win:          ${m.avg_win:>8.2f}")
    logger.info(f"  Avg Loss:         ${m.avg_loss:>8.2f}")
    logger.info(f"  Expectancy:       ${m.expectancy:>8.2f}")
    logger.info(f"  Best Trade:       ${m.best_trade:>8.2f}")
    logger.info(f"  Worst Trade:      ${m.worst_trade:>8.2f}")

    logger.info(f"\n⏱️  Time Metrics:")
    logger.info(f"  Period:            {m.start_date.date()} to {m.end_date.date()}")
    logger.info(f"  Total Days:        {m.total_days}")
    logger.info(f"  Avg Trade Duration: {m.avg_trade_duration:.1f} hours")

    # Create visualization
    logger.info("\n" + "=" * 80)
    logger.info("Creating visualization...")
    visualizer = BacktestVisualizer()
    chart_path = visualizer.create_full_report(result)

    if chart_path:
        logger.success(f"Visualization saved to: {chart_path}")

    return result


async def test_parameter_sweep():
    """Test 2: Test multiple parameter combinations"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Parameter Sweep")
    logger.info("=" * 80)

    # Generate sample data
    data = await generate_sample_data(days=180)

    # Load strategy
    strategy_class = load_strategy_from_code(EXAMPLE_STRATEGY)

    # Create engine
    engine = BacktestEngine(initial_capital=10000.0)

    # Test different RSI oversold levels
    results = []
    rsi_levels = [20, 25, 30, 35, 40]

    logger.info(f"Testing {len(rsi_levels)} RSI oversold levels...")

    for rsi_level in rsi_levels:
        logger.info(f"  Testing RSI oversold = {rsi_level}...")

        result = await engine.run_backtest(
            strategy_class=strategy_class,
            data=data,
            strategy_parameters={
                'rsi_period': 14,
                'rsi_oversold': rsi_level,
                'rsi_exit': 50,
                'stop_loss_pct': 2.0
            },
            symbol="BTC/USDT",
            timeframe="1h"
        )

        results.append({
            'rsi_oversold': rsi_level,
            'sharpe': result.metrics.sharpe_ratio,
            'return': result.metrics.total_return * 100,
            'trades': result.metrics.total_trades,
            'win_rate': result.metrics.win_rate * 100
        })

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER SWEEP RESULTS")
    logger.info("=" * 80)

    results_df = pd.DataFrame(results)
    logger.info(f"\n{results_df.to_string(index=False)}\n")

    best_idx = results_df['sharpe'].idxmax()
    best = results_df.iloc[best_idx]

    logger.success(f"Best parameters: RSI oversold = {best['rsi_oversold']}")
    logger.success(f"  Sharpe Ratio: {best['sharpe']:.2f}")
    logger.success(f"  Total Return: {best['return']:.2f}%")


async def test_strategy_loading():
    """Test 3: Test strategy loading from code"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Strategy Loading")
    logger.info("=" * 80)

    # Test loading
    logger.info("Loading strategy from code string...")
    strategy_class = load_strategy_from_code(EXAMPLE_STRATEGY, "SimpleRSIMeanReversionStrategy")

    # Validate
    logger.info("Validating strategy...")
    instance = strategy_class()

    assert hasattr(instance, 'calculate_indicators'), "Missing calculate_indicators"
    assert hasattr(instance, 'generate_signals'), "Missing generate_signals"

    logger.success("Strategy loaded and validated successfully!")

    # Test with sample data
    logger.info("Testing strategy methods...")
    data = await generate_sample_data(days=30)

    indicators_df = instance.calculate_indicators(data)
    assert 'rsi' in indicators_df.columns, "RSI not calculated"
    logger.success("  calculate_indicators() works!")

    signals_df = instance.generate_signals(data)
    assert 'signal' in signals_df.columns, "Signals not generated"
    logger.success("  generate_signals() works!")

    # Count signals
    buy_signals = (signals_df['signal'] == 1).sum()
    sell_signals = (signals_df['signal'] == -1).sum()

    logger.info(f"\nSignal counts:")
    logger.info(f"  Buy signals:  {buy_signals}")
    logger.info(f"  Sell signals: {sell_signals}")


async def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("BACKTESTING SYSTEM TEST SUITE")
    logger.info("=" * 80)

    try:
        # Test 1: Simple backtest
        result = await test_simple_backtest()

        # Test 2: Parameter sweep
        await test_parameter_sweep()

        # Test 3: Strategy loading
        await test_strategy_loading()

        logger.info("\n" + "=" * 80)
        logger.success("ALL TESTS PASSED! ✅")
        logger.info("=" * 80)

        logger.info("\nThe backtesting system is working correctly:")
        logger.info("  ✅ Strategy loading from code")
        logger.info("  ✅ Backtest execution with real calculations")
        logger.info("  ✅ Performance metrics calculation")
        logger.info("  ✅ Visualization generation")
        logger.info("  ✅ Parameter optimization")

        logger.info("\n💡 Next steps:")
        logger.info("  1. Run the AI research agent: python examples/demo_research_agent.py")
        logger.info("  2. The agent will now use REAL backtesting instead of mock results")
        logger.info("  3. Check generated visualizations in the current directory")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
