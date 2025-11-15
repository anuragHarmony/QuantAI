"""
Run AI Agent on Market Data

Runs the autonomous AI research agent on the downloaded/generated market data
to discover profitable trading strategies.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import pandas as pd
from datetime import datetime
from loguru import logger
import os

from research.agent.agent_factory import AgentFactory
from research.agent.core.orchestrator import ResearchConfig


async def load_market_data(data_dir: Path) -> dict:
    """Load all market data from CSV files"""
    logger.info("Loading market data...")

    data_files = list(data_dir.glob("*.csv"))

    if not data_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    datasets = {}

    for file in data_files:
        # Parse filename: BTCUSDT_1h_20240101_to_20240701.csv
        parts = file.stem.split("_")
        symbol = parts[0]  # BTCUSDT
        timeframe = parts[1]  # 1h

        # Load data
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        # Store
        key = f"{symbol}_{timeframe}"
        datasets[key] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': df,
            'file': file
        }

    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


async def run_agent_research(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    iterations: int = 3
):
    """Run the AI research agent"""

    logger.info("=" * 80)
    logger.info("AUTONOMOUS AI RESEARCH AGENT")
    logger.info("=" * 80)
    logger.info(f"Target Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Max Iterations: {iterations}")
    logger.info("=" * 80)

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("\n" + "=" * 80)
        logger.error("ERROR: ANTHROPIC_API_KEY not set")
        logger.error("=" * 80)
        logger.error("\nPlease set your Anthropic API key:")
        logger.error("  export ANTHROPIC_API_KEY=your_api_key_here")
        logger.error("\nOr for this session only:")
        logger.error("  ANTHROPIC_API_KEY=your_key python scripts/run_agent_on_data.py")
        logger.error("=" * 80)
        return

    # Load market data
    data_dir = Path("/home/user/QuantAI/data/market_data")
    datasets = await load_market_data(data_dir)

    # Find the requested dataset
    key = f"{symbol}_{timeframe}"
    if key not in datasets:
        logger.error(f"Dataset {key} not found!")
        logger.info(f"\nAvailable datasets:")
        for k in sorted(datasets.keys()):
            logger.info(f"  - {k}")
        return

    dataset = datasets[key]
    logger.info(f"\nUsing dataset: {dataset['file'].name}")
    logger.info(f"  Candles: {len(dataset['data'])}")
    logger.info(f"  Date range: {dataset['data'].index[0]} to {dataset['data'].index[-1]}")
    logger.info(f"  Price range: ${dataset['data']['close'].min():.2f} - ${dataset['data']['close'].max():.2f}")

    # Create research config
    # Use first 60% for backtesting (allowing the agent to find patterns)
    data_len = len(dataset['data'])
    backtest_end_idx = int(data_len * 0.6)
    backtest_data = dataset['data'].iloc[:backtest_end_idx]

    config = ResearchConfig(
        max_iterations=iterations,
        min_sharpe_threshold=1.0,  # Lower threshold since this is synthetic data
        diversity_threshold=0.7,
        backtest_symbol=f"{symbol[:3]}/USDT",  # BTC/USDT format
        backtest_timeframe=timeframe,
        backtest_start_date=str(backtest_data.index[0]),
        backtest_end_date=str(backtest_data.index[-1]),
        learning_interval=2  # Learn every 2 experiments
    )

    logger.info(f"\nBacktest period: {backtest_data.index[0]} to {backtest_data.index[-1]}")
    logger.info(f"Backtest candles: {len(backtest_data)}")

    # Create AI agent
    logger.info("\nInitializing AI research agent...")

    try:
        agent = await AgentFactory.create_research_agent(
            anthropic_api_key=api_key,
            config=config
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        return

    # Run autonomous research
    logger.info("\n" + "=" * 80)
    logger.info("STARTING AUTONOMOUS RESEARCH")
    logger.info("=" * 80)
    logger.info("\nThe agent will autonomously:")
    logger.info("  1. Analyze market data and detect patterns")
    logger.info("  2. Generate strategy hypotheses using AI")
    logger.info("  3. Convert hypotheses to executable Python code")
    logger.info("  4. Run real backtests with accurate metrics")
    logger.info("  5. Reflect on results and learn")
    logger.info("  6. Iterate to discover profitable strategies")
    logger.info("\nThis may take several minutes...")
    logger.info("=" * 80)

    try:
        session = await agent.run_research_loop(
            objective=f"Discover profitable trading strategies for {symbol} on {timeframe} timeframe",
            focus="momentum and mean reversion strategies",
            max_iterations=iterations
        )

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("RESEARCH COMPLETED")
        logger.info("=" * 80)
        logger.info(f"\nSession ID: {session.id}")
        logger.info(f"Duration: {(session.end_time - session.start_time).total_seconds() / 60:.1f} minutes")
        logger.info(f"Experiments run: {session.experiments_run}")
        logger.info(f"Successful experiments: {session.successful_experiments}")
        logger.info(f"Success rate: {session.successful_experiments / max(session.experiments_run, 1) * 100:.1f}%")
        logger.info(f"Best Sharpe ratio: {session.best_sharpe:.2f}")

        # Get best strategies
        logger.info("\n" + "=" * 80)
        logger.info("BEST STRATEGIES DISCOVERED")
        logger.info("=" * 80)

        best_strategies = await agent.get_best_strategies(top_k=3, min_sharpe=0.5)

        if best_strategies:
            for i, strategy in enumerate(best_strategies, 1):
                logger.info(f"\n### Strategy {i}: {strategy.hypothesis.get('description', 'N/A')}")
                logger.info(f"  Rationale: {strategy.hypothesis.get('rationale', 'N/A')[:100]}...")
                logger.info(f"\n  Performance:")
                logger.info(f"    Sharpe Ratio:    {strategy.results.get('sharpe_ratio', 'N/A'):>8.2f}")
                logger.info(f"    Total Return:    {strategy.results.get('total_return', 'N/A'):>8.1f}%")
                logger.info(f"    Win Rate:        {strategy.results.get('win_rate', 0)*100:>8.1f}%")
                logger.info(f"    Total Trades:    {strategy.results.get('total_trades', 'N/A'):>8}")
                logger.info(f"    Max Drawdown:    {strategy.results.get('max_drawdown', 'N/A'):>8.1f}%")
                logger.info(f"    Profit Factor:   {strategy.results.get('profit_factor', 'N/A'):>8.2f}")
                logger.info(f"\n  Strategy Details:")
                logger.info(f"    Indicators: {', '.join(strategy.hypothesis.get('indicators', []))}")
                logger.info(f"    Entry: {strategy.hypothesis.get('entry_logic', 'N/A')}")
                logger.info(f"    Exit: {strategy.hypothesis.get('exit_logic', 'N/A')}")

        else:
            logger.warning("\nNo profitable strategies found. Try:")
            logger.warning("  - Running more iterations")
            logger.warning("  - Using different market data")
            logger.warning("  - Adjusting the min_sharpe_threshold")

        # Generate research report
        logger.info("\n" + "=" * 80)
        logger.info("Generating research report...")

        report = await agent.generate_research_report(session)
        report_path = f"research_report_{symbol}_{timeframe}_{session.id}.md"

        with open(report_path, 'w') as f:
            f.write(report)

        logger.success(f"Research report saved to: {report_path}")

        # Display insights
        if session.insights_learned:
            logger.info("\n" + "=" * 80)
            logger.info("INSIGHTS LEARNED")
            logger.info("=" * 80)

            for i, insight in enumerate(session.insights_learned[:5], 1):
                logger.info(f"{i}. {insight}")

        logger.info("\n" + "=" * 80)
        logger.success("✓ AUTONOMOUS RESEARCH COMPLETE!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nResearch failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point"""

    # Default to BTC/USDT 1h
    # You can modify these to test different coins/timeframes
    await run_agent_research(
        symbol="BTCUSDT",
        timeframe="1h",
        iterations=3  # Start with 3 iterations (can increase later)
    )


if __name__ == "__main__":
    asyncio.run(main())
