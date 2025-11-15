"""
Manual AI Agent Research Session

This script demonstrates the full AI agent workflow by manually executing
each phase that the autonomous agent would perform:

1. KNOWLEDGE RETRIEVAL - Query knowledge graph for trading concepts
2. DATA ANALYTICS - Analyze market data for patterns and regimes
3. HYPOTHESIS GENERATION - Create strategy based on knowledge + patterns
4. CODE GENERATION - Write executable strategy code
5. BACKTESTING - Run real backtest with accurate metrics
6. REFLECTION - Analyze results and extract insights

This verifies all agent components work together.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Import all agent components
from research.knowledge.memory_graph import InMemoryKnowledgeGraph
from research.knowledge.integrated_retriever import IntegratedKnowledgeRetriever
from research.data.binance_provider import BinanceDataProvider
from research.data.feature_engineer import TechnicalFeatureEngineer
from research.data.pattern_detector import SimplePatternDetector
from research.agent.memory.simple_memory import SimpleEpisodicMemory, SimpleSemanticMemory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from the module files to avoid directory conflicts
from backtesting import BacktestEngine, load_strategy_from_code


async def phase1_knowledge_retrieval():
    """Phase 1: Query Knowledge Graph"""
    logger.info("=" * 80)
    logger.info("PHASE 1: KNOWLEDGE RETRIEVAL")
    logger.info("=" * 80)

    # Initialize knowledge graph
    logger.info("Initializing knowledge graph...")
    kg = InMemoryKnowledgeGraph()
    await kg.seed_with_defaults()

    # Query for mean reversion concepts
    logger.info("\nQuerying for 'mean reversion' concepts...")
    concepts = await kg.query_concepts(filters={'name_contains': 'reversion'})

    logger.info(f"\nFound {len(concepts)} related concepts:")
    for concept in concepts:
        logger.info(f"  - {concept.name} ({concept.type.value})")
        logger.info(f"    Description: {concept.description}")

        # Get related concepts
        related = await kg.find_related_concepts(concept.id)
        if related:
            logger.info(f"    Related to: {', '.join([c.name for c in related[:3]])}")

    # Query for indicators
    logger.info("\nQuerying for indicator concepts...")
    indicators = await kg.query_concepts(filters={'type': 'indicator'})

    logger.info(f"\nFound {len(indicators)} indicators:")
    for ind in indicators[:5]:
        logger.info(f"  - {ind.name}: {ind.description}")

    logger.success("\n✓ Knowledge retrieval complete!")

    return kg, concepts, indicators


async def phase2_data_analytics():
    """Phase 2: Analyze Market Data"""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: DATA ANALYTICS")
    logger.info("=" * 80)

    # Load market data
    logger.info("Loading BTC/USDT 1h data...")
    df = pd.read_csv('/home/user/QuantAI/data/market_data/BTCUSDT_1h_20250519_to_20251115.csv',
                     index_col=0, parse_dates=True)

    logger.info(f"  Loaded {len(df)} candles")
    logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Initialize analytics tools
    feature_engineer = TechnicalFeatureEngineer()
    pattern_detector = SimplePatternDetector()

    # Calculate technical indicators
    logger.info("\nCalculating technical indicators...")
    df_with_indicators = await feature_engineer.calculate_technical_indicators(
        df,
        indicators=['rsi', 'bollinger', 'atr', 'sma', 'ema']
    )

    logger.info("  ✓ Calculated: RSI, Bollinger Bands, ATR, SMA, EMA")
    logger.info(f"    Current RSI: {df_with_indicators['rsi_14'].iloc[-1]:.2f}")

    # Calculate BB position manually
    df_with_indicators['bb_position'] = ((df_with_indicators['close'] - df_with_indicators['bb_lower_20']) /
                                          (df_with_indicators['bb_upper_20'] - df_with_indicators['bb_lower_20']) * 2 - 1)
    logger.info(f"    Current BB position: {df_with_indicators['bb_position'].iloc[-1]:.2f}")

    # Detect patterns
    logger.info("\nDetecting market patterns...")
    trends = await pattern_detector.detect_trends(df)

    logger.info(f"  Found {len(trends)} trend patterns:")
    for trend in trends[:3]:
        logger.info(f"    - {trend.pattern_type}: {trend.properties}")

    # Detect market regime
    logger.info("\nDetecting market regime...")
    regimes = await pattern_detector.detect_regimes(df)

    if regimes:
        regime = regimes[0]
        logger.info(f"  Current regime: {regime.regime_type}")
        logger.info(f"  Confidence: {regime.confidence:.2f}")
        logger.info(f"  Characteristics: {regime.characteristics}")

    # Detect support/resistance
    logger.info("\nDetecting support/resistance levels...")
    levels = await pattern_detector.detect_support_resistance(df)

    if levels:
        support = [l for l in levels if l.pattern_type == 'support']
        resistance = [l for l in levels if l.pattern_type == 'resistance']

        logger.info(f"  Found {len(support)} support levels")
        logger.info(f"  Found {len(resistance)} resistance levels")

        if support:
            logger.info(f"    Strongest support: ${support[0].properties['level']:.2f}")
        if resistance:
            logger.info(f"    Strongest resistance: ${resistance[0].properties['level']:.2f}")

    logger.success("\n✓ Data analytics complete!")

    return df_with_indicators, regimes, trends, levels


async def phase3_hypothesis_generation(kg, df_with_indicators, regimes):
    """Phase 3: Generate Strategy Hypothesis"""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: HYPOTHESIS GENERATION")
    logger.info("=" * 80)

    # Analyze data for opportunities
    logger.info("Analyzing data for mean reversion opportunities...")

    # Check RSI oversold conditions
    oversold = df_with_indicators['rsi_14'] < 35
    at_lower_bb = df_with_indicators['bb_position'] < -0.8

    mean_rev_signals = oversold & at_lower_bb
    signal_count = mean_rev_signals.sum()

    logger.info(f"  Found {signal_count} potential mean reversion signals")
    logger.info(f"    RSI < 35: {oversold.sum()} times")
    logger.info(f"    Price at lower BB: {at_lower_bb.sum()} times")

    # Analyze forward returns
    returns_after_signal = []
    for idx in df_with_indicators[mean_rev_signals].index[:100]:  # Sample 100
        pos = df_with_indicators.index.get_loc(idx)
        if pos < len(df_with_indicators) - 20:
            future_ret = (df_with_indicators['close'].iloc[pos+20] - df_with_indicators['close'].iloc[pos]) / df_with_indicators['close'].iloc[pos]
            returns_after_signal.append(future_ret * 100)

    if returns_after_signal:
        avg_return = np.mean(returns_after_signal)
        win_rate = (np.array(returns_after_signal) > 0).sum() / len(returns_after_signal)

        logger.info(f"\n  Forward returns analysis:")
        logger.info(f"    Avg 20-candle return: {avg_return:.2f}%")
        logger.info(f"    Win rate: {win_rate*100:.1f}%")

    # Generate hypothesis
    hypothesis = {
        "description": "BTC Mean Reversion using RSI and Bollinger Bands",
        "rationale": f"Data analysis shows {signal_count} oversold conditions (RSI<35 + price at lower BB) "
                    f"with {win_rate*100:.1f}% win rate and {avg_return:.2f}% avg return over 20 candles. "
                    f"Mean reversion works in {regimes[0].regime_type if regimes else 'various'} market conditions.",
        "indicators": ["RSI", "Bollinger Bands", "ATR"],
        "entry_logic": "Price <= Lower Bollinger Band AND RSI < 35 (oversold)",
        "exit_logic": "RSI > 50 (return to neutral) OR stop loss triggered",
        "risk_management": {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 5.0,
            "position_size_pct": 100.0
        },
        "target_regime": regimes[0].regime_type if regimes else "any",
        "expected_sharpe": 1.5,
        "expected_win_rate": win_rate
    }

    logger.info("\n📋 STRATEGY HYPOTHESIS GENERATED:")
    logger.info(f"  Description: {hypothesis['description']}")
    logger.info(f"  Rationale: {hypothesis['rationale']}")
    logger.info(f"  Indicators: {', '.join(hypothesis['indicators'])}")
    logger.info(f"  Entry: {hypothesis['entry_logic']}")
    logger.info(f"  Exit: {hypothesis['exit_logic']}")
    logger.info(f"  Expected Sharpe: {hypothesis['expected_sharpe']}")
    logger.info(f"  Expected Win Rate: {hypothesis['expected_win_rate']*100:.1f}%")

    logger.success("\n✓ Hypothesis generation complete!")

    return hypothesis


async def phase4_code_generation(hypothesis):
    """Phase 4: Generate Strategy Code"""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: CODE GENERATION")
    logger.info("=" * 80)

    logger.info("Generating executable Python code from hypothesis...")

    # Generate strategy code
    code = f'''import pandas as pd
import numpy as np
from typing import Dict, Any


class BTCMeanReversionStrategy:
    """
    {hypothesis["description"]}

    Rationale: {hypothesis["rationale"]}

    Entry: {hypothesis["entry_logic"]}
    Exit: {hypothesis["exit_logic"]}
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        self.name = "BTCMeanReversion"
        self.parameters = parameters or {{}}

        self.parameters.setdefault('bb_period', 20)
        self.parameters.setdefault('bb_std', 2.0)
        self.parameters.setdefault('rsi_period', 14)
        self.parameters.setdefault('rsi_entry', 35)
        self.parameters.setdefault('rsi_exit', 50)
        self.parameters.setdefault('stop_loss_pct', {hypothesis["risk_management"]["stop_loss_pct"]})
        self.parameters.setdefault('take_profit_pct', {hypothesis["risk_management"]["take_profit_pct"]})

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and RSI"""
        df = data.copy()

        # Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']

        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        df['bb_std'] = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)

        # RSI
        rsi_period = self.parameters['rsi_period']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(data)

        # Initialize signal column
        df['signal'] = 0

        # Entry: Price at/below lower BB AND RSI oversold
        entry_condition = (df['close'] <= df['bb_lower']) & (df['rsi'] < self.parameters['rsi_entry'])

        # Exit: RSI returns to neutral
        exit_condition = df['rsi'] > self.parameters['rsi_exit']

        # Set signals
        df.loc[entry_condition, 'signal'] = 1   # Buy
        df.loc[exit_condition, 'signal'] = -1   # Sell

        return df
'''

    logger.info("✓ Code generated successfully!")
    logger.info(f"  Class name: BTCMeanReversionStrategy")
    logger.info(f"  Lines of code: {len(code.split(chr(10)))}")

    # Validate code
    logger.info("\nValidating code syntax...")
    try:
        compile(code, '<string>', 'exec')
        logger.success("  ✓ Code syntax valid!")
    except SyntaxError as e:
        logger.error(f"  ✗ Syntax error: {e}")
        return None

    logger.success("\n✓ Code generation complete!")

    return code


async def phase5_backtesting(code, df):
    """Phase 5: Run Backtest"""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: BACKTESTING")
    logger.info("=" * 80)

    logger.info("Loading strategy from generated code...")

    # Load strategy class
    strategy_class = load_strategy_from_code(code, "BTCMeanReversionStrategy")
    logger.success("  ✓ Strategy loaded successfully!")

    # Initialize backtest engine
    logger.info("\nInitializing backtest engine...")
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.001,  # 0.1%
        slippage=0.0005,   # 0.05%
        position_size_pct=1.0,
        enable_shorting=True
    )

    # Run backtest
    logger.info(f"\nRunning backtest on {len(df)} candles...")
    logger.info(f"  Period: {df.index[0]} to {df.index[-1]}")

    result = await engine.run_backtest(
        strategy_class=strategy_class,
        data=df,
        strategy_parameters={
            'rsi_entry': 35,
            'rsi_exit': 50,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 5.0
        },
        symbol="BTC/USDT",
        timeframe="1h"
    )

    logger.success("\n✓ Backtest complete!")

    return result


async def phase6_reflection(result, hypothesis):
    """Phase 6: Reflect on Results"""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: REFLECTION & ANALYSIS")
    logger.info("=" * 80)

    m = result.metrics

    # Determine if successful
    is_successful = m.sharpe_ratio > 1.0 and m.total_trades >= 30

    logger.info(f"\nStrategy Assessment: {'✓ SUCCESSFUL' if is_successful else '✗ FAILED'}")

    logger.info(f"\n📊 Performance Metrics:")
    logger.info(f"  Total Return:      {m.total_return*100:>8.2f}%")
    logger.info(f"  Annual Return:     {m.annualized_return*100:>8.2f}%")
    logger.info(f"  Sharpe Ratio:      {m.sharpe_ratio:>8.2f}")
    logger.info(f"  Sortino Ratio:     {m.sortino_ratio:>8.2f}")
    logger.info(f"  Calmar Ratio:      {m.calmar_ratio:>8.2f}")

    logger.info(f"\n📉 Risk Metrics:")
    logger.info(f"  Max Drawdown:      {m.max_drawdown*100:>8.2f}%")
    logger.info(f"  Volatility:        {m.volatility*100:>8.2f}%")

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

    # Extract insights
    logger.info(f"\n💡 Insights Learned:")

    insights = []

    if is_successful:
        insights.append(f"✓ Mean reversion works well with Sharpe {m.sharpe_ratio:.2f}")
        insights.append(f"✓ RSI + Bollinger Bands is an effective combination")
        insights.append(f"✓ Win rate of {m.win_rate*100:.1f}% validates the approach")

        if m.profit_factor > 2.0:
            insights.append(f"✓ Excellent profit factor ({m.profit_factor:.2f}) - winners >> losers")

        if m.max_drawdown < 0.15:
            insights.append(f"✓ Low drawdown ({m.max_drawdown*100:.1f}%) indicates good risk management")
    else:
        if m.sharpe_ratio < 1.0:
            insights.append(f"✗ Sharpe ratio too low ({m.sharpe_ratio:.2f}) - poor risk-adjusted returns")

        if m.total_trades < 30:
            insights.append(f"✗ Insufficient trades ({m.total_trades}) for statistical significance")

        if m.win_rate < 0.5:
            insights.append(f"✗ Win rate below 50% ({m.win_rate*100:.1f}%) - strategy needs improvement")

    for i, insight in enumerate(insights, 1):
        logger.info(f"  {i}. {insight}")

    # Recommendations
    logger.info(f"\n📝 Recommendations:")

    if is_successful:
        logger.info("  1. Consider testing on other cryptocurrencies (ETH, BNB, SOL)")
        logger.info("  2. Try different timeframes (4h, 1d) for longer-term positions")
        logger.info("  3. Experiment with dynamic stop losses based on ATR")
        logger.info("  4. Add trend filter to avoid mean reversion in strong trends")
    else:
        logger.info("  1. Adjust RSI entry threshold (try 30 instead of 35)")
        logger.info("  2. Add additional filters to reduce false signals")
        logger.info("  3. Consider combining with trend indicators")
        logger.info("  4. Test on different market regimes")

    logger.success("\n✓ Reflection complete!")

    return insights


async def main():
    """Main research session"""
    logger.info("\n" + "=" * 80)
    logger.info("MANUAL AI AGENT RESEARCH SESSION")
    logger.info("Demonstrating Full Agent Workflow")
    logger.info("=" * 80)

    try:
        # Phase 1: Knowledge Retrieval
        kg, concepts, indicators = await phase1_knowledge_retrieval()

        # Phase 2: Data Analytics
        df, regimes, trends, levels = await phase2_data_analytics()

        # Phase 3: Hypothesis Generation
        hypothesis = await phase3_hypothesis_generation(kg, df, regimes)

        # Phase 4: Code Generation
        code = await phase4_code_generation(hypothesis)

        if code is None:
            logger.error("Code generation failed!")
            return

        # Phase 5: Backtesting
        result = await phase5_backtesting(code, df)

        # Phase 6: Reflection
        insights = await phase6_reflection(result, hypothesis)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("RESEARCH SESSION COMPLETE")
        logger.info("=" * 80)

        logger.success("\n✓ All agent components verified:")
        logger.success("  ✓ Knowledge Graph - queried for concepts")
        logger.success("  ✓ Data Analytics - calculated indicators, detected patterns")
        logger.success("  ✓ Hypothesis Generation - created strategy from analysis")
        logger.success("  ✓ Code Generation - converted hypothesis to Python")
        logger.success("  ✓ Backtesting - ran real simulation with accurate metrics")
        logger.success("  ✓ Reflection - analyzed results and extracted insights")

        logger.info(f"\nFinal Result:")
        logger.info(f"  Strategy: {hypothesis['description']}")
        logger.info(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        logger.info(f"  Total Return: {result.metrics.total_return*100:.2f}%")
        logger.info(f"  Win Rate: {result.metrics.win_rate*100:.1f}%")
        logger.info(f"  Total Trades: {result.metrics.total_trades}")

        if result.metrics.sharpe_ratio > 1.0:
            logger.success(f"\n🎉 PROFITABLE STRATEGY DISCOVERED!")
        else:
            logger.warning(f"\n⚠️  Strategy needs improvement - continue iterating")

        logger.info("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"\nResearch session failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
