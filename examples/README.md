# Examples

This directory contains example scripts demonstrating how to use the QuantAI research agent.

## Autonomous AI Research Agent

The AI research agent autonomously discovers trading strategies by:
1. Querying knowledge sources (knowledge graph, RAG, past experiments)
2. Generating strategy hypotheses using LLM reasoning
3. Converting hypotheses to executable Python code
4. Running backtests to evaluate strategies
5. Reflecting on results and extracting insights
6. Iterating to find better strategies

### Quick Start

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run the demo
python examples/demo_research_agent.py
```

### Demo Options

The demo script includes several examples:

#### 1. Simple Research (Recommended for first run)
Runs 5 research iterations with default configuration:
```python
python examples/demo_research_agent.py
# Select option 1
```

#### 2. Advanced Research
Runs 10 iterations with custom configuration:
```python
# Select option 2
```

#### 3. Hypothesis Generation Only
Generate strategy hypotheses without running backtests:
```python
# Select option 3
```

#### 4. Code Generation Demo
Generate executable Python code from a hypothesis:
```python
# Select option 4
```

### Using the Agent Programmatically

```python
import asyncio
from research.agent.agent_factory import AgentFactory
from research.agent.core.orchestrator import ResearchConfig

async def run_research():
    # Create agent with custom config
    config = ResearchConfig(
        max_iterations=10,
        min_sharpe_threshold=1.5,
        backtest_symbol="BTC/USDT",
        backtest_timeframe="1h"
    )

    agent = await AgentFactory.create_research_agent(config=config)

    # Run autonomous research
    session = await agent.run_research_loop(
        objective="Discover profitable momentum strategies",
        focus="momentum",
        max_iterations=10
    )

    # Get best strategies
    best = await agent.get_best_strategies(top_k=5)

    for strategy in best:
        print(f"Strategy: {strategy.hypothesis['description']}")
        print(f"Sharpe: {strategy.results['sharpe_ratio']:.2f}")

    # Generate report
    report = await agent.generate_research_report(session)
    print(report)

asyncio.run(run_research())
```

### Configuration Options

```python
ResearchConfig(
    max_iterations=10,              # Maximum research iterations
    min_sharpe_threshold=1.5,       # Minimum Sharpe for success
    diversity_threshold=0.7,        # Hypothesis diversity (0-1)
    parallel_experiments=1,         # Parallel experiments (not yet implemented)
    backtest_symbol="BTC/USDT",    # Symbol to backtest
    backtest_timeframe="1h",        # Timeframe
    backtest_start_date="2024-01-01",
    backtest_end_date="2024-06-01",
    learning_interval=3             # Extract insights every N experiments
)
```

### Understanding the Output

The agent will log its progress through each phase:

```
================================================================================
ITERATION 1/10
================================================================================

Phase 1: Generating hypothesis...
Generated: RSI mean reversion strategy with Bollinger Bands filter

Phase 2: Generating code...
Generated valid code: RSIMeanReversionBollingerStrategy

Phase 3: Running backtest...
Backtest complete. Sharpe: 1.85

Phase 4: Reflecting on results...
Analysis complete. Success: True

Progress: 1/10
Experiments run: 1
Successful: 1
Best Sharpe: 1.85
```

At the end, you'll get a summary:

```
================================================================================
RESEARCH SESSION COMPLETED
================================================================================

Objective: Discover profitable momentum strategies
Duration: 12.3 minutes
Experiments run: 10
Successful experiments: 4 (40.0%)
Best Sharpe ratio: 2.15
Best strategy ID: abc123def456

Insights learned (5):
1. RSI works well in ranging markets when combined with Bollinger Bands
2. Momentum strategies perform better with shorter timeframes (1h vs 4h)
3. Stop losses should be adaptive based on volatility (ATR-based)
4. Mean reversion strategies need >= 30 trades for statistical significance
5. Combining trend and momentum indicators reduces false signals
```

### Research Report

The agent generates a comprehensive markdown report saved to `research_report_{session_id}.md`:

```markdown
# Research Report: Discover profitable momentum strategies

## Session Summary
- **Session ID**: abc123def456
- **Duration**: 12.3 minutes
- **Experiments Run**: 10
- **Successful Experiments**: 4 (40.0%)
- **Best Sharpe Ratio**: 2.15

## Top Strategies

### 1. RSI Mean Reversion with Bollinger Bands
**Performance**:
- Sharpe Ratio: 2.15
- Total Return: 45.2%
- Win Rate: 58.5%
- Total Trades: 87

**Strategy**:
- Indicators: RSI, Bollinger Bands, ATR
- Entry: RSI < 30 and price touches lower Bollinger Band
- Exit: RSI > 50 or price reaches upper Bollinger Band

## Insights Learned
1. **[indicator]** RSI works well in ranging markets when combined with Bollinger Bands
   - Confidence: 0.85

## Next Steps
1. Explore adaptive stop losses based on market volatility
   - Rationale: Fixed stop losses performed poorly in volatile markets
   - Priority: high
```

### Notes

- **Mock Backtesting**: The demo uses mock backtest results for demonstration. In production, integrate with your backtesting framework.
- **API Costs**: Each LLM call costs money. The simple demo (5 iterations) makes ~15-20 API calls.
- **Time**: Each iteration takes 10-30 seconds depending on LLM response time.

### Troubleshooting

**"ANTHROPIC_API_KEY environment variable not set"**
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

**Import errors**
Make sure you're in the project root directory:
```bash
cd /path/to/QuantAI
python examples/demo_research_agent.py
```

**Install dependencies**
```bash
pip install anthropic loguru pandas numpy ccxt scipy networkx
```

### Next Steps

After running the demos, you can:
1. Integrate with your backtesting framework (replace mock results)
2. Add your RAG system for querying books/documentation
3. Deploy Neo4j for production knowledge graph
4. Add more data sources beyond Binance
5. Implement parallel experiment execution
6. Add human-in-the-loop for reviewing strategies before backtesting
