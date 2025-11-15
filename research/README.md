# QuantAI Research Module

Autonomous AI research agent for discovering profitable trading strategies.

## Overview

The QuantAI research module implements an autonomous AI agent that acts like a quantitative researcher. It combines:
- **Knowledge Graph**: Structured concepts (indicators, strategies, market regimes)
- **Data Analytics**: Market data ingestion, feature engineering, pattern detection
- **LLM Reasoning**: Strategy hypothesis generation using Claude
- **Code Generation**: Converts ideas to executable Python code
- **Memory Systems**: Learns from past experiments
- **Reflection**: Analyzes results and extracts insights

## Architecture

```
research/
├── agent/                    # AI Agent Core
│   ├── core/                # Core research components
│   │   ├── hypothesis.py    # Hypothesis generation
│   │   ├── code_generator.py # Code generation
│   │   ├── reflection.py    # Reflection & learning
│   │   └── orchestrator.py  # Research orchestration
│   ├── llm/                 # LLM providers
│   │   ├── interface.py     # Abstract ILLM interface
│   │   └── claude_provider.py # Claude implementation
│   ├── memory/              # Memory systems
│   │   ├── interface.py     # Memory interfaces
│   │   └── simple_memory.py # In-memory implementations
│   └── agent_factory.py     # Agent factory
│
├── knowledge/               # Knowledge Graph
│   ├── interface.py         # Knowledge graph interface
│   ├── memory_graph.py      # In-memory implementation
│   └── integrated_retriever.py # RAG + KG + Memory retriever
│
└── data/                    # Data Analytics
    ├── interface.py         # Data provider interfaces
    ├── binance_provider.py  # Binance data fetching
    ├── feature_engineer.py  # Technical indicators
    └── pattern_detector.py  # Pattern detection
```

## Quick Start

```python
import asyncio
from research.agent.agent_factory import AgentFactory

async def main():
    # Create agent (uses ANTHROPIC_API_KEY env var)
    agent = await AgentFactory.create_simple_agent()

    # Run autonomous research
    session = await agent.run_research_loop(
        objective="Discover profitable mean reversion strategies",
        focus="mean reversion",
        max_iterations=5
    )

    # Get best strategies
    best = await agent.get_best_strategies(top_k=3)

    for strategy in best:
        print(f"Sharpe: {strategy.results['sharpe_ratio']:.2f}")
        print(f"Strategy: {strategy.hypothesis['description']}")

asyncio.run(main())
```

See `examples/demo_research_agent.py` for complete examples.

## How It Works

### Research Loop

The agent runs an autonomous research loop:

```
1. HYPOTHESIS GENERATION
   ├─ Query knowledge graph for relevant concepts
   ├─ Analyze current market data and patterns
   ├─ Review past successful/failed experiments
   ├─ Use LLM to synthesize new strategy hypothesis
   └─ Output: StrategyHypothesis with indicators, entry/exit logic

2. CODE GENERATION
   ├─ Convert hypothesis to Python code template
   ├─ Use LLM to implement calculate_indicators()
   ├─ Use LLM to implement generate_signals()
   ├─ Validate syntax and structure
   └─ Output: Executable Python strategy class

3. BACKTESTING
   ├─ Run strategy on historical data
   ├─ Calculate metrics (Sharpe, win rate, etc.)
   └─ Output: Backtest results

4. REFLECTION
   ├─ Analyze why strategy succeeded/failed
   ├─ Compare to similar past experiments
   ├─ Extract specific insights
   └─ Output: Analysis with learnings

5. LEARNING
   ├─ Save experiment to episodic memory
   ├─ Extract general principles (every N experiments)
   ├─ Update semantic memory with insights
   └─ Generate suggestions for next experiments

6. ITERATE
   └─ Return to step 1 with new knowledge
```

### Knowledge Sources

The agent queries three knowledge sources:

**1. Knowledge Graph** (structured concepts)
- Indicators: RSI, MACD, Bollinger Bands, ATR, etc.
- Strategies: Mean Reversion, Momentum, Trend Following, etc.
- Market Regimes: Trending, Ranging, Volatile, etc.
- Relationships: "Mean Reversion USES RSI", "Works in Ranging markets"

**2. Episodic Memory** (past experiments)
- Experiment results (Sharpe, win rate, total return)
- What worked and what didn't
- Similar experiments for comparison

**3. Semantic Memory** (learned principles)
- General insights extracted from multiple experiments
- "RSI works well in ranging markets with Bollinger Bands"
- "Stop losses should be ATR-based for volatility adaptation"

## Components

### Hypothesis Generator

Generates testable strategy hypotheses using LLM + knowledge:

```python
from research.agent.core import HypothesisGenerator

generator = HypothesisGenerator(llm, knowledge_retriever)

# Generate single hypothesis
hypothesis = await generator.generate_hypothesis(
    focus="mean reversion",
    market_data=data,
    patterns=detected_patterns,
    regime=current_regime
)

# Generate diverse batch
hypotheses = await generator.generate_batch(
    count=5,
    diversity=0.8
)

# Generate variation
variation = await generator.generate_variation(
    base_hypothesis=hypothesis,
    variation_type="improve"
)
```

### Code Generator

Converts hypotheses to executable Python code:

```python
from research.agent.core import CodeGenerator

generator = CodeGenerator(llm)

# Generate code
generated = await generator.generate_strategy_code(hypothesis)

if generated.is_valid:
    print(generated.code)
    await generator.save_to_file(generated, "strategy.py")
else:
    print(f"Errors: {generated.validation_errors}")
```

### Reflection Engine

Analyzes results and learns:

```python
from research.agent.core import ReflectionEngine

reflection = ReflectionEngine(llm, semantic_memory)

# Analyze single experiment
analysis = await reflection.analyze_experiment(experiment)
print(analysis['is_successful'])
print(analysis['insights'])

# Extract general insights
insights = await reflection.extract_insights(
    experiments=all_experiments,
    min_experiments=3
)

# Suggest next experiments
suggestions = await reflection.suggest_next_experiments(
    past_experiments=experiments,
    insights=insights,
    count=3
)
```

### Research Orchestrator

Coordinates the full research loop:

```python
from research.agent.core import ResearchOrchestrator, ResearchConfig

# Configure
config = ResearchConfig(
    max_iterations=10,
    min_sharpe_threshold=1.5,
    backtest_symbol="BTC/USDT",
    backtest_timeframe="1h"
)

# Create orchestrator
orchestrator = ResearchOrchestrator(
    llm=llm,
    knowledge_retriever=knowledge,
    data_provider=data_provider,
    feature_engineer=feature_engineer,
    pattern_detector=pattern_detector,
    episodic_memory=episodic_memory,
    semantic_memory=semantic_memory,
    working_memory=working_memory,
    config=config
)

# Run research
session = await orchestrator.run_research_loop(
    objective="Discover high Sharpe strategies",
    max_iterations=10
)

# Get results
best = await orchestrator.get_best_strategies(top_k=5)
report = await orchestrator.generate_research_report(session)
```

## Data Analytics

### Data Provider

Fetches market data from exchanges:

```python
from research.data import BinanceDataProvider

provider = BinanceDataProvider()

# Fetch OHLCV data
data = await provider.fetch_ohlcv(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 1)
)

# Returns MarketData with pandas DataFrame
print(data.data.head())
```

### Feature Engineer

Calculates 50+ technical indicators:

```python
from research.data import TechnicalFeatureEngineer

engineer = TechnicalFeatureEngineer()

# Calculate all indicators
df = await engineer.calculate_technical_indicators(data.data)

# Or specific indicators
df = await engineer.calculate_technical_indicators(
    data.data,
    indicators=["rsi", "macd", "bollinger"]
)

# Available indicators:
# - SMA, EMA (multiple periods)
# - RSI, MACD, Bollinger Bands
# - ATR, ADX, Stochastic
# - OBV, VWAP
# - Plus regime indicators
```

### Pattern Detector

Detects patterns and market regimes:

```python
from research.data import SimplePatternDetector

detector = SimplePatternDetector()

# Detect trends
trends = await detector.detect_trends(data.data)

# Detect market regime
regimes = await detector.detect_regimes(data.data)
print(regimes[0].regime_type)  # trending, ranging, volatile, etc.

# Detect support/resistance
levels = await detector.detect_support_resistance(data.data)

# Detect seasonality
seasonality = await detector.detect_seasonality(data.data)
```

## Memory Systems

### Episodic Memory

Stores past experiments:

```python
from research.agent.memory import SimpleEpisodicMemory, Experiment

memory = SimpleEpisodicMemory()

# Save experiment
experiment = Experiment(
    id="exp_001",
    hypothesis={"description": "RSI mean reversion"},
    code="...",
    results={"sharpe_ratio": 1.85, "win_rate": 0.58},
    timestamp=datetime.now(),
    insights=[]
)

await memory.save_experiment(experiment)

# Query experiments
successful = await memory.get_successful_experiments(min_sharpe=1.5)

# Find similar
similar = await memory.find_similar_experiments("mean reversion")
```

### Semantic Memory

Stores learned principles:

```python
from research.agent.memory import SimpleSemanticMemory, Principle

memory = SimpleSemanticMemory()

# Save principle
principle = Principle(
    principle="RSI works well in ranging markets with BB filter",
    category="indicator",
    confidence=0.85,
    supporting_evidence=["exp_001", "exp_005"]
)

await memory.save_principle(principle)

# Query principles
principles = await memory.query_principles(
    category="indicator",
    min_confidence=0.7
)
```

## Configuration

### Research Config

```python
from research.agent.core import ResearchConfig

config = ResearchConfig(
    # Iteration limits
    max_iterations=10,
    min_sharpe_threshold=1.5,

    # Hypothesis diversity
    diversity_threshold=0.7,  # 0-1, higher = more diverse

    # Backtesting
    backtest_symbol="BTC/USDT",
    backtest_timeframe="1h",
    backtest_start_date="2024-01-01",
    backtest_end_date="2024-06-01",

    # Learning
    learning_interval=3,  # Extract insights every N experiments

    # Execution
    parallel_experiments=1  # Not yet implemented
)
```

### Agent Factory

```python
from research.agent.agent_factory import AgentFactory

# Simple agent (default config)
agent = await AgentFactory.create_simple_agent(
    anthropic_api_key="sk-..."
)

# Custom agent
agent = await AgentFactory.create_research_agent(
    anthropic_api_key="sk-...",
    config=custom_config,
    use_neo4j=False  # Use in-memory for now
)
```

## Production Integration

### Backtesting Framework

The orchestrator currently uses mock backtest results. To integrate with your backtesting framework:

```python
# In orchestrator.py, replace _run_backtest():

async def _run_backtest(
    self,
    generated_code: GeneratedCode
) -> Dict[str, Any]:
    # Import your backtesting framework
    from backtesting import BacktestRunner

    # Load generated strategy
    strategy_class = load_strategy_from_code(generated_code.code)

    # Run backtest
    runner = BacktestRunner(
        strategy=strategy_class,
        symbol=self.config.backtest_symbol,
        timeframe=self.config.backtest_timeframe,
        start_date=self.config.backtest_start_date,
        end_date=self.config.backtest_end_date
    )

    results = await runner.run()

    return {
        "sharpe_ratio": results.sharpe,
        "total_return": results.total_return,
        "win_rate": results.win_rate,
        "total_trades": results.total_trades,
        "max_drawdown": results.max_drawdown,
        "profit_factor": results.profit_factor
    }
```

### Neo4j Knowledge Graph

To use Neo4j instead of in-memory:

```python
# 1. Implement Neo4jKnowledgeGraph (research/knowledge/neo4j_graph.py)
from research.knowledge.interface import IKnowledgeGraph

class Neo4jKnowledgeGraph(IKnowledgeGraph):
    def __init__(self, uri, user, password):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    async def add_concept(self, concept: Concept):
        # Implement using Cypher queries
        pass

# 2. Update agent factory
agent = await AgentFactory.create_research_agent(
    use_neo4j=True,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)
```

### RAG System Integration

To integrate your existing RAG system:

```python
# In agent_factory.py:

from your_rag_system import RAGSystem

knowledge_retriever = IntegratedKnowledgeRetriever(
    rag_system=RAGSystem(),  # Your RAG implementation
    knowledge_graph=knowledge_graph,
    episodic_memory=episodic_memory,
    semantic_memory=semantic_memory
)
```

## Design Principles

All components follow SOLID principles:

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Extend via interfaces, don't modify existing code
- **Liskov Substitution**: Implementations are swappable
- **Interface Segregation**: Focused interfaces (IDataProvider, IFeatureEngineer, etc.)
- **Dependency Inversion**: Depend on abstractions (interfaces), not concrete classes

## Performance Notes

- **API Costs**: Each research iteration makes 2-4 LLM calls
- **Time**: ~10-30 seconds per iteration (depends on LLM latency)
- **Memory**: In-memory implementations suitable for development; use databases for production
- **Parallelization**: Not yet implemented; future work for parallel experiment execution

## Examples

See `examples/` directory for:
- Simple research demo (5 iterations)
- Advanced research demo (10 iterations, custom config)
- Hypothesis generation only
- Code generation demo
- Complete documentation

## Next Steps

1. **Integrate backtesting**: Replace mock results with real backtests
2. **Add RAG system**: Integrate your book/documentation query system
3. **Production databases**: PostgreSQL for experiments, Neo4j for knowledge graph
4. **Vector search**: Add embeddings for semantic search in memory
5. **Parallel execution**: Run multiple experiments concurrently
6. **Human-in-the-loop**: Add approval step before backtesting
7. **Live trading**: Integrate with execution system

## License

See main project LICENSE file.
