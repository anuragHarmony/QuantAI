# QuantAI - AI-Human Collaborative Quant Research Platform

An intelligent quantitative trading research platform that combines AI-powered knowledge retrieval with production-grade backtesting infrastructure.

## 🎯 Overview

QuantAI is not just another trading bot - it's a **research partnership platform** designed to enhance human quant research with AI capabilities. The system features:

- **Knowledge Engine**: RAG-powered system that learns from quant books and trading experiences
- **AI Tool Framework**: Extensible function-calling framework for trading operations
- **Production-Ready Architecture**: SOLID principles, async/await, comprehensive abstractions
- **Multi-Stage Retrieval**: Advanced semantic search with re-ranking and deduplication
- **Backtesting Infrastructure**: High-performance testing framework (coming in Phase 2B)

## 🏗️ Architecture

### Core Components

```
QuantAI/
├── knowledge_engine/       # RAG knowledge system
│   ├── ingest/            # Document processing & extraction
│   ├── graph/             # Neo4j knowledge graph
│   ├── retrieval/         # Multi-stage semantic search
│   └── experiences/       # Market insights storage
├── backtesting/           # Backtesting infrastructure
│   ├── engine/            # Core backtesting logic
│   ├── data/              # Market data management
│   ├── strategies/        # Strategy implementations
│   └── parallel/          # Multi-strategy runner
├── ai_agent/              # AI reasoning & tools
│   ├── reasoner/          # LLM & embedding providers
│   ├── strategy_generator/# AI strategy creation
│   ├── feedback_loop/     # Test-learn-iterate cycle
│   └── tools/             # AI function calling tools
├── shared/                # Shared utilities
│   ├── models/            # Pydantic models & interfaces
│   ├── config/            # Configuration management
│   └── utils/             # URL fetcher, caching, etc.
└── api/                   # FastAPI REST API
```

## ✨ Features

### Phase 2A - Completed ✅

- ✅ **SOLID Architecture**: Abstract base classes for all major components
- ✅ **Async/Await**: Full async support throughout
- ✅ **URL Fetching**: Document downloading with retry logic
- ✅ **AI Tool Framework**: OpenAI & Anthropic function calling support
- ✅ **Trading Tools**: Market data, indicators, signals, metrics
- ✅ **LLM Providers**: OpenAI (GPT-4) & Anthropic (Claude) integration
- ✅ **Embedding Providers**: OpenAI, SentenceTransformers, Hybrid
- ✅ **Vector Store**: ChromaDB & FAISS implementations
- ✅ **Knowledge Graph**: Neo4j with relationship traversal
- ✅ **Document Processing**: PDF extraction with structure preservation
- ✅ **Caching Layer**: Redis with in-memory fallback
- ✅ **RAG Pipeline**: Multi-stage retrieval with re-ranking
- ✅ **REST API**: FastAPI with tool execution endpoints
- ✅ **CLI Tool**: Interactive command-line interface

### Phase 2B - In Progress 🚧

- ✅ **Event System**: Production event-driven architecture (10k+ events/sec)
  - Type-safe events (Market Data, Orders, Positions, Portfolio)
  - Pub/sub event bus (in-memory + Redis for multi-process)
  - Event filters with composition
  - Event persistence for replay (coming)

- 🚧 **Exchange Connectors**: Multi-exchange framework (in progress)
- 🔜 **Order Management System**: Professional OMS with pre-trade checks
- 🔜 **Portfolio Manager**: Real-time P&L and risk limits
- 🔜 **Strategy Framework**: Simple event-driven strategies
- 🔜 **Simulation Engine**: Data recording and replay

### Coming Later (Phase 3+)

- 🔜 Full backtesting engine with vectorbt
- 🔜 Parallel strategy testing
- 🔜 AI strategy generator
- 🔜 Feedback loop system
- 🔜 Market regime detection
- 🔜 Web dashboard

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Optional: Docker (for databases)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QuantAI
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. (Optional) Start databases with Docker:
```bash
# Redis
docker run -d -p 6379:6379 redis:latest

# Neo4j
docker run -d -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# PostgreSQL
docker run -d -p 5432:5432 \
  -e POSTGRES_DB=quantai \
  -e POSTGRES_USER=quantai \
  -e POSTGRES_PASSWORD=quantai \
  postgres:latest
```

### Quick Test

Test the AI tool framework:

```bash
poetry run python cli.py list-tools
```

Test market data fetching:

```bash
poetry run python cli.py test-market-data AAPL 2023-01-01 --end-date 2024-01-01
```

Test a trading strategy:

```bash
poetry run python cli.py test-strategy AAPL --strategy MA_CROSSOVER
```

### Start the API Server

```bash
poetry run python cli.py start-api
```

Then visit http://localhost:8000/docs for the interactive API documentation.

## 📖 Usage Examples

### 1. Using the CLI

**List available AI tools:**
```bash
poetry run python cli.py list-tools
```

**Interactive chat with AI:**
```bash
poetry run python cli.py chat --interactive
```

**Test a strategy:**
```bash
poetry run python cli.py test-strategy AAPL \
  --strategy MA_CROSSOVER \
  --start-date 2023-01-01 \
  --end-date 2024-01-01
```

### 2. Using the Python API

```python
import asyncio
from ai_agent.tools.trading_tools import GetMarketDataTool, GenerateSignalsTool

async def main():
    # Fetch market data
    data_tool = GetMarketDataTool()
    result = await data_tool.execute(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )

    if result.success:
        prices = result.result.data["close"]

        # Generate signals
        signal_tool = GenerateSignalsTool()
        signals = await signal_tool.execute(
            strategy="MA_CROSSOVER",
            prices=prices
        )

        print(f"Buy signals: {signals.result['num_buy']}")
        print(f"Sell signals: {signals.result['num_sell']}")

asyncio.run(main())
```

### 3. Using the REST API

```bash
# Execute a tool via API
curl -X POST "http://localhost:8000/tools/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "get_market_data",
    "parameters": {
      "symbol": "AAPL",
      "start_date": "2023-01-01"
    }
  }'

# Chat with AI
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Get market data for AAPL",
    "use_tools": true
  }'
```

## 🧪 Testing

Run tests:
```bash
poetry run pytest
```

With coverage:
```bash
poetry run pytest --cov=. --cov-report=html
```

## 🛠️ Development

### Adding New AI Tools

1. Create a new tool class inheriting from `BaseTool`:

```python
from ai_agent.tools.base import BaseTool, ToolParameter, ToolParameterType, ToolResult

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return "Description of what this tool does"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="param1",
                type=ToolParameterType.STRING,
                description="Parameter description",
                required=True
            )
        ]

    async def execute(self, **kwargs) -> ToolResult:
        # Tool implementation
        result = do_something(kwargs["param1"])
        return ToolResult(success=True, result=result)
```

2. Register the tool:

```python
from ai_agent.tools.base import global_registry

global_registry.register(MyCustomTool())
```

### Project Structure

- **SOLID Principles**: All major components have abstract base interfaces in `shared/models/base.py`
- **Async First**: All I/O operations use async/await
- **Type Safety**: Pydantic models for data validation
- **Logging**: Loguru for structured logging
- **Configuration**: Pydantic Settings for environment-based config

## 📚 Documentation

For detailed documentation, see:
- [Master Plan](docs/plans/broad_plan.md) - Complete implementation roadmap
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server running)

## 🔑 Environment Variables

Key environment variables (see `.env.example` for full list):

```bash
# LLM API Keys
LLM_OPENAI_API_KEY=your-openai-key
LLM_ANTHROPIC_API_KEY=your-anthropic-key

# Database connections
DB_POSTGRES_HOST=localhost
NEO4J_URI=bolt://localhost:7687
REDIS_HOST=localhost
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints throughout
5. Follow async/await patterns

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- FastAPI - Modern web framework
- OpenAI & Anthropic - LLM providers
- ChromaDB - Vector database
- Neo4j - Graph database
- Pydantic - Data validation
- And many more excellent open-source projects

---

**Status**: Phase 2A Complete ✅

Next up: Phase 2B - Full backtesting engine and parallel strategy testing
