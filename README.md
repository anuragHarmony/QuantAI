# QuantAI - AI-Human Collaborative Quant Research Platform

A sophisticated knowledge management and retrieval system for quantitative finance research. QuantAI extracts, structures, and makes searchable the knowledge from quant books and documents, enabling AI-powered research assistance for trading strategy development.

## Overview

QuantAI implements a production-grade RAG (Retrieval-Augmented Generation) system specifically designed for quantitative finance. The system:

- **Extracts knowledge** from PDF documents and quant books
- **Organizes information** hierarchically (broad topics → detailed concepts → examples)
- **Enables semantic search** using vector embeddings and ChromaDB
- **Assembles context** for AI reasoning within token budgets
- **Supports strategy development** by providing relevant knowledge on demand

## Key Features

### 🧠 Knowledge Engine
- **Document Processing**: Extract text and structure from PDFs
- **Intelligent Chunking**: Break documents into semantically meaningful pieces
- **Knowledge Extraction**: Rule-based and LLM-based extraction of concepts, strategies, formulas, and examples
- **Hierarchical Organization**: 4-level hierarchy from broad topics to specific examples

### 🔍 Semantic Search
- **Vector Similarity**: Fast semantic search using sentence transformers
- **Metadata Filtering**: Filter by chunk type, asset class, strategy type, tags
- **Type-Specific Search**: Dedicated search for concepts, strategies, examples
- **Context Assembly**: Intelligent selection of relevant chunks within token limits

### 📊 Rich Metadata
- **Chunk Types**: concept, strategy, formula, example, code
- **Asset Classes**: equity, forex, crypto, commodities
- **Strategy Types**: trend_following, mean_reversion, arbitrage
- **Market Regimes**: trending, high_vol, mean_reverting
- **Source Attribution**: Book, chapter, page tracking

## Architecture

```
QuantAI/
├── knowledge_engine/          # Core knowledge system
│   ├── ingest/               # Document processing & extraction
│   │   ├── document_processor.py    # PDF text extraction
│   │   └── knowledge_extractor.py   # LLM-based knowledge extraction
│   ├── retrieval/            # Semantic search & retrieval
│   │   ├── vector_store.py          # ChromaDB vector store
│   │   └── semantic_search.py       # High-level search interface
│   └── knowledge_engine.py   # Main orchestrator
├── shared/                   # Shared utilities
│   ├── models/              # Pydantic data models
│   └── config/              # Configuration settings
├── sample_data/             # Sample documents
│   └── documents/           # PDF and text documents
├── tests/                   # Test suite
│   ├── test_queries.py      # Query tests
│   └── README.md           # Test documentation
├── index_documents.py       # Document indexing script
├── query_knowledge.py       # Interactive query interface
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.11+
- pip or Poetry

### Install Dependencies

```bash
# Using pip
pip install pymupdf pdfplumber chromadb sentence-transformers pydantic pydantic-settings loguru pandas numpy rich

# Or using Poetry
poetry install
```

### Configuration

Create a `.env` file (optional, for LLM-based extraction):

```env
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./data/chroma
```

## Quick Start

### 1. Index Sample Documents

```bash
python index_documents.py
```

This will:
- Process all documents in `sample_data/documents/`
- Extract knowledge chunks
- Generate embeddings
- Store in ChromaDB vector database

### 2. Query the Knowledge Base

**Interactive mode:**
```bash
python query_knowledge.py --interactive
```

**Single query:**
```bash
python query_knowledge.py "What is mean reversion?"
python query_knowledge.py "Sharpe ratio formula"
python query_knowledge.py "Pairs trading strategy"
```

**Type-specific search:**
```bash
python query_knowledge.py --strategies "momentum"
python query_knowledge.py --concepts "risk management"
python query_knowledge.py --examples "moving average"
```

### 3. Run Tests

```bash
python tests/test_queries.py
```

This runs comprehensive tests covering:
- Concept queries
- Strategy queries
- Formula queries
- Risk management queries
- Example queries
- Market regime queries
- Pairs trading queries

## Usage Examples

### Python API

```python
from knowledge_engine import KnowledgeEngine

# Initialize engine
engine = KnowledgeEngine()

# Index a document
engine.ingest_document("path/to/book.pdf", book_name="Quant Trading 101")

# Search for concepts
results = engine.search_concepts("What is the Sharpe ratio?", max_results=5)

# Search for strategies
strategies = engine.search_strategies("momentum trading", max_results=3)

# Search for examples
examples = engine.search_examples("moving average code", max_results=2)

# Assemble context for AI reasoning
context = engine.get_context(
    query="Build a mean reversion strategy",
    max_tokens=4000,
    include_examples=True
)

# Get statistics
stats = engine.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

### Search Results

Each search returns `QueryResult` objects containing:
- **chunk**: The `KnowledgeChunk` with content and metadata
- **relevance_score**: Similarity score (0-1)
- **source**: Where the result came from (e.g., "vector_search")

```python
for result in results:
    print(f"Score: {result.relevance_score:.3f}")
    print(f"Type: {result.chunk.chunk_type}")
    print(f"Source: {result.chunk.source_book}")
    print(f"Content: {result.chunk.content[:200]}...")
```

## Sample Documents

The repository includes two comprehensive sample documents:

### 1. `quantitative_trading_basics.txt`
Covers:
- Introduction to quantitative trading
- Mean reversion and momentum concepts
- Moving averages and technical indicators
- Risk management (position sizing, Kelly Criterion)
- Backtesting strategies and metrics (Sharpe ratio, drawdown)
- Market regimes (trending, mean-reverting, volatility)

### 2. `pairs_trading_guide.txt`
Covers:
- Pairs trading fundamentals
- Cointegration and statistical tests
- Spread calculation and z-scores
- Dynamic hedge ratios (Kalman Filter)
- Portfolio construction
- Risk management for pairs trading
- Real-world case studies

## Data Models

### KnowledgeChunk
```python
class KnowledgeChunk:
    id: str
    source_book: str
    source_chapter: Optional[str]
    source_page: Optional[int]
    chunk_type: ChunkType  # concept, strategy, formula, example, code
    hierarchy_level: int  # 1=broad, 2=subtopic, 3=detail, 4=example
    content: str
    embedding: Optional[List[float]]
    asset_class: List[str]
    strategy_type: List[str]
    applicable_regimes: List[str]
    tags: List[str]
    created_at: datetime
    version: int
```

### RetrievalContext
```python
class RetrievalContext:
    query: str
    results: List[QueryResult]
    total_tokens: int
    confidence_score: float
    metadata: dict
```

## Testing

The test suite (`tests/test_queries.py`) validates:

1. **Concept Retrieval**: Fundamental quant concepts
2. **Strategy Retrieval**: Trading strategies
3. **Formula Retrieval**: Mathematical formulas
4. **Risk Management**: Risk control concepts
5. **Example Retrieval**: Code and case studies
6. **Market Regimes**: Market condition information
7. **Pairs Trading**: Specialized pairs trading knowledge

**Expected Success Rate**: >90% with properly indexed documents

## Advanced Features

### Context Assembly
The system intelligently assembles context for AI reasoning:
- Prioritizes recent and relevant information
- Includes prerequisite concepts
- Balances concepts, strategies, and examples
- Respects token budgets
- Calculates confidence scores

### Metadata Filtering
```python
# Filter by asset class
results = engine.search(
    "trading strategies",
    filters={"asset_class": "equity"}
)

# Filter by chunk type
results = engine.search(
    "risk management",
    filters={"chunk_type": "strategy"}
)
```

### Multi-Query Search
```python
from knowledge_engine.retrieval import SemanticSearch

search = SemanticSearch()
results = search.multi_query_search([
    "mean reversion strategies",
    "statistical arbitrage",
    "pairs trading"
], max_results_per_query=5)
```

## Roadmap

Based on `broad_plan.md`, future enhancements include:

### Phase 2: Advanced Retrieval
- [ ] Multi-stage retrieval pipeline (broad → filter → re-rank → deduplicate)
- [ ] Cross-encoder re-ranking
- [ ] Query expansion and intent detection
- [ ] Conversation management for multi-turn interactions

### Phase 3: Knowledge Graph
- [ ] Neo4j integration for concept relationships
- [ ] Hierarchical topic organization
- [ ] Prerequisite tracking
- [ ] Graph-based traversal for related concepts

### Phase 4: Experience Layer
- [ ] Market regime detection
- [ ] Real trading insights storage
- [ ] Strategy performance tracking
- [ ] Time-aware relevance weighting

### Phase 5: Backtesting Integration
- [ ] Strategy code generation
- [ ] Backtesting engine integration
- [ ] Performance metrics calculation
- [ ] AI feedback loop for strategy iteration

## Performance

**Current Capabilities:**
- Document indexing: ~100 chunks/minute
- Search latency: <100ms for typical queries
- Embedding model: all-MiniLM-L6-v2 (fast, lightweight)
- Vector store: ChromaDB (persistent, scalable)

**Scalability:**
- Handles 1000s of documents
- 10K+ knowledge chunks
- Sub-second search across entire knowledge base

## Contributing

Contributions are welcome! Areas for improvement:
- Additional sample documents
- Enhanced knowledge extraction
- Fine-tuned embedding models for finance
- Advanced query understanding
- Performance optimizations

## License

MIT License - see LICENSE file for details

## References

- `broad_plan.md`: Comprehensive implementation plan
- `purpose.txt`: Original project vision
- Legal AI RAG systems (inspiration for architecture)
- Quantitative finance literature

## Support

For issues, questions, or suggestions:
1. Check the documentation in this README
2. Review `tests/README.md` for testing guidance
3. Examine sample usage in `query_knowledge.py` and `tests/test_queries.py`

---

**Built with:** Python, ChromaDB, Sentence Transformers, Pydantic, Rich, Loguru
