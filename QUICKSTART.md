# QuantAI Quick Start Guide

## Installation Status

The dependencies are currently being installed. Large packages like `sentence-transformers` and `chromadb` can take 15-20 minutes to install as they download model files.

## Step-by-Step Setup

### 1. Verify Installation

Once the installation completes, verify that all packages are installed:

```bash
python -c "import chromadb; import sentence_transformers; import pymupdf; print('✓ All packages installed successfully!')"
```

### 2. Index Sample Documents

Run the indexing script to process the sample documents and populate the knowledge base:

```bash
python index_documents.py
```

**Expected output:**
```
Starting document indexing
Found 2 documents to index
Processing quantitative_trading_basics.txt...
Processing pairs_trading_guide.txt...
Total chunks indexed: ~40-60 chunks
```

This will create a `data/chroma/` directory with the vector database.

### 3. Test the System

Run the comprehensive test suite:

```bash
python tests/test_queries.py
```

**What the tests do:**
- Test 7 different query categories
- Verify semantic search works correctly
- Check retrieval quality and ranking
- Validate context assembly
- Show detailed query examples

**Expected success rate**: >90% with proper indexing

### 4. Interactive Querying

Try the interactive query interface:

```bash
python query_knowledge.py --interactive
```

**Example queries to try:**
```
Query: What is mean reversion?
Query: Sharpe ratio formula
Query: How to implement pairs trading?
Query: Risk management strategies
Query: Moving average crossover example
```

**Commands available:**
- `stats` - Show knowledge base statistics
- `help` - Show available commands
- `exit` or `quit` - Exit the program

### 5. Single Query Mode

Query directly from command line:

```bash
# General search
python query_knowledge.py "What is the Kelly Criterion?"

# Search for strategies only
python query_knowledge.py --strategies "momentum"

# Search for concepts only
python query_knowledge.py --concepts "drawdown"

# Search for examples only
python query_knowledge.py --examples "moving average code"
```

## Sample Queries by Category

### Concept Queries
- "What is mean reversion?"
- "Explain cointegration"
- "What is quantitative trading?"
- "Define momentum trading"

### Strategy Queries
- "Moving average crossover strategy"
- "Pairs trading implementation"
- "Statistical arbitrage approach"
- "Mean reversion trading strategy"

### Formula Queries
- "Sharpe ratio calculation"
- "Kelly Criterion formula"
- "Z-score formula"
- "Moving average formula"

### Risk Management
- "Position sizing techniques"
- "Drawdown management strategies"
- "Stop loss implementation"
- "Risk controls in trading"

### Market Regimes
- "Trading in trending markets"
- "High volatility strategies"
- "Mean reverting market conditions"
- "Regime detection methods"

### Pairs Trading
- "Cointegration testing"
- "Hedge ratio calculation"
- "Spread trading signals"
- "Pair selection criteria"

## Understanding the Results

### Result Display Format

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Search Results for: 'your query'          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌──────┬────────┬──────────┬──────────────────┐
│ Rank │ Score  │ Type     │ Source           │
├──────┼────────┼──────────┼──────────────────┤
│ 1    │ 0.876  │ concept  │ trading_basics   │
│ 2    │ 0.823  │ strategy │ pairs_trading    │
└──────┴────────┴──────────┴──────────────────┘

Top Result: [Actual content displayed here]
```

### Score Interpretation
- **0.9 - 1.0**: Excellent match, highly relevant
- **0.8 - 0.9**: Very good match, relevant
- **0.7 - 0.8**: Good match, somewhat relevant
- **< 0.7**: Lower relevance, may be tangentially related

### Result Types
- **concept**: Fundamental concepts and definitions
- **strategy**: Trading strategies and approaches
- **formula**: Mathematical formulas and calculations
- **example**: Code examples and case studies
- **code**: Implementation code snippets

## Python API Usage

### Basic Usage

```python
from knowledge_engine import KnowledgeEngine

# Initialize
engine = KnowledgeEngine()

# Search
results = engine.search("What is the Sharpe ratio?")

# Display results
for result in results:
    print(f"Score: {result.relevance_score:.3f}")
    print(f"Content: {result.chunk.content[:200]}...")
    print()
```

### Type-Specific Search

```python
# Search concepts
concepts = engine.search_concepts("risk management", max_results=5)

# Search strategies
strategies = engine.search_strategies("momentum", max_results=3)

# Search examples
examples = engine.search_examples("code", max_results=2)
```

### Context Assembly for AI

```python
# Assemble context for AI reasoning
context = engine.get_context(
    query="Build a mean reversion strategy with proper risk management",
    max_tokens=4000,
    include_examples=True
)

print(f"Confidence: {context.confidence_score:.3f}")
print(f"Total tokens: {context.total_tokens}")
print(f"Results included: {len(context.results)}")

# Use context with your LLM
for result in context.results:
    print(result.chunk.content)
```

### Indexing Your Own Documents

```python
from pathlib import Path

# Index a PDF book
num_chunks = engine.ingest_document(
    Path("path/to/your/book.pdf"),
    book_name="My Quant Book"
)

print(f"Indexed {num_chunks} chunks")
```

## Knowledge Base Statistics

Check the current state of your knowledge base:

```python
stats = engine.get_stats()
print(stats)
# Output:
# {
#     'total_chunks': 45,
#     'vector_store': 'ChromaDB',
#     'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
# }
```

## Troubleshooting

### Import Errors

If you see import errors:
```bash
pip install -r requirements.txt
```

### No Results Found

If queries return no results:
1. Verify indexing completed: `python index_documents.py`
2. Check data directory exists: `ls -la data/chroma/`
3. Verify chunks were created: Run the stats command

### Low Relevance Scores

If results have low relevance:
1. Try rephrasing your query
2. Use more specific terms
3. Try type-specific search (--strategies, --concepts)

### Installation Issues

If packages fail to install:
```bash
# Try installing one at a time
pip install pymupdf
pip install chromadb
pip install sentence-transformers
pip install pydantic pydantic-settings
pip install loguru rich pandas numpy
```

## Next Steps

1. **Add Your Own Documents**: Place PDF files in `sample_data/documents/` and run indexing
2. **Integrate with LLM**: Use the context assembly feature with OpenAI/Claude
3. **Customize Extraction**: Modify `knowledge_extractor.py` for domain-specific extraction
4. **Extend Metadata**: Add your own tags and categories in the models
5. **Build Applications**: Use the API to build custom trading research tools

## Performance Tips

- **Batch Indexing**: Index multiple documents at once for efficiency
- **Adjust Chunk Size**: Modify chunk_size parameter (default: 1000) based on your documents
- **Filter by Type**: Use type-specific search to narrow results
- **Context Budget**: Adjust max_tokens based on your LLM's context window

## Sample Document Content

The system includes two comprehensive documents:

1. **quantitative_trading_basics.txt** (8000+ words)
   - Covers core quant concepts
   - Risk management techniques
   - Backtesting methodologies
   - Market regimes

2. **pairs_trading_guide.txt** (9000+ words)
   - Pairs trading fundamentals
   - Cointegration analysis
   - Portfolio construction
   - Real-world case studies

## Questions or Issues?

- Check the main README.md for detailed documentation
- Review tests/README.md for testing guidance
- Examine the code examples in query_knowledge.py
- Look at the comprehensive tests in tests/test_queries.py

---

**Ready to start?** Once installation completes, run:
```bash
python index_documents.py && python query_knowledge.py --interactive
```
