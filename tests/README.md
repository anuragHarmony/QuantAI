# QuantAI Testing Suite

This directory contains comprehensive tests for the QuantAI knowledge base system.

## Test Files

### `test_queries.py`
Main test suite for verifying semantic search and retrieval functionality.

**Test Categories:**
- Concept Queries: Tests retrieval of fundamental concepts (mean reversion, momentum, etc.)
- Strategy Queries: Tests retrieval of trading strategies
- Formula Queries: Tests retrieval of mathematical formulas and calculations
- Risk Management Queries: Tests retrieval of risk management concepts
- Example Queries: Tests retrieval of code examples and case studies
- Market Regime Queries: Tests retrieval of market regime information
- Pairs Trading Queries: Tests retrieval of pairs trading specific knowledge

## Running Tests

### Run all tests:
```bash
python tests/test_queries.py
```

### Test specific queries interactively:
```python
from tests.test_queries import QueryTester

tester = QueryTester()

# Test a specific query
tester.detailed_query_test("What is the sharpe ratio?")

# Test context assembly
tester.test_context_assembly("Build a momentum strategy")
```

## Test Queries

### Example Concept Queries
- "What is mean reversion?"
- "Explain momentum trading"
- "How do moving averages work?"

### Example Strategy Queries
- "Moving average crossover strategy"
- "Pairs trading strategy"
- "Statistical arbitrage strategy"

### Example Formula Queries
- "Sharpe ratio formula"
- "Kelly criterion formula"
- "Z-score calculation"

### Example Risk Management Queries
- "Position sizing techniques"
- "How to manage drawdowns"
- "Stop loss strategies"

## Expected Outcomes

Tests verify that:
1. Queries return relevant results
2. Results are properly ranked by relevance
3. Context assembly works within token limits
4. Different chunk types (concept, strategy, formula, example) are retrievable
5. Metadata filtering works correctly

## Troubleshooting

### No results found
- Ensure documents have been indexed: `python index_documents.py`
- Check that the vector store contains data
- Verify embedding model is loaded correctly

### Low relevance scores
- Check query phrasing
- Ensure documents contain relevant content
- Consider adjusting chunk size and overlap parameters

### Performance issues
- Monitor token usage in context assembly
- Adjust max_results parameter
- Consider using filters to narrow search space
