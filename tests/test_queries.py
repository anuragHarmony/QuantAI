"""
Comprehensive test suite for knowledge base queries.

This module tests various query types to verify that the semantic search
and retrieval system is working correctly.
"""

import sys
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine import KnowledgeEngine
from shared.models.knowledge import QueryResult


console = Console()


class QueryTester:
    """Test various queries against the knowledge base."""

    def __init__(self):
        """Initialize the query tester."""
        console.print("[bold blue]Initializing Knowledge Engine...[/bold blue]")
        self.engine = KnowledgeEngine(use_llm_extraction=False)
        console.print(f"[green]✓ Knowledge Engine loaded[/green]")

        # Get stats
        stats = self.engine.get_stats()
        console.print(f"  Total chunks: {stats['total_chunks']}")
        console.print(f"  Embedding model: {stats['embedding_model']}")
        console.print()

    def run_all_tests(self):
        """Run all test queries."""
        console.rule("[bold cyan]Query Tests[/bold cyan]")

        # Test categories
        test_suites = [
            ("Concept Queries", self.test_concept_queries),
            ("Strategy Queries", self.test_strategy_queries),
            ("Formula Queries", self.test_formula_queries),
            ("Risk Management Queries", self.test_risk_management_queries),
            ("Example Queries", self.test_example_queries),
            ("Market Regime Queries", self.test_market_regime_queries),
            ("Pairs Trading Queries", self.test_pairs_trading_queries),
        ]

        total_tests = 0
        passed_tests = 0

        for suite_name, test_func in test_suites:
            console.print(f"\n[bold yellow]Running {suite_name}...[/bold yellow]")
            results = test_func()

            for test_name, success, query, num_results in results:
                total_tests += 1
                if success:
                    passed_tests += 1
                    console.print(f"  [green]✓[/green] {test_name}: {num_results} results")
                else:
                    console.print(f"  [red]✗[/red] {test_name}: No results found")

        # Summary
        console.print()
        console.rule("[bold cyan]Test Summary[/bold cyan]")
        console.print(f"Total tests: {total_tests}")
        console.print(f"[green]Passed: {passed_tests}[/green]")
        console.print(f"[red]Failed: {total_tests - passed_tests}[/red]")

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        console.print(f"Success rate: {success_rate:.1f}%")

    def test_concept_queries(self) -> List[tuple]:
        """Test queries for general concepts."""
        test_cases = [
            ("Mean Reversion", "What is mean reversion?"),
            ("Momentum Trading", "Explain momentum trading"),
            ("Moving Averages", "How do moving averages work?"),
            ("Quantitative Trading", "What is quantitative trading?"),
            ("Cointegration", "Explain cointegration"),
        ]

        results = []
        for test_name, query in test_cases:
            query_results = self.engine.search_concepts(query, max_results=5)
            success = len(query_results) > 0
            results.append((test_name, success, query, len(query_results)))

            if success and query_results:
                self._display_top_result(query, query_results[0])

        return results

    def test_strategy_queries(self) -> List[tuple]:
        """Test queries for trading strategies."""
        test_cases = [
            ("MA Crossover", "moving average crossover strategy"),
            ("Pairs Trading", "pairs trading strategy"),
            ("Statistical Arbitrage", "statistical arbitrage strategy"),
            ("Mean Reversion Strategy", "mean reversion trading strategy"),
        ]

        results = []
        for test_name, query in test_cases:
            query_results = self.engine.search_strategies(query, max_results=5)
            success = len(query_results) > 0
            results.append((test_name, success, query, len(query_results)))

            if success and query_results:
                self._display_top_result(query, query_results[0])

        return results

    def test_formula_queries(self) -> List[tuple]:
        """Test queries for formulas and calculations."""
        test_cases = [
            ("Sharpe Ratio", "sharpe ratio formula"),
            ("Kelly Criterion", "kelly criterion formula"),
            ("Moving Average", "simple moving average calculation"),
            ("Z-Score", "z-score calculation"),
        ]

        results = []
        for test_name, query in test_cases:
            query_results = self.engine.search(query, max_results=5)
            success = len(query_results) > 0
            results.append((test_name, success, query, len(query_results)))

            if success and query_results:
                self._display_top_result(query, query_results[0])

        return results

    def test_risk_management_queries(self) -> List[tuple]:
        """Test queries for risk management concepts."""
        test_cases = [
            ("Position Sizing", "position sizing techniques"),
            ("Drawdown Management", "how to manage drawdowns"),
            ("Stop Loss", "stop loss strategies"),
            ("Risk Controls", "risk management in trading"),
        ]

        results = []
        for test_name, query in test_cases:
            query_results = self.engine.search(query, max_results=5)
            success = len(query_results) > 0
            results.append((test_name, success, query, len(query_results)))

            if success and query_results:
                self._display_top_result(query, query_results[0])

        return results

    def test_example_queries(self) -> List[tuple]:
        """Test queries for examples and code."""
        test_cases = [
            ("Code Example", "moving average strategy code example"),
            ("Backtest Example", "backtesting example"),
            ("Z-Score Example", "z-score calculation example"),
        ]

        results = []
        for test_name, query in test_cases:
            query_results = self.engine.search_examples(query, max_results=5)
            success = len(query_results) > 0
            results.append((test_name, success, query, len(query_results)))

            if success and query_results:
                self._display_top_result(query, query_results[0])

        return results

    def test_market_regime_queries(self) -> List[tuple]:
        """Test queries for market regimes."""
        test_cases = [
            ("Trending Markets", "trending market strategies"),
            ("High Volatility", "high volatility trading"),
            ("Mean Reverting Markets", "mean reverting market conditions"),
            ("Market Regimes", "different market regimes"),
        ]

        results = []
        for test_name, query in test_cases:
            query_results = self.engine.search(query, max_results=5)
            success = len(query_results) > 0
            results.append((test_name, success, query, len(query_results)))

            if success and query_results:
                self._display_top_result(query, query_results[0])

        return results

    def test_pairs_trading_queries(self) -> List[tuple]:
        """Test queries specific to pairs trading."""
        test_cases = [
            ("Hedge Ratio", "hedge ratio calculation"),
            ("Spread Trading", "spread calculation in pairs trading"),
            ("Cointegration Test", "testing for cointegration"),
            ("Pair Selection", "how to select pairs for trading"),
        ]

        results = []
        for test_name, query in test_cases:
            query_results = self.engine.search(query, max_results=5)
            success = len(query_results) > 0
            results.append((test_name, success, query, len(query_results)))

            if success and query_results:
                self._display_top_result(query, query_results[0])

        return results

    def _display_top_result(self, query: str, result: QueryResult):
        """Display the top result for a query."""
        # Only display occasionally to avoid clutter
        pass

    def detailed_query_test(self, query: str):
        """
        Run a detailed test for a specific query.

        Args:
            query: The query to test
        """
        console.rule(f"[bold cyan]Detailed Query: {query}[/bold cyan]")

        # Search
        results = self.engine.search(query, max_results=10)

        if not results:
            console.print("[red]No results found![/red]")
            return

        # Display results in a table
        table = Table(title=f"Search Results for: '{query}'")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Score", style="magenta", width=8)
        table.add_column("Type", style="green", width=10)
        table.add_column("Source", style="yellow", width=30)
        table.add_column("Preview", style="white", width=60)

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            score = f"{result.relevance_score:.3f}"
            preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content

            table.add_row(
                str(i),
                score,
                chunk.chunk_type,
                f"{chunk.source_book[:30]}",
                preview
            )

        console.print(table)

        # Display top result in detail
        if results:
            top_result = results[0]
            panel = Panel(
                top_result.chunk.content,
                title=f"[bold]Top Result (Score: {top_result.relevance_score:.3f})[/bold]",
                subtitle=f"{top_result.chunk.source_book}",
                border_style="green"
            )
            console.print(panel)

    def test_context_assembly(self, query: str):
        """
        Test context assembly for AI reasoning.

        Args:
            query: The query to assemble context for
        """
        console.rule(f"[bold cyan]Context Assembly: {query}[/bold cyan]")

        # Assemble context
        context = self.engine.get_context(query, max_tokens=4000, include_examples=True)

        console.print(f"Query: [cyan]{context.query}[/cyan]")
        console.print(f"Results included: {len(context.results)}")
        console.print(f"Estimated tokens: {context.total_tokens}")
        console.print(f"Confidence score: {context.confidence_score:.3f}")
        console.print(f"Metadata: {context.metadata}")

        # Display results
        table = Table(title="Context Components")
        table.add_column("Type", style="green", width=10)
        table.add_column("Score", style="magenta", width=8)
        table.add_column("Source", style="yellow", width=30)
        table.add_column("Content Length", style="cyan", width=15)

        for result in context.results:
            chunk = result.chunk
            table.add_row(
                chunk.chunk_type,
                f"{result.relevance_score:.3f}",
                f"{chunk.source_book}",
                f"{len(chunk.content)} chars"
            )

        console.print(table)


def main():
    """Main entry point for query tests."""
    console.print(Panel.fit(
        "[bold cyan]QuantAI Knowledge Base Query Tests[/bold cyan]\n"
        "Testing semantic search and retrieval capabilities",
        border_style="blue"
    ))

    tester = QueryTester()

    # Run all automated tests
    tester.run_all_tests()

    console.print("\n")
    console.rule("[bold cyan]Detailed Query Examples[/bold cyan]")

    # Run some detailed tests
    detailed_queries = [
        "What is the sharpe ratio and how is it calculated?",
        "How do I implement a pairs trading strategy?",
        "What are the best strategies for high volatility markets?"
    ]

    for query in detailed_queries:
        console.print()
        tester.detailed_query_test(query)

    # Test context assembly
    console.print("\n")
    console.rule("[bold cyan]Context Assembly Tests[/bold cyan]")
    tester.test_context_assembly("Build a mean reversion strategy with risk management")


if __name__ == "__main__":
    main()
