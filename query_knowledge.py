"""
Interactive script to query the knowledge base.

Usage:
    python query_knowledge.py "your query here"
    python query_knowledge.py --interactive
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_engine import KnowledgeEngine


console = Console()


def display_results(query: str, results, max_display: int = 5):
    """Display search results in a formatted table."""
    if not results:
        console.print("[red]No results found![/red]")
        return

    console.print(f"\n[bold cyan]Found {len(results)} results for:[/bold cyan] '{query}'\n")

    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Type", style="yellow", width=10)
    table.add_column("Source", style="blue", width=35)

    for i, result in enumerate(results[:max_display], 1):
        chunk = result.chunk
        table.add_row(
            str(i),
            f"{result.relevance_score:.3f}",
            chunk.chunk_type,
            f"{chunk.source_book}"
        )

    console.print(table)

    # Show top result content
    if results:
        top_result = results[0]
        panel = Panel(
            top_result.chunk.content[:500] + ("..." if len(top_result.chunk.content) > 500 else ""),
            title=f"[bold]Top Result (Relevance: {top_result.relevance_score:.3f})[/bold]",
            subtitle=f"Source: {top_result.chunk.source_book} | Type: {top_result.chunk.chunk_type}",
            border_style="green",
            expand=False
        )
        console.print("\n")
        console.print(panel)


def interactive_mode(engine: KnowledgeEngine):
    """Run in interactive query mode."""
    console.print(Panel.fit(
        "[bold cyan]QuantAI Knowledge Base - Interactive Query Mode[/bold cyan]\n"
        "Type your queries or commands below.\n"
        "Commands: 'exit', 'quit', 'stats', 'help'",
        border_style="blue"
    ))

    while True:
        console.print()
        query = Prompt.ask("[bold yellow]Query[/bold yellow]")

        if not query:
            continue

        query_lower = query.lower().strip()

        # Handle commands
        if query_lower in ['exit', 'quit', 'q']:
            console.print("[cyan]Goodbye![/cyan]")
            break

        elif query_lower == 'stats':
            stats = engine.get_stats()
            console.print(Panel(
                f"[bold]Knowledge Base Statistics[/bold]\n\n"
                f"Total chunks: {stats['total_chunks']}\n"
                f"Vector store: {stats['vector_store']}\n"
                f"Embedding model: {stats['embedding_model']}",
                border_style="cyan"
            ))

        elif query_lower == 'help':
            console.print(Panel(
                "[bold]Available Commands[/bold]\n\n"
                "[cyan]stats[/cyan] - Show knowledge base statistics\n"
                "[cyan]exit/quit[/cyan] - Exit the program\n"
                "[cyan]help[/cyan] - Show this help message\n\n"
                "[bold]Query Syntax[/bold]\n\n"
                "Simply type your question or search query.\n"
                "Examples:\n"
                "  - What is mean reversion?\n"
                "  - Sharpe ratio formula\n"
                "  - Pairs trading strategy",
                border_style="cyan"
            ))

        else:
            # Perform search
            with console.status("[bold green]Searching...", spinner="dots"):
                results = engine.search(query, max_results=10)

            display_results(query, results)


def single_query_mode(engine: KnowledgeEngine, query: str):
    """Run a single query and exit."""
    console.print(f"[bold cyan]Searching for:[/bold cyan] '{query}'\n")

    with console.status("[bold green]Searching...", spinner="dots"):
        results = engine.search(query, max_results=10)

    display_results(query, results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query the QuantAI knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_knowledge.py "What is mean reversion?"
  python query_knowledge.py --interactive
  python query_knowledge.py --strategies "momentum"
  python query_knowledge.py --concepts "sharpe ratio"
        """
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (if not provided, enters interactive mode)'
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '-s', '--strategies',
        metavar='QUERY',
        help='Search for strategies'
    )

    parser.add_argument(
        '-c', '--concepts',
        metavar='QUERY',
        help='Search for concepts'
    )

    parser.add_argument(
        '-e', '--examples',
        metavar='QUERY',
        help='Search for examples'
    )

    parser.add_argument(
        '-n', '--max-results',
        type=int,
        default=10,
        help='Maximum number of results to return (default: 10)'
    )

    args = parser.parse_args()

    # Initialize engine
    console.print("[bold blue]Initializing Knowledge Engine...[/bold blue]")
    try:
        engine = KnowledgeEngine(use_llm_extraction=False)
        stats = engine.get_stats()

        if stats['total_chunks'] == 0:
            console.print("[bold red]Warning:[/bold red] Knowledge base is empty!")
            console.print("Please run: [cyan]python index_documents.py[/cyan]")
            return

        console.print(f"[green]✓ Loaded {stats['total_chunks']} chunks[/green]\n")
    except Exception as e:
        console.print(f"[bold red]Error initializing engine:[/bold red] {e}")
        return

    # Determine mode
    if args.interactive or (not args.query and not args.strategies and not args.concepts and not args.examples):
        interactive_mode(engine)
    elif args.strategies:
        results = engine.search_strategies(args.strategies, max_results=args.max_results)
        display_results(args.strategies, results)
    elif args.concepts:
        results = engine.search_concepts(args.concepts, max_results=args.max_results)
        display_results(args.concepts, results)
    elif args.examples:
        results = engine.search_examples(args.examples, max_results=args.max_results)
        display_results(args.examples, results)
    elif args.query:
        single_query_mode(engine, args.query)


if __name__ == "__main__":
    main()
