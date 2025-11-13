"""
Interactive RAG-powered Q&A with the knowledge base.

This script uses OpenAI's LLM to generate answers based on retrieved context
from the knowledge base.

Usage:
    python ask_ai.py "your question here"
    python ask_ai.py --interactive  # Interactive chat mode
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_engine import KnowledgeEngine
from ai_agent.reasoner import RAGPipeline
from shared.config.settings import settings


console = Console()


def display_answer(result: dict):
    """Display AI-generated answer with sources."""
    console.print()

    # Display answer
    answer_md = Markdown(result["answer"])
    panel = Panel(
        answer_md,
        title="[bold cyan]AI Answer[/bold cyan]",
        border_style="cyan",
        expand=False
    )
    console.print(panel)

    # Display metadata
    console.print()
    console.print(f"[dim]Model: {result['model']}[/dim]")
    console.print(f"[dim]Sources used: {result['num_sources']}[/dim]")
    console.print(f"[dim]Confidence: {result['confidence']:.3f}[/dim]")
    console.print(f"[dim]Context tokens: {result['context_tokens']}[/dim]")

    # Display sources
    if result.get("sources"):
        console.print()
        console.print("[bold yellow]Sources:[/bold yellow]")
        for i, source in enumerate(result["sources"][:5], 1):
            console.print(f"  {i}. [cyan]{source['book']}[/cyan] ({source['type']}) - Relevance: {source['relevance']:.3f}")


def interactive_mode(rag_pipeline: RAGPipeline):
    """Run in interactive Q&A mode."""
    console.print(Panel.fit(
        "[bold cyan]QuantAI RAG Assistant - Interactive Mode[/bold cyan]\n"
        "Ask questions about quantitative trading, strategies, risk management, and more.\n"
        "Powered by OpenAI + Knowledge Base Retrieval\n\n"
        "Commands: 'exit', 'quit', 'help'",
        border_style="blue"
    ))

    conversation_history = []

    while True:
        console.print()
        question = Prompt.ask("[bold yellow]Your question[/bold yellow]")

        if not question:
            continue

        question_lower = question.lower().strip()

        # Handle commands
        if question_lower in ['exit', 'quit', 'q']:
            console.print("[cyan]Goodbye![/cyan]")
            break

        elif question_lower == 'help':
            console.print(Panel(
                "[bold]Available Commands[/bold]\n\n"
                "[cyan]exit/quit[/cyan] - Exit the program\n"
                "[cyan]help[/cyan] - Show this help message\n\n"
                "[bold]Example Questions[/bold]\n\n"
                "- What is mean reversion and how can I trade it?\n"
                "- Explain the Sharpe ratio and how to calculate it\n"
                "- How do I implement a pairs trading strategy?\n"
                "- What are the best risk management practices?\n"
                "- Compare momentum vs mean reversion strategies",
                border_style="cyan"
            ))

        else:
            # Process question
            with console.status("[bold green]Thinking...", spinner="dots"):
                try:
                    # Add to conversation history
                    conversation_history.append({"role": "user", "content": question})

                    # Get answer (with retrieval for first message, or use chat for multi-turn)
                    if len(conversation_history) == 1:
                        result = rag_pipeline.ask(question, include_citations=True)
                    else:
                        result = rag_pipeline.chat(conversation_history, include_retrieval=True)

                    # Add assistant response to history
                    conversation_history.append({"role": "assistant", "content": result["answer"]})

                    # Display answer
                    display_answer(result)

                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {e}")
                    logger.error(f"Error processing question: {e}", exc_info=True)


def single_question_mode(rag_pipeline: RAGPipeline, question: str):
    """Answer a single question and exit."""
    console.print(f"[bold cyan]Question:[/bold cyan] {question}\n")

    with console.status("[bold green]Searching knowledge base and generating answer...", spinner="dots"):
        try:
            result = rag_pipeline.ask(question, include_citations=True)
            display_answer(result)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ask questions about quantitative trading using AI + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ask_ai.py "What is mean reversion?"
  python ask_ai.py "How do I calculate the Sharpe ratio?"
  python ask_ai.py --interactive
        """
    )

    parser.add_argument(
        'question',
        nargs='?',
        help='Question to ask (if not provided, enters interactive mode)'
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--no-citations',
        action='store_true',
        help='Disable source citations in responses'
    )

    args = parser.parse_args()

    # Check for API key
    if not settings.openai_api_key:
        console.print("[bold red]Error:[/bold red] OpenAI API key not found!")
        console.print("Please set OPENAI_API_KEY in your .env file:")
        console.print("  OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    # Initialize knowledge engine
    console.print("[bold blue]Initializing Knowledge Engine...[/bold blue]")
    try:
        engine = KnowledgeEngine(use_llm_extraction=False)
        stats = engine.get_stats()

        if stats['total_chunks'] == 0:
            console.print("[bold yellow]Warning:[/bold yellow] Knowledge base is empty!")
            console.print("Please run: [cyan]python index_documents.py[/cyan]")
            console.print("The AI will answer without specific document context.\n")
            engine = None
        else:
            console.print(f"[green]✓ Loaded {stats['total_chunks']} knowledge chunks[/green]")

    except Exception as e:
        console.print(f"[yellow]Warning: Could not load knowledge base: {e}[/yellow]")
        console.print("The AI will answer without specific document context.\n")
        engine = None

    # Initialize RAG pipeline
    console.print("[bold blue]Initializing RAG Pipeline...[/bold blue]")
    try:
        rag_pipeline = RAGPipeline(knowledge_engine=engine)
        console.print(f"[green]✓ RAG Pipeline ready (Model: {settings.llm_model})[/green]\n")
    except Exception as e:
        console.print(f"[bold red]Error initializing RAG pipeline:[/bold red] {e}")
        sys.exit(1)

    # Determine mode
    if args.interactive or not args.question:
        interactive_mode(rag_pipeline)
    else:
        single_question_mode(rag_pipeline, args.question)


if __name__ == "__main__":
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # Only show warnings and errors

    main()
