#!/usr/bin/env python3
"""
QuantAI CLI - Command-line interface for testing
"""
import asyncio
import sys
from pathlib import Path
from loguru import logger
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_agent.tools.base import global_registry
from ai_agent.tools.trading_tools import (
    GetMarketDataTool,
    CalculateIndicatorTool,
    GenerateSignalsTool,
    CalculateMetricsTool
)
from ai_agent.reasoner.llm_providers import LLMProviderFactory
from shared.config.settings import settings


app = typer.Typer(help="QuantAI CLI - AI-Human Collaborative Quant Research Platform")
console = Console()


@app.command()
def info():
    """Display system information"""
    console.print(Panel.fit(
        f"[bold blue]{settings.app_name}[/bold blue] v{settings.app_version}\n"
        f"Environment: {settings.environment}\n"
        f"Default LLM: {settings.llm.default_llm_provider}/{settings.llm.default_model}",
        title="System Information"
    ))


@app.command()
def list_tools():
    """List all available AI tools"""
    # Register tools
    global_registry.register(GetMarketDataTool())
    global_registry.register(CalculateIndicatorTool())
    global_registry.register(GenerateSignalsTool())
    global_registry.register(CalculateMetricsTool())

    tools = global_registry.get_all_tools()

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="green")

    for tool in tools:
        table.add_row(
            tool.name,
            tool.category or "general",
            tool.description
        )

    console.print(table)


@app.command()
def test_market_data(
    symbol: str = typer.Argument(..., help="Stock symbol (e.g., AAPL)"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)")
):
    """Test market data fetching"""
    async def run():
        console.print(f"[cyan]Fetching market data for {symbol}...[/cyan]")

        tool = GetMarketDataTool()
        result = await tool.execute(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if result.success:
            data = result.result
            console.print(Panel.fit(
                f"[green]✓ Success![/green]\n"
                f"Symbol: {data.symbol}\n"
                f"Period: {data.start_date} to {data.end_date}\n"
                f"Bars: {data.num_bars}\n"
                f"Latest Close: ${data.data['close'][-1]:.2f}",
                title="Market Data"
            ))

            # Show first few prices
            console.print("\n[bold]First 5 closes:[/bold]")
            for i in range(min(5, len(data.data['close']))):
                console.print(f"  {data.data['dates'][i]}: ${data.data['close'][i]:.2f}")
        else:
            console.print(f"[red]✗ Error: {result.error}[/red]")

    asyncio.run(run())


@app.command()
def test_strategy(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    strategy: str = typer.Option("MA_CROSSOVER", help="Strategy type"),
    start_date: str = typer.Option("2023-01-01", help="Start date"),
    end_date: str = typer.Option("2024-01-01", help="End date")
):
    """Test a trading strategy"""
    async def run():
        console.print(f"[cyan]Testing {strategy} strategy on {symbol}...[/cyan]")

        # 1. Fetch data
        console.print("[yellow]1. Fetching market data...[/yellow]")
        data_tool = GetMarketDataTool()
        data_result = await data_tool.execute(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if not data_result.success:
            console.print(f"[red]✗ Data fetch failed: {data_result.error}[/red]")
            return

        prices = data_result.result.data["close"]
        console.print(f"[green]✓ Fetched {len(prices)} price bars[/green]")

        # 2. Generate signals
        console.print("[yellow]2. Generating trading signals...[/yellow]")
        signal_tool = GenerateSignalsTool()
        signal_result = await signal_tool.execute(
            strategy=strategy,
            prices=prices
        )

        if not signal_result.success:
            console.print(f"[red]✗ Signal generation failed: {signal_result.error}[/red]")
            return

        signals = signal_result.result["signals"]
        console.print(f"[green]✓ Generated signals: {signal_result.result['num_buy']} buy, "
                     f"{signal_result.result['num_sell']} sell[/green]")

        # 3. Calculate returns
        console.print("[yellow]3. Calculating returns...[/yellow]")
        returns = []
        for i in range(1, len(prices)):
            if signals[i-1] == 1:  # Long
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            elif signals[i-1] == -1:  # Short
                ret = (prices[i-1] - prices[i]) / prices[i-1]
                returns.append(ret)

        if returns:
            # 4. Calculate metrics
            console.print("[yellow]4. Calculating performance metrics...[/yellow]")
            metrics_tool = CalculateMetricsTool()
            metrics_result = await metrics_tool.execute(returns=returns)

            if metrics_result.success:
                metrics = metrics_result.result

                console.print(Panel.fit(
                    f"[bold green]Strategy Performance[/bold green]\n\n"
                    f"Total Return: {metrics['total_return']:.2%}\n"
                    f"Annual Return: {metrics['annualized_return']:.2%}\n"
                    f"Volatility: {metrics['volatility']:.2%}\n"
                    f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                    f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                    f"Win Rate: {metrics['win_rate']:.2%}\n"
                    f"Trades: {metrics['num_trades']}",
                    title=f"{strategy} on {symbol}"
                ))
            else:
                console.print(f"[red]✗ Metrics calculation failed: {metrics_result.error}[/red]")
        else:
            console.print("[yellow]⚠ No trades generated[/yellow]")

    asyncio.run(run())


@app.command()
def chat(
    message: str = typer.Option(None, "--message", "-m", help="Message to send"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode")
):
    """Chat with AI assistant"""
    async def run():
        try:
            # Register tools
            global_registry.register(GetMarketDataTool())
            global_registry.register(CalculateIndicatorTool())
            global_registry.register(GenerateSignalsTool())
            global_registry.register(CalculateMetricsTool())

            # Create LLM provider
            console.print("[cyan]Initializing AI assistant...[/cyan]")
            llm = LLMProviderFactory.create_default()

            if interactive:
                console.print(Panel.fit(
                    "[bold]Interactive Chat Mode[/bold]\n"
                    "Type 'exit' or 'quit' to end the conversation",
                    title="QuantAI Chat"
                ))

                while True:
                    user_message = console.input("[bold blue]You:[/bold blue] ")

                    if user_message.lower() in ['exit', 'quit', 'q']:
                        console.print("[yellow]Goodbye![/yellow]")
                        break

                    console.print("[cyan]AI is thinking...[/cyan]")

                    # Get available tools
                    tools = global_registry.get_openai_functions()

                    if tools:
                        # Use function calling
                        response, function_call = await llm.complete_with_functions(
                            user_message,
                            tools
                        )

                        if function_call:
                            console.print(f"[yellow]→ Calling tool: {function_call['name']}[/yellow]")

                            # Execute the function
                            tool_result = await global_registry.execute_tool(
                                function_call["name"],
                                **function_call["arguments"]
                            )

                            # Generate final response
                            final_prompt = f"""
Original query: {user_message}

Tool called: {function_call['name']}
Tool result: {tool_result.result}

Please provide a helpful response to the user.
"""
                            response = await llm.complete(final_prompt)

                    else:
                        response = await llm.complete(user_message)

                    console.print(f"[bold green]AI:[/bold green] {response}\n")

            elif message:
                console.print(f"[bold blue]You:[/bold blue] {message}")
                console.print("[cyan]AI is thinking...[/cyan]")

                response = await llm.complete(message)

                console.print(f"[bold green]AI:[/bold green] {response}")
            else:
                console.print("[red]Please provide a message with -m or use -i for interactive mode[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("Chat error")

    asyncio.run(run())


@app.command()
def start_api(
    host: str = typer.Option("0.0.0.0", help="API host"),
    port: int = typer.Option(8000, help="API port"),
    reload: bool = typer.Option(True, help="Enable auto-reload")
):
    """Start the FastAPI server"""
    console.print(Panel.fit(
        f"[bold]Starting API Server[/bold]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Auto-reload: {reload}\n\n"
        f"Open http://{host}:{port}/docs for API documentation",
        title="QuantAI API"
    ))

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    app()
