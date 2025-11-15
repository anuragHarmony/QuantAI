"""
Demo: Autonomous AI Research Agent

This script demonstrates how to use the AI research agent to autonomously
discover profitable trading strategies.

The agent will:
1. Query the knowledge graph for trading concepts
2. Analyze market data and detect patterns
3. Generate strategy hypotheses using LLM
4. Convert hypotheses to executable Python code
5. Run backtests to evaluate strategies
6. Reflect on results and learn
7. Iterate to find better strategies

Usage:
    export ANTHROPIC_API_KEY=your_api_key_here
    python examples/demo_research_agent.py
"""
import asyncio
import os
from datetime import datetime
from loguru import logger

from research.agent.agent_factory import AgentFactory
from research.agent.core.orchestrator import ResearchConfig


async def demo_simple_research():
    """
    Simple demo: Run a few research iterations
    """
    logger.info("=" * 80)
    logger.info("DEMO: Simple Autonomous Research")
    logger.info("=" * 80)

    # Create agent with simple configuration
    agent = await AgentFactory.create_simple_agent()

    # Run research
    session = await agent.run_research_loop(
        objective="Discover profitable mean reversion strategies for BTC",
        focus="mean reversion",
        max_iterations=5
    )

    # Get best strategies
    best_strategies = await agent.get_best_strategies(top_k=3)

    logger.info("\n" + "=" * 80)
    logger.info("BEST STRATEGIES FOUND")
    logger.info("=" * 80)

    for i, strategy in enumerate(best_strategies, 1):
        logger.info(f"\n{i}. {strategy.hypothesis.get('description', 'N/A')}")
        logger.info(f"   Sharpe: {strategy.results.get('sharpe_ratio', 'N/A'):.2f}")
        logger.info(f"   Win Rate: {strategy.results.get('win_rate', 'N/A'):.1%}")
        logger.info(f"   Indicators: {', '.join(strategy.hypothesis.get('indicators', []))}")

    # Generate report
    report = await agent.generate_research_report(session)

    # Save report
    report_path = f"research_report_{session.id}.md"
    with open(report_path, 'w') as f:
        f.write(report)

    logger.success(f"\nResearch report saved to: {report_path}")


async def demo_advanced_research():
    """
    Advanced demo: Custom configuration with more iterations
    """
    logger.info("=" * 80)
    logger.info("DEMO: Advanced Autonomous Research")
    logger.info("=" * 80)

    # Custom configuration
    config = ResearchConfig(
        max_iterations=10,
        min_sharpe_threshold=1.5,
        diversity_threshold=0.8,
        backtest_symbol="BTC/USDT",
        backtest_timeframe="1h",
        backtest_start_date="2024-01-01",
        backtest_end_date="2024-06-01",
        learning_interval=3  # Extract insights every 3 experiments
    )

    # Create agent
    agent = await AgentFactory.create_research_agent(config=config)

    # Run research
    session = await agent.run_research_loop(
        objective="Discover high Sharpe ratio strategies across different market regimes",
        max_iterations=10
    )

    # Get best strategies
    best_strategies = await agent.get_best_strategies(
        top_k=5,
        min_sharpe=1.5
    )

    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 STRATEGIES")
    logger.info("=" * 80)

    for i, strategy in enumerate(best_strategies, 1):
        logger.info(f"\n### Strategy {i}")
        logger.info(f"Description: {strategy.hypothesis.get('description', 'N/A')}")
        logger.info(f"Rationale: {strategy.hypothesis.get('rationale', 'N/A')}")
        logger.info(f"\nPerformance:")
        logger.info(f"  - Sharpe Ratio: {strategy.results.get('sharpe_ratio', 'N/A'):.2f}")
        logger.info(f"  - Total Return: {strategy.results.get('total_return', 'N/A'):.1f}%")
        logger.info(f"  - Win Rate: {strategy.results.get('win_rate', 'N/A'):.1%}")
        logger.info(f"  - Total Trades: {strategy.results.get('total_trades', 'N/A')}")
        logger.info(f"\nStrategy Details:")
        logger.info(f"  - Indicators: {', '.join(strategy.hypothesis.get('indicators', []))}")
        logger.info(f"  - Entry: {strategy.hypothesis.get('entry_logic', 'N/A')}")
        logger.info(f"  - Exit: {strategy.hypothesis.get('exit_logic', 'N/A')}")

    # Generate and save report
    report = await agent.generate_research_report(session)
    report_path = f"research_report_{session.id}.md"

    with open(report_path, 'w') as f:
        f.write(report)

    logger.success(f"\nResearch report saved to: {report_path}")


async def demo_hypothesis_generation():
    """
    Demo: Just generate hypotheses without running full research loop
    """
    logger.info("=" * 80)
    logger.info("DEMO: Hypothesis Generation")
    logger.info("=" * 80)

    # Create agent
    agent = await AgentFactory.create_simple_agent()

    # Generate batch of diverse hypotheses
    hypotheses = await agent.hypothesis_generator.generate_batch(
        count=5,
        diversity=0.8,
        focus="momentum strategies"
    )

    logger.info("\n" + "=" * 80)
    logger.info("GENERATED HYPOTHESES")
    logger.info("=" * 80)

    for i, hyp in enumerate(hypotheses, 1):
        logger.info(f"\n### Hypothesis {i}")
        logger.info(f"Description: {hyp.description}")
        logger.info(f"Rationale: {hyp.rationale}")
        logger.info(f"Indicators: {', '.join(hyp.indicators)}")
        logger.info(f"Entry Logic: {hyp.entry_logic}")
        logger.info(f"Exit Logic: {hyp.exit_logic}")
        logger.info(f"Target Regime: {hyp.target_regime}")
        logger.info(f"Expected Sharpe: {hyp.expected_sharpe}")


async def demo_code_generation():
    """
    Demo: Generate code from a hypothesis
    """
    logger.info("=" * 80)
    logger.info("DEMO: Code Generation")
    logger.info("=" * 80)

    # Create agent
    agent = await AgentFactory.create_simple_agent()

    # Generate a hypothesis
    logger.info("Generating hypothesis...")
    hypothesis = await agent.hypothesis_generator.generate_hypothesis(
        focus="RSI mean reversion"
    )

    logger.info(f"\nHypothesis: {hypothesis.description}")

    # Generate code
    logger.info("\nGenerating code...")
    generated = await agent.code_generator.generate_strategy_code(hypothesis)

    if generated.is_valid:
        logger.success(f"\nCode generation successful!")
        logger.info(f"Class name: {generated.class_name}")
        logger.info(f"\nGenerated code:\n")
        logger.info("=" * 80)
        print(generated.code)
        logger.info("=" * 80)

        # Save code
        code_path = f"generated_{generated.class_name}.py"
        await agent.code_generator.save_to_file(generated, code_path)
        logger.success(f"\nCode saved to: {code_path}")

    else:
        logger.error(f"Code generation failed: {generated.validation_errors}")


async def main():
    """Main demo selector"""
    print("\n" + "=" * 80)
    print("AI RESEARCH AGENT DEMOS")
    print("=" * 80)
    print("\nAvailable demos:")
    print("1. Simple Research (5 iterations)")
    print("2. Advanced Research (10 iterations)")
    print("3. Hypothesis Generation Only")
    print("4. Code Generation Demo")
    print("5. Run All Demos")
    print("\nNote: Demos use mock backtest results for demonstration.")
    print("In production, integrate with your backtesting framework.")

    choice = input("\nSelect demo (1-5): ").strip()

    if choice == "1":
        await demo_simple_research()
    elif choice == "2":
        await demo_advanced_research()
    elif choice == "3":
        await demo_hypothesis_generation()
    elif choice == "4":
        await demo_code_generation()
    elif choice == "5":
        logger.info("Running all demos...")
        await demo_hypothesis_generation()
        await demo_code_generation()
        await demo_simple_research()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n" + "=" * 80)
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("=" * 80)
        print("\nPlease set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY=your_api_key_here")
        print("\nOr pass it when creating the agent:")
        print("  agent = await AgentFactory.create_simple_agent(anthropic_api_key='...')")
        print()
        exit(1)

    # Run demos
    asyncio.run(main())
