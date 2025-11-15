"""
PnL Recorder Example

Demonstrates automatic P&L file recording during live trading.

The PnLRecorder:
1. Subscribes to portfolio value events
2. Records snapshots to CSV/JSON every N seconds
3. Creates timestamped files per session
4. Useful for monitoring and post-trade analysis

Output files are created in ./data/pnl/ by default.
"""

import asyncio
from decimal import Decimal
from loguru import logger

from trading.events.bus import InMemoryEventBus
from trading.portfolio import PortfolioManager, PnLRecorder, create_pnl_recorder
from trading.events.position import PortfolioValueEvent


async def example_basic_usage():
    """Basic PnL recorder usage"""

    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic PnL Recording")
    logger.info("=" * 80)

    # Setup
    event_bus = InMemoryEventBus()
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=Decimal("10000"),
        update_interval=1.0  # Update every second
    )

    # Create PnL recorder - dumps every 10 seconds
    pnl_recorder = PnLRecorder(
        event_bus=event_bus,
        dump_interval=10.0,  # Dump to file every 10 seconds
        output_dir="./data/pnl",
        format="csv",  # CSV format
    )

    # Start both
    await portfolio.start()
    await pnl_recorder.start()

    logger.info(f"PnL will be recorded to: {pnl_recorder.get_filepath()}")

    # Simulate trading for 35 seconds
    logger.info("Simulating trading for 35 seconds...")
    logger.info("PnL will be dumped at t=10s, t=20s, t=30s, and on stop")

    await asyncio.sleep(35)

    # Stop
    await pnl_recorder.stop()  # Performs final dump
    await portfolio.stop()

    # Show stats
    stats = pnl_recorder.get_statistics()
    logger.info(f"\nRecording statistics:")
    logger.info(f"  Session: {stats['session_name']}")
    logger.info(f"  File: {stats['filepath']}")
    logger.info(f"  Snapshots recorded: {stats['snapshots_recorded']}")

    logger.success("✓ Example 1 complete - check ./data/pnl/ for output file\n")


async def example_json_format():
    """PnL recording with JSON format"""

    logger.info("=" * 80)
    logger.info("EXAMPLE 2: JSON Format Recording")
    logger.info("=" * 80)

    event_bus = InMemoryEventBus()
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=Decimal("50000"),
        update_interval=1.0
    )

    # Use JSON Lines format
    pnl_recorder = PnLRecorder(
        event_bus=event_bus,
        dump_interval=5.0,  # Faster dumps for demo
        output_dir="./data/pnl",
        format="json",  # JSON Lines format
        session_name="json_example_session"  # Custom session name
    )

    await portfolio.start()
    await pnl_recorder.start()

    logger.info(f"Recording to JSON: {pnl_recorder.get_filepath()}")
    logger.info("Simulating trading for 15 seconds...")

    await asyncio.sleep(15)

    await pnl_recorder.stop()
    await portfolio.stop()

    # Show recorded snapshots
    snapshots = pnl_recorder.get_snapshots()
    logger.info(f"\nRecorded {len(snapshots)} snapshots")

    if snapshots:
        latest = snapshots[-1]
        logger.info(f"Latest snapshot:")
        logger.info(f"  Timestamp: {latest['timestamp']}")
        logger.info(f"  Total Value: {latest['total_value']}")
        logger.info(f"  Total PnL: {latest['total_pnl']}")

    logger.success("✓ Example 2 complete\n")


async def example_convenience_function():
    """Using the convenience function"""

    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Convenience Function")
    logger.info("=" * 80)

    event_bus = InMemoryEventBus()
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=Decimal("25000")
    )

    # Quick setup using convenience function
    pnl_recorder = create_pnl_recorder(
        event_bus=event_bus,
        dump_interval=3.0,  # Fast dumps for demo
        format="csv"
    )

    await portfolio.start()
    await pnl_recorder.start()

    logger.info(f"Quick setup complete: {pnl_recorder.get_filepath()}")
    logger.info("Running for 10 seconds...")

    await asyncio.sleep(10)

    await pnl_recorder.stop()
    await portfolio.stop()

    logger.success("✓ Example 3 complete\n")


async def example_with_simulated_pnl():
    """Simulate actual PnL changes"""

    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Simulated PnL Changes")
    logger.info("=" * 80)

    event_bus = InMemoryEventBus()
    portfolio = PortfolioManager(
        event_bus=event_bus,
        initial_capital=Decimal("100000"),
        update_interval=0.5
    )

    pnl_recorder = PnLRecorder(
        event_bus=event_bus,
        dump_interval=2.0,  # Dump every 2 seconds
        output_dir="./data/pnl",
        format="csv",
        session_name="simulated_pnl_session"
    )

    await portfolio.start()
    await pnl_recorder.start()

    logger.info("Simulating P&L changes...")

    # Manually publish portfolio value events to simulate trading
    for i in range(10):
        # Simulate gradually changing PnL
        pnl = Decimal(i * 100 - 500)  # Start -500, end +400
        total_value = Decimal("100000") + pnl

        event = PortfolioValueEvent(
            exchange="simulation",
            symbol="PORTFOLIO",
            cash=Decimal("100000"),
            equity=Decimal("0"),
            total_value=total_value,
            unrealized_pnl=pnl,
            realized_pnl=Decimal("0"),
            total_pnl=pnl,
            return_pct=pnl / Decimal("100000") * Decimal("100"),
            initial_value=Decimal("100000"),
        )

        await event_bus.publish(event)

        logger.info(f"  t={i}s: PnL = {pnl:+.2f}, Return = {event.return_pct:+.2f}%")

        await asyncio.sleep(1)

    await asyncio.sleep(2)  # Let final dump happen

    await pnl_recorder.stop()
    await portfolio.stop()

    # Verify snapshots
    snapshots = pnl_recorder.get_snapshots()
    logger.info(f"\n✓ Recorded {len(snapshots)} PnL snapshots")

    if snapshots:
        logger.info("\nPnL progression:")
        for snap in snapshots:
            logger.info(
                f"  {snap['timestamp'].strftime('%H:%M:%S')} | "
                f"PnL: {snap['total_pnl']:>10} | "
                f"Return: {snap['return_pct']:>6.2f}%"
            )

    logger.success(f"✓ Example 4 complete - check {pnl_recorder.get_filepath()}\n")


async def main():
    """Run all examples"""

    logger.info("\n🎯 PnL Recorder Examples\n")

    try:
        # Example 1: Basic usage
        await example_basic_usage()

        # Example 2: JSON format
        await example_json_format()

        # Example 3: Convenience function
        await example_convenience_function()

        # Example 4: Simulated PnL
        await example_with_simulated_pnl()

        logger.success("\n" + "=" * 80)
        logger.success("✓ All examples completed successfully!")
        logger.success("=" * 80)
        logger.info("\nCheck ./data/pnl/ directory for output files")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True
    )

    asyncio.run(main())
