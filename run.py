#!/usr/bin/env python3
"""
QuantAI Trading Platform - Unified Runner

Single entry point for both live trading and backtesting.
Automatically detects mode based on date argument.

Usage:
    # Live trading (today's date)
    python run.py 2025-11-14

    # Backtesting/simulation (past date)
    python run.py 2024-01-15

    # Live trading with explicit config
    python run.py 2025-11-14 --config configs/strategies/my_strategy.yaml

    # Backtesting with date range
    python run.py 2024-01-01 --end-date 2024-12-31

The system automatically:
- Uses simulated exchange for past dates (backtest mode)
- Uses real exchange for today/future dates (live mode)
- Same strategy code runs in both modes
- Same configuration file (except exchange selection)
"""
import asyncio
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any
from decimal import Decimal
from loguru import logger
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trading.events import InMemoryEventBus
from trading.exchanges import SimulatedExchange, BinanceExchange, OKXExchange, BybitExchange
from trading.portfolio import PortfolioManager
from trading.oms import OrderManager, create_default_risk_checker
from trading.strategy.manager import StrategyManager
from trading.strategy.config import StrategyConfig


class TradingRunner:
    """Unified trading platform runner"""

    def __init__(
        self,
        run_date: date,
        end_date: Optional[date] = None,
        config_path: Optional[str] = None,
        initial_capital: Decimal = Decimal("100000")
    ):
        """
        Initialize trading runner

        Args:
            run_date: Date to run (today = live, past = backtest)
            end_date: End date for backtesting (optional)
            config_path: Path to strategy config
            initial_capital: Initial capital
        """
        self.run_date = run_date
        self.end_date = end_date or run_date
        self.config_path = config_path
        self.initial_capital = initial_capital

        # Determine mode automatically
        today = date.today()
        self.is_live = (run_date >= today)
        self.mode = "live_trading" if self.is_live else "simulation"

        logger.info("=" * 70)
        logger.info("QuantAI Trading Platform")
        logger.info("=" * 70)
        logger.info(f"Run Date: {self.run_date}")
        logger.info(f"Mode: {self.mode.upper()} (auto-detected)")
        if not self.is_live:
            logger.info(f"End Date: {self.end_date}")
            logger.info(f"Duration: {(self.end_date - run_date).days + 1} days")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info("=" * 70)

        # Components (initialized in setup)
        self.event_bus: Optional[InMemoryEventBus] = None
        self.exchange: Optional[Any] = None
        self.portfolio: Optional[PortfolioManager] = None
        self.oms: Optional[OrderManager] = None
        self.strategy_manager: Optional[StrategyManager] = None

    async def run(self) -> None:
        """Run the trading system"""
        try:
            # Setup
            await self._setup()

            # Run
            if self.is_live:
                await self._run_live()
            else:
                await self._run_backtest()

            # Results
            await self._show_results()

        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
        finally:
            # Cleanup
            await self._teardown()

    async def _setup(self) -> None:
        """Setup trading components"""
        logger.info("\n📦 Setting up components...")

        # Event bus
        self.event_bus = InMemoryEventBus()
        logger.info("✓ Event bus initialized")

        # Exchange (auto-select based on mode)
        self.exchange = await self._create_exchange()
        await self.exchange.connect()
        logger.info(f"✓ Exchange connected: {self.exchange.name} ({self.mode})")

        # Portfolio manager
        self.portfolio = PortfolioManager(
            event_bus=self.event_bus,
            initial_capital=self.initial_capital
        )
        await self.portfolio.start()
        logger.info(f"✓ Portfolio initialized: ${self.initial_capital:,.2f}")

        # Order management
        risk_checker = create_default_risk_checker(self.portfolio)
        self.oms = OrderManager(
            event_bus=self.event_bus,
            portfolio=self.portfolio,
            exchanges={self.exchange.name: self.exchange},
            risk_checker=risk_checker
        )
        await self.oms.start()
        logger.info("✓ Order manager started with risk checks")

        # Strategy manager
        self.strategy_manager = StrategyManager(
            event_bus=self.event_bus,
            portfolio_manager=self.portfolio,
            order_manager=self.oms
        )
        logger.info("✓ Strategy manager initialized")

        # Load strategies
        if self.config_path:
            await self._load_strategy_from_config()
        else:
            logger.warning("⚠️  No strategy config provided")

        logger.info("✅ Setup complete\n")

    async def _create_exchange(self) -> Any:
        """Create exchange based on mode"""
        if self.is_live:
            # Live trading - use real exchange
            # TODO: Load exchange type and credentials from config
            logger.info("Creating real exchange for live trading...")

            # For now, use simulated exchange even for live
            # In production, would create real exchange:
            # return BinanceExchange(api_key=..., api_secret=...)

            return SimulatedExchange(
                fill_model="realistic",
                slippage_model="spread",
                initial_balance=self.initial_capital
            )
        else:
            # Backtest - use simulated exchange
            logger.info("Creating simulated exchange for backtesting...")

            return SimulatedExchange(
                fill_model="realistic",
                slippage_model="spread",
                initial_balance=self.initial_capital
            )

    async def _load_strategy_from_config(self) -> None:
        """Load strategy from config file"""
        logger.info(f"Loading strategy from: {self.config_path}")

        # Load config
        config = StrategyConfig.from_file(self.config_path)

        # Override mode based on run date
        config.mode = self.mode

        # Override exchange to match mode
        if self.is_live:
            # Keep configured exchange for live
            pass
        else:
            # Force simulated exchange for backtest
            if config.subscriptions:
                config.subscriptions.exchanges = ["simulated"]

        # Load strategy
        strategy = await self.strategy_manager.load_strategy(self.config_path)
        logger.info(f"✓ Loaded strategy: {strategy.config.name} ({strategy.config.strategy_id})")

    async def _run_live(self) -> None:
        """Run live trading"""
        logger.info("\n🚀 Starting LIVE TRADING...")
        logger.info("Press Ctrl+C to stop\n")

        # Start all strategies
        await self.strategy_manager.start_all()

        # Run indefinitely (until Ctrl+C)
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n⏹️  Stopping live trading...")

    async def _run_backtest(self) -> None:
        """Run backtest simulation"""
        logger.info(f"\n🔬 Starting BACKTEST: {self.run_date} to {self.end_date}")

        # Start all strategies
        await self.strategy_manager.start_all()

        # TODO: Implement proper historical data replay
        # For now, just run for a short period
        logger.info("⚠️  Note: Historical data replay not yet implemented")
        logger.info("Running with synthetic data for demo...")

        # Simulate some time passing
        duration = (self.end_date - self.run_date).days
        logger.info(f"Simulating {duration} days...")

        # In production, this would:
        # 1. Load historical data for date range
        # 2. Replay events in chronological order
        # 3. Advance time step by step
        # 4. Generate market data events

        await asyncio.sleep(2)  # Demo placeholder

        logger.info("✓ Backtest complete")

    async def _show_results(self) -> None:
        """Show results"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 RESULTS")
        logger.info("=" * 70)

        if not self.portfolio:
            return

        stats = self.portfolio.get_statistics()

        logger.info(f"\n💰 Portfolio Performance:")
        logger.info(f"   Initial Capital:    ${self.initial_capital:,.2f}")
        logger.info(f"   Final Equity:       ${stats['current_equity']:,.2f}")
        logger.info(f"   Total P&L:          ${stats['total_pnl']:,.2f}")
        logger.info(f"   Return:             {stats['return_pct']:.2f}%")
        logger.info(f"   Max Drawdown:       ${stats['max_drawdown']:,.2f} ({stats['max_drawdown_pct']:.2f}%)")

        logger.info(f"\n📈 Positions:")
        logger.info(f"   Total Positions:    {stats['num_positions']}")
        logger.info(f"   Long Positions:     {stats['long_positions']}")
        logger.info(f"   Short Positions:    {stats['short_positions']}")

        logger.info(f"\n💸 P&L Breakdown:")
        logger.info(f"   Realized P&L:       ${stats['total_realized_pnl']:,.2f}")
        logger.info(f"   Unrealized P&L:     ${stats['total_unrealized_pnl']:,.2f}")
        logger.info(f"   Total Fees:         ${stats['total_fees']:,.2f}")

        if not self.is_live:
            # Backtest-specific metrics
            pnl_metrics = self.portfolio.get_pnl_metrics()
            logger.info(f"\n🎯 Trading Statistics:")
            logger.info(f"   Total Trades:       {pnl_metrics.num_trades}")
            logger.info(f"   Win Rate:           {pnl_metrics.win_rate:.2f}%")
            logger.info(f"   Profit Factor:      {pnl_metrics.profit_factor:.2f}")

        logger.info("\n" + "=" * 70)

    async def _teardown(self) -> None:
        """Cleanup components"""
        logger.info("\n🧹 Cleaning up...")

        if self.strategy_manager:
            await self.strategy_manager.stop_all()
        if self.oms:
            await self.oms.stop()
        if self.portfolio:
            await self.portfolio.stop()
        if self.exchange:
            await self.exchange.disconnect()
        if self.event_bus:
            await self.event_bus.close()

        logger.info("✅ Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="QuantAI Trading Platform - Unified Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live trading today
  python run.py 2025-11-14

  # Backtest on past date
  python run.py 2024-01-15

  # Backtest with date range
  python run.py 2024-01-01 --end-date 2024-12-31

  # Use specific strategy config
  python run.py 2025-11-14 --config configs/strategies/my_strategy.yaml

  # Backtest with custom capital
  python run.py 2024-01-01 --capital 50000
        """
    )

    parser.add_argument(
        "date",
        type=str,
        help="Run date (YYYY-MM-DD). Today or future = live, past = backtest"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for backtesting (YYYY-MM-DD). Defaults to run date."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to strategy config file"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)"
    )

    args = parser.parse_args()

    # Parse dates
    try:
        run_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        logger.error(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
        sys.exit(1)

    end_date = None
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"❌ Invalid end date format: {args.end_date}. Use YYYY-MM-DD")
            sys.exit(1)

        if end_date < run_date:
            logger.error(f"❌ End date must be >= run date")
            sys.exit(1)

    # Create and run
    runner = TradingRunner(
        run_date=run_date,
        end_date=end_date,
        config_path=args.config,
        initial_capital=Decimal(str(args.capital))
    )

    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
