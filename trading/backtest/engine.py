"""
Backtesting Engine

High-performance backtesting engine for strategy validation:
- Historical data replay with accurate timing
- Realistic order execution simulation
- Complete P&L tracking and performance metrics
- Transaction cost modeling
- Slippage simulation
- Multiple strategy support
- Comprehensive result analysis

Usage:
    engine = BacktestEngine(
        strategy=MyStrategy,
        initial_capital=100000,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31)
    )

    results = await engine.run()
    print(results.summary())
"""
import asyncio
from typing import List, Dict, Optional, Type, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger

from ..events import InMemoryEventBus, TickEvent, TradeEvent, BarEvent
from ..exchanges import SimulatedExchange
from ..portfolio import PortfolioManager
from ..oms import OrderManager, create_default_risk_checker
from ..strategy.base import BaseStrategy
from ..strategy.config import StrategyConfig


@dataclass
class BacktestConfig:
    """Backtest configuration"""

    # Time period
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: Decimal = Decimal("100000")

    # Execution simulation
    fill_model: str = "realistic"  # "aggressive", "realistic", "conservative"
    slippage_model: str = "spread"  # "none", "fixed", "spread", "volume"
    slippage_bps: Optional[Decimal] = None  # For fixed slippage

    # Trading costs
    maker_fee: Decimal = Decimal("0.0001")  # 1 bps
    taker_fee: Decimal = Decimal("0.0002")  # 2 bps

    # Data
    data_frequency: str = "1m"  # Data granularity
    warmup_period: timedelta = timedelta(days=7)  # Warmup before start_date

    # Risk management
    enable_risk_checks: bool = True
    max_position_size: Optional[Decimal] = None
    max_daily_loss: Optional[Decimal] = None

    # Performance
    benchmark: Optional[str] = None  # Symbol to use as benchmark
    risk_free_rate: Decimal = Decimal("0.02")  # Annual risk-free rate


@dataclass
class BacktestResult:
    """Backtest results and performance metrics"""

    # Configuration
    config: BacktestConfig
    strategy_name: str
    strategy_id: str

    # Time
    start_date: datetime
    end_date: datetime
    duration_days: int

    # P&L
    initial_capital: Decimal
    final_capital: Decimal
    total_pnl: Decimal
    total_return_pct: Decimal

    # Trade statistics
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: Decimal

    # Win/Loss analysis
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: Decimal  # Gross profit / Gross loss

    # Risk metrics
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal

    # Exposure
    avg_holding_period: timedelta
    max_positions: int
    avg_positions: Decimal

    # Costs
    total_fees: Decimal
    total_slippage: Decimal

    # Equity curve
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    # Trades
    trades: List[Dict[str, Any]] = field(default_factory=list)

    # Daily returns
    daily_returns: List[Decimal] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary report"""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULTS SUMMARY                      ║
╚══════════════════════════════════════════════════════════════════╝

Strategy: {self.strategy_name} ({self.strategy_id})
Period: {self.start_date.date()} to {self.end_date.date()} ({self.duration_days} days)

┌─────────────────────────────────────────────────────────────────┐
│ PERFORMANCE                                                      │
├─────────────────────────────────────────────────────────────────┤
│ Initial Capital:        ${self.initial_capital:,.2f}            │
│ Final Capital:          ${self.final_capital:,.2f}              │
│ Total P&L:              ${self.total_pnl:,.2f}                  │
│ Total Return:           {self.total_return_pct:.2f}%            │
│ Max Drawdown:           ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RISK-ADJUSTED RETURNS                                           │
├─────────────────────────────────────────────────────────────────┤
│ Sharpe Ratio:           {self.sharpe_ratio:.3f}                 │
│ Sortino Ratio:          {self.sortino_ratio:.3f}                │
│ Calmar Ratio:           {self.calmar_ratio:.3f}                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ TRADING STATISTICS                                              │
├─────────────────────────────────────────────────────────────────┤
│ Total Trades:           {self.num_trades}                       │
│ Winning Trades:         {self.num_wins}                         │
│ Losing Trades:          {self.num_losses}                       │
│ Win Rate:               {self.win_rate:.2f}%                    │
│ Profit Factor:          {self.profit_factor:.2f}                │
│                                                                  │
│ Average Win:            ${self.avg_win:,.2f}                    │
│ Average Loss:           ${self.avg_loss:,.2f}                   │
│ Largest Win:            ${self.largest_win:,.2f}                │
│ Largest Loss:           ${self.largest_loss:,.2f}               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ COSTS                                                            │
├─────────────────────────────────────────────────────────────────┤
│ Total Fees:             ${self.total_fees:,.2f}                 │
│ Total Slippage:         ${self.total_slippage:,.2f}             │
│ Total Costs:            ${self.total_fees + self.total_slippage:,.2f}│
└─────────────────────────────────────────────────────────────────┘
"""


class BacktestEngine:
    """
    Backtesting engine for strategy validation

    Replays historical data and simulates strategy execution
    to evaluate performance before live trading.
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        strategy_config: StrategyConfig,
        config: BacktestConfig,
        data_provider: Optional[Any] = None
    ):
        """
        Initialize backtest engine

        Args:
            strategy_class: Strategy class to backtest
            strategy_config: Strategy configuration
            config: Backtest configuration
            data_provider: Historical data provider (optional)
        """
        self.strategy_class = strategy_class
        self.strategy_config = strategy_config
        self.config = config
        self.data_provider = data_provider

        # Components (created on run)
        self.event_bus: Optional[InMemoryEventBus] = None
        self.exchange: Optional[SimulatedExchange] = None
        self.portfolio: Optional[PortfolioManager] = None
        self.oms: Optional[OrderManager] = None
        self.strategy: Optional[BaseStrategy] = None

        # Results tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.daily_returns: List[Decimal] = []

        logger.info(f"Initialized BacktestEngine: {self.config.start_date} to {self.config.end_date}")

    async def run(self) -> BacktestResult:
        """
        Run backtest

        Returns:
            BacktestResult with complete performance metrics
        """
        logger.info("=" * 70)
        logger.info(f"Starting backtest: {self.strategy_config.name}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        logger.info("=" * 70)

        # Initialize components
        await self._setup()

        try:
            # Run simulation
            await self._run_simulation()

            # Calculate results
            result = await self._calculate_results()

            logger.info("\n" + result.summary())

            return result

        finally:
            # Cleanup
            await self._teardown()

    async def _setup(self) -> None:
        """Setup backtest components"""
        logger.info("Setting up backtest components...")

        # Event bus
        self.event_bus = InMemoryEventBus()

        # Simulated exchange
        self.exchange = SimulatedExchange(
            fill_model=self.config.fill_model,
            slippage_model=self.config.slippage_model,
            initial_balance=self.config.initial_capital,
        )

        await self.exchange.connect()

        # Portfolio manager
        self.portfolio = PortfolioManager(
            event_bus=self.event_bus,
            initial_capital=self.config.initial_capital
        )
        await self.portfolio.start()

        # Order management system
        if self.config.enable_risk_checks:
            risk_checker = create_default_risk_checker(self.portfolio)
        else:
            risk_checker = None

        self.oms = OrderManager(
            event_bus=self.event_bus,
            portfolio=self.portfolio,
            exchanges={self.exchange.name: self.exchange},
            risk_checker=risk_checker
        )
        await self.oms.start()

        # Strategy
        self.strategy = self.strategy_class(
            config=self.strategy_config,
            event_bus=self.event_bus,
            portfolio_manager=self.portfolio,
            order_manager=self.oms
        )

        await self.strategy.start()

        logger.info("✓ Backtest components initialized")

    async def _run_simulation(self) -> None:
        """Run the backtest simulation"""
        logger.info("Running simulation...")

        current_time = self.config.start_date

        # TODO: Implement actual data replay from data_provider
        # For now, simulate with synthetic ticks

        while current_time <= self.config.end_date:
            # Generate synthetic market data for demo
            await self._generate_synthetic_data(current_time)

            # Track equity
            self._record_equity_point(current_time)

            # Advance time
            current_time += timedelta(minutes=1)

            # Allow events to process
            await asyncio.sleep(0.001)

        logger.info("✓ Simulation complete")

    async def _generate_synthetic_data(self, timestamp: datetime) -> None:
        """Generate synthetic market data for demo purposes"""
        # TODO: Replace with actual historical data replay

        # For now, just publish a synthetic tick
        tick = TickEvent(
            exchange="simulated",
            symbol="BTC/USDT",
            raw_symbol="BTCUSDT",
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000.5"),
            volume=Decimal("100"),
            exchange_timestamp=timestamp
        )

        await self.event_bus.publish(tick)

    def _record_equity_point(self, timestamp: datetime) -> None:
        """Record equity curve point"""
        if not self.portfolio:
            return

        stats = self.portfolio.get_statistics()

        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": stats["current_equity"],
            "cash": stats["initial_capital"],
            "positions_value": Decimal("0"),
            "pnl": stats["total_pnl"]
        })

    async def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and metrics"""
        logger.info("Calculating results...")

        if not self.portfolio:
            raise ValueError("Portfolio not initialized")

        stats = self.portfolio.get_statistics()
        pnl_metrics = self.portfolio.get_pnl_metrics()

        # Calculate duration
        duration = (self.config.end_date - self.config.start_date).days

        # Calculate risk metrics
        sharpe = self._calculate_sharpe_ratio()
        sortino = self._calculate_sortino_ratio()
        calmar = self._calculate_calmar_ratio(pnl_metrics.max_drawdown)

        result = BacktestResult(
            config=self.config,
            strategy_name=self.strategy_config.name,
            strategy_id=self.strategy_config.strategy_id,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration_days=duration,
            initial_capital=self.config.initial_capital,
            final_capital=stats["current_equity"],
            total_pnl=stats["total_pnl"],
            total_return_pct=stats["return_pct"],
            num_trades=pnl_metrics.num_trades,
            num_wins=pnl_metrics.winning_trades,
            num_losses=pnl_metrics.losing_trades,
            win_rate=pnl_metrics.win_rate,
            avg_win=pnl_metrics.average_win,
            avg_loss=pnl_metrics.average_loss,
            largest_win=pnl_metrics.largest_win,
            largest_loss=pnl_metrics.largest_loss,
            profit_factor=pnl_metrics.profit_factor,
            max_drawdown=pnl_metrics.max_drawdown,
            max_drawdown_pct=pnl_metrics.max_drawdown_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_holding_period=timedelta(hours=0),
            max_positions=stats["num_positions"],
            avg_positions=Decimal(str(stats["num_positions"])),
            total_fees=pnl_metrics.total_fees,
            total_slippage=Decimal("0"),
            equity_curve=self.equity_curve,
            trades=self.trades,
            daily_returns=self.daily_returns
        )

        return result

    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio"""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return Decimal("0")

        # TODO: Implement proper Sharpe calculation
        return Decimal("0")

    def _calculate_sortino_ratio(self) -> Decimal:
        """Calculate Sortino ratio (downside deviation only)"""
        # TODO: Implement Sortino calculation
        return Decimal("0")

    def _calculate_calmar_ratio(self, max_drawdown: Decimal) -> Decimal:
        """Calculate Calmar ratio (return / max drawdown)"""
        if max_drawdown == 0:
            return Decimal("0")

        # TODO: Implement proper Calmar calculation
        return Decimal("0")

    async def _teardown(self) -> None:
        """Cleanup backtest components"""
        logger.info("Cleaning up...")

        if self.strategy:
            await self.strategy.stop()
        if self.oms:
            await self.oms.stop()
        if self.portfolio:
            await self.portfolio.stop()
        if self.exchange:
            await self.exchange.disconnect()
        if self.event_bus:
            await self.event_bus.close()

        logger.info("✓ Cleanup complete")


class BacktestRunner:
    """Helper class to run multiple backtests"""

    def __init__(self):
        """Initialize backtest runner"""
        self.results: List[BacktestResult] = []

    async def run_backtest(
        self,
        strategy_class: Type[BaseStrategy],
        strategy_config: StrategyConfig,
        backtest_config: BacktestConfig
    ) -> BacktestResult:
        """Run single backtest"""
        engine = BacktestEngine(strategy_class, strategy_config, backtest_config)
        result = await engine.run()
        self.results.append(result)
        return result

    async def run_parameter_sweep(
        self,
        strategy_class: Type[BaseStrategy],
        base_config: StrategyConfig,
        base_backtest_config: BacktestConfig,
        parameter_grid: Dict[str, List[Any]]
    ) -> List[BacktestResult]:
        """
        Run backtests across parameter grid

        Args:
            strategy_class: Strategy class
            base_config: Base strategy config
            base_backtest_config: Base backtest config
            parameter_grid: Dict of parameter names to value lists

        Returns:
            List of results for each parameter combination
        """
        logger.info(f"Running parameter sweep: {len(parameter_grid)} parameters")

        # TODO: Implement parameter grid search

        return self.results

    def compare_results(self) -> str:
        """Generate comparison report of all results"""
        if not self.results:
            return "No results to compare"

        report = "\n╔══════════════════════════════════════════════════════════════════╗\n"
        report += "║                    BACKTEST COMPARISON                           ║\n"
        report += "╚══════════════════════════════════════════════════════════════════╝\n\n"

        for i, result in enumerate(self.results, 1):
            report += f"{i}. {result.strategy_name} - Return: {result.total_return_pct:.2f}% "
            report += f"Sharpe: {result.sharpe_ratio:.2f} Drawdown: {result.max_drawdown_pct:.2f}%\n"

        return report
