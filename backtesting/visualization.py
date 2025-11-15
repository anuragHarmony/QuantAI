"""
Backtest Visualization

Creates charts and plots for backtest results.
"""
from typing import Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Visualization features disabled.")

from backtesting.engine import BacktestResult


class BacktestVisualizer:
    """
    Create visualizations for backtest results

    Single Responsibility: Generate charts and plots
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Visualization disabled: matplotlib not available")
            return

        self.style = style
        plt.style.use(style)

    def create_full_report(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Create comprehensive visual report

        Args:
            result: Backtest result
            save_path: Path to save figure (default: auto-generate)
            show: Whether to show plot (default: False)

        Returns:
            Path where figure was saved, or None if not saved
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot create report: matplotlib not available")
            return None

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, result)

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_drawdown(ax2, result)

        # 3. Returns distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax3, result)

        # 4. Monthly returns heatmap
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_monthly_returns(ax4, result)

        # 5. Trade analysis
        ax5 = fig.add_subplot(gs[3, 0])
        self._plot_trade_analysis(ax5, result)

        # 6. Metrics table
        ax6 = fig.add_subplot(gs[3, 1])
        self._plot_metrics_table(ax6, result)

        # Title
        fig.suptitle(
            f"{result.strategy_name} - Backtest Results\n"
            f"{result.symbol} {result.timeframe} | {result.start_date.date()} to {result.end_date.date()}",
            fontsize=16,
            fontweight='bold'
        )

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Report saved to: {save_path}")
        elif not show:
            # Auto-generate path
            save_path = f"backtest_{result.strategy_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Report saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def _plot_equity_curve(self, ax, result: BacktestResult):
        """Plot equity curve with buy/sell markers"""
        equity = result.equity_curve

        # Plot equity
        ax.plot(equity.index, equity.values, linewidth=2, label='Equity', color='#2E86AB')

        # Add buy/sell markers if we have trade data
        if result.trades:
            long_entries = [t for t in result.trades if t.direction == 'long']
            short_entries = [t for t in result.trades if t.direction == 'short']

            if long_entries:
                entry_times = [t.entry_time for t in long_entries]
                entry_equity = [equity.loc[t.entry_time] for t in long_entries if t.entry_time in equity.index]
                ax.scatter(entry_times[:len(entry_equity)], entry_equity, marker='^', color='green',
                          s=100, alpha=0.7, label='Long Entry', zorder=5)

            if short_entries:
                entry_times = [t.entry_time for t in short_entries]
                entry_equity = [equity.loc[t.entry_time] for t in short_entries if t.entry_time in equity.index]
                ax.scatter(entry_times[:len(entry_equity)], entry_equity, marker='v', color='red',
                          s=100, alpha=0.7, label='Short Entry', zorder=5)

        ax.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_drawdown(self, ax, result: BacktestResult):
        """Plot drawdown curve"""
        equity = result.equity_curve
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, linewidth=1.5, color='darkred')

        ax.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_returns_distribution(self, ax, result: BacktestResult):
        """Plot distribution of returns"""
        if not result.trades:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
            return

        returns = [t.pnl_pct * 100 for t in result.trades]

        ax.hist(returns, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break-even')
        ax.axvline(np.mean(returns), color='green', linestyle='--', linewidth=2, alpha=0.7, label='Mean')

        ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_monthly_returns(self, ax, result: BacktestResult):
        """Plot monthly returns heatmap"""
        equity = result.equity_curve
        returns = equity.pct_change()

        # Resample to monthly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

        if len(monthly_returns) < 3:
            ax.text(0.5, 0.5, 'Insufficient data\nfor monthly view',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Returns', fontsize=12, fontweight='bold')
            return

        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })

        pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')

        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.index)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', rotation=270, labelpad=15)

        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if not np.isnan(pivot.iloc[i, j]):
                    text = ax.text(j, i, f'{pivot.iloc[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')

    def _plot_trade_analysis(self, ax, result: BacktestResult):
        """Plot trade analysis"""
        if not result.trades:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trade Analysis', fontsize=12, fontweight='bold')
            return

        # Cumulative P&L
        cumulative_pnl = np.cumsum([t.pnl for t in result.trades])

        ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='#2E86AB')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        # Color winning/losing trades
        for i, trade in enumerate(result.trades):
            color = 'green' if trade.pnl > 0 else 'red'
            ax.axvline(i, color=color, alpha=0.1, linewidth=0.5)

        ax.set_title('Cumulative P&L by Trade', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)

    def _plot_metrics_table(self, ax, result: BacktestResult):
        """Plot metrics as table"""
        ax.axis('tight')
        ax.axis('off')

        metrics = result.metrics

        # Prepare data
        data = [
            ['Metric', 'Value'],
            ['Total Return', f'{metrics.total_return*100:.2f}%'],
            ['Annual Return', f'{metrics.annualized_return*100:.2f}%'],
            ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
            ['Sortino Ratio', f'{metrics.sortino_ratio:.2f}'],
            ['Calmar Ratio', f'{metrics.calmar_ratio:.2f}'],
            ['Max Drawdown', f'{metrics.max_drawdown*100:.2f}%'],
            ['Volatility', f'{metrics.volatility*100:.2f}%'],
            ['Win Rate', f'{metrics.win_rate*100:.1f}%'],
            ['Profit Factor', f'{metrics.profit_factor:.2f}'],
            ['Total Trades', f'{metrics.total_trades}'],
            ['Avg Win', f'${metrics.avg_win:.2f}'],
            ['Avg Loss', f'${metrics.avg_loss:.2f}'],
            ['Expectancy', f'${metrics.expectancy:.2f}'],
        ]

        # Create table
        table = ax.table(cellText=data, cellLoc='left', loc='center',
                        colWidths=[0.5, 0.5])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8E8E8')

        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)

    def create_equity_curve_only(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> Optional[str]:
        """Create simple equity curve plot"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_equity_curve(ax, result)

        plt.title(f"{result.strategy_name} - Equity Curve", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        elif not show:
            save_path = f"equity_{result.strategy_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return save_path


logger.info("Backtest visualization module loaded")
