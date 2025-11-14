"""
Report Generator

Generate formatted performance reports in multiple formats:
- Text reports (console-friendly)
- JSON reports (machine-readable)
- CSV exports (for spreadsheets)
- HTML reports (for web viewing)
"""
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from loguru import logger
import json
import csv
from pathlib import Path


class ReportFormat(Enum):
    """Report output formats"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    HTML = "html"


class ReportGenerator:
    """
    Generate formatted performance reports

    Converts performance metrics into various report formats.
    """

    def __init__(self):
        logger.info("Initialized ReportGenerator")

    def generate_text_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate console-friendly text report

        Args:
            metrics: Dictionary of metrics from PerformanceAnalyzer

        Returns:
            Formatted text report
        """
        performance = metrics.get("performance")
        risk = metrics.get("risk")
        trades = metrics.get("trades")

        report = []
        report.append("╔" + "═" * 68 + "╗")
        report.append("║" + " " * 20 + "PERFORMANCE REPORT" + " " * 30 + "║")
        report.append("╚" + "═" * 68 + "╝")
        report.append("")

        # Performance section
        if performance:
            report.append("┌─ PERFORMANCE METRICS " + "─" * 46 + "┐")
            report.append(f"│ Total Return:          {self._format_pct(performance.total_return_pct):>12} │")
            report.append(f"│ CAGR:                  {self._format_pct(performance.cagr * 100):>12} │")
            report.append(f"│ Daily Avg Return:      {self._format_pct(performance.daily_return_avg * 100):>12} │")
            report.append(f"│ Monthly Avg Return:    {self._format_pct(performance.monthly_return_avg * 100):>12} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Sharpe Ratio:          {float(performance.sharpe_ratio):>12.3f} │")
            report.append(f"│ Sortino Ratio:         {float(performance.sortino_ratio):>12.3f} │")
            report.append(f"│ Calmar Ratio:          {float(performance.calmar_ratio):>12.3f} │")
            if performance.omega_ratio:
                report.append(f"│ Omega Ratio:           {float(performance.omega_ratio):>12.3f} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Period: {performance.start_date.date()} to {performance.end_date.date()} ({performance.num_days} days)" + " " * (68 - len(f"Period: {performance.start_date.date()} to {performance.end_date.date()} ({performance.num_days} days)") - 2) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # Risk section
        if risk:
            report.append("┌─ RISK METRICS " + "─" * 53 + "┐")
            report.append(f"│ Max Drawdown:          {self._format_pct(risk.max_drawdown_pct * 100):>12} (${float(risk.max_drawdown):,.2f})" + " " * (68 - len(f"Max Drawdown:          {self._format_pct(risk.max_drawdown_pct * 100):>12} (${float(risk.max_drawdown):,.2f})") - 2) + "│")
            report.append(f"│ Current Drawdown:      {self._format_pct(risk.current_drawdown_pct * 100):>12} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Daily Volatility:      {self._format_pct(risk.daily_volatility * 100):>12} │")
            report.append(f"│ Annual Volatility:     {self._format_pct(risk.annual_volatility * 100):>12} │")
            report.append("│" + " " * 68 + "│")
            if risk.var_95:
                report.append(f"│ 95% VaR:               {self._format_pct(risk.var_95 * 100):>12} │")
            if risk.var_99:
                report.append(f"│ 99% VaR:               {self._format_pct(risk.var_99 * 100):>12} │")
            if risk.cvar_95:
                report.append(f"│ 95% CVaR:              {self._format_pct(risk.cvar_95 * 100):>12} │")
            report.append("│" + " " * 68 + "│")
            if risk.skewness:
                report.append(f"│ Skewness:              {float(risk.skewness):>12.3f} │")
            if risk.kurtosis:
                report.append(f"│ Kurtosis:              {float(risk.kurtosis):>12.3f} │")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # Trade metrics section
        if trades:
            report.append("┌─ TRADE STATISTICS " + "─" * 49 + "┐")
            report.append(f"│ Total Trades:          {trades.num_trades:>12,d} │")
            report.append(f"│ Winning Trades:        {trades.num_winning_trades:>12,d} │")
            report.append(f"│ Losing Trades:         {trades.num_losing_trades:>12,d} │")
            report.append(f"│ Win Rate:              {self._format_pct(trades.win_rate):>12} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Gross Profit:          ${float(trades.gross_profit):>11,.2f} │")
            report.append(f"│ Gross Loss:            ${float(trades.gross_loss):>11,.2f} │")
            report.append(f"│ Net Profit:            ${float(trades.net_profit):>11,.2f} │")
            report.append(f"│ Profit Factor:         {float(trades.profit_factor):>12.2f} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Average Win:           ${float(trades.avg_win):>11,.2f} │")
            report.append(f"│ Average Loss:          ${float(trades.avg_loss):>11,.2f} │")
            report.append(f"│ Avg Win/Loss Ratio:    {float(trades.avg_win_loss_ratio):>12.2f} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Largest Win:           ${float(trades.largest_win):>11,.2f} │")
            report.append(f"│ Largest Loss:          ${float(trades.largest_loss):>11,.2f} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Max Consecutive Wins:  {trades.max_consecutive_wins:>12,d} │")
            report.append(f"│ Max Consecutive Loss:  {trades.max_consecutive_losses:>12,d} │")
            report.append("│" + " " * 68 + "│")
            report.append(f"│ Total Fees:            ${float(trades.total_fees):>11,.2f} │")
            if trades.avg_holding_period:
                report.append(f"│ Avg Holding Period:    {str(trades.avg_holding_period).split('.')[0]:>12} │")
            report.append("└" + "─" * 68 + "┘")

        return "\n".join(report)

    def generate_json_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate JSON report

        Args:
            metrics: Dictionary of metrics

        Returns:
            JSON string
        """
        # Convert Decimal to float for JSON serialization
        json_metrics = self._convert_decimals(metrics)

        return json.dumps(json_metrics, indent=2, default=str)

    def generate_comparison_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate comparison report for multiple strategies

        Args:
            results: List of metric dictionaries

        Returns:
            Formatted comparison table
        """
        if not results:
            return "No results to compare"

        report = []
        report.append("╔" + "═" * 98 + "╗")
        report.append("║" + " " * 35 + "STRATEGY COMPARISON" + " " * 44 + "║")
        report.append("╚" + "═" * 98 + "╝")
        report.append("")

        # Header
        report.append(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>8} {'Sortino':>8} {'Max DD':>10} {'Win Rate':>10} {'Trades':>8}")
        report.append("─" * 100)

        # Each strategy
        for result in results:
            perf = result.get("performance", {})
            risk = result.get("risk", {})
            trades = result.get("trades", {})

            name = result.get("name", "Unknown")[:30]
            ret = self._format_pct(perf.total_return_pct if hasattr(perf, 'total_return_pct') else 0)
            sharpe = f"{float(perf.sharpe_ratio) if hasattr(perf, 'sharpe_ratio') else 0:>8.2f}"
            sortino = f"{float(perf.sortino_ratio) if hasattr(perf, 'sortino_ratio') else 0:>8.2f}"
            max_dd = self._format_pct(risk.max_drawdown_pct * 100 if hasattr(risk, 'max_drawdown_pct') else 0)
            win_rate = self._format_pct(trades.win_rate if hasattr(trades, 'win_rate') else 0)
            num_trades = f"{trades.num_trades if hasattr(trades, 'num_trades') else 0:>8,d}"

            report.append(f"{name:<30} {ret:>10} {sharpe} {sortino} {max_dd:>10} {win_rate:>10} {num_trades}")

        return "\n".join(report)

    def _format_pct(self, value: float) -> str:
        """Format percentage value"""
        return f"{value:>7.2f}%"

    def _convert_decimals(self, obj):
        """Recursively convert Decimal to float for JSON serialization"""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._convert_decimals(obj.__dict__)
        else:
            return obj


def export_to_csv(
    metrics: Dict[str, Any],
    filepath: str,
    include_trades: bool = False
) -> None:
    """
    Export metrics to CSV file

    Args:
        metrics: Dictionary of metrics
        filepath: Output file path
        include_trades: Whether to include trade-level data
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write performance metrics
        writer.writerow(["PERFORMANCE METRICS"])
        writer.writerow(["Metric", "Value"])

        performance = metrics.get("performance")
        if performance:
            writer.writerow(["Total Return %", float(performance.total_return_pct)])
            writer.writerow(["CAGR %", float(performance.cagr * 100)])
            writer.writerow(["Sharpe Ratio", float(performance.sharpe_ratio)])
            writer.writerow(["Sortino Ratio", float(performance.sortino_ratio)])
            writer.writerow(["Calmar Ratio", float(performance.calmar_ratio)])

        writer.writerow([])

        # Write risk metrics
        writer.writerow(["RISK METRICS"])
        writer.writerow(["Metric", "Value"])

        risk = metrics.get("risk")
        if risk:
            writer.writerow(["Max Drawdown %", float(risk.max_drawdown_pct * 100)])
            writer.writerow(["Annual Volatility %", float(risk.annual_volatility * 100)])
            if risk.var_95:
                writer.writerow(["95% VaR %", float(risk.var_95 * 100)])

        writer.writerow([])

        # Write trade metrics
        writer.writerow(["TRADE METRICS"])
        writer.writerow(["Metric", "Value"])

        trades = metrics.get("trades")
        if trades:
            writer.writerow(["Total Trades", trades.num_trades])
            writer.writerow(["Win Rate %", float(trades.win_rate)])
            writer.writerow(["Profit Factor", float(trades.profit_factor)])
            writer.writerow(["Average Win", float(trades.avg_win)])
            writer.writerow(["Average Loss", float(trades.avg_loss)])

    logger.info(f"Exported metrics to CSV: {filepath}")


def export_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Export metrics to JSON file

    Args:
        metrics: Dictionary of metrics
        filepath: Output file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Decimals to float
    def convert(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return convert(obj.__dict__)
        else:
            return obj

    json_metrics = convert(metrics)

    with open(filepath, 'w') as f:
        json.dump(json_metrics, f, indent=2, default=str)

    logger.info(f"Exported metrics to JSON: {filepath}")


logger.info("Report generator module loaded")
