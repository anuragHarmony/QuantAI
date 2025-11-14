"""
Portfolio Persistence

Save and load portfolio state including:
- Positions
- Trade history
- P&L state
- Equity curve

Formats:
- JSON: Human-readable, easy to edit
- Parquet: Efficient, columnar storage
"""
import json
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from loguru import logger

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("PyArrow not available - Parquet persistence disabled")


class PortfolioStateSerializer:
    """Serialize/deserialize portfolio state"""

    @staticmethod
    def serialize_position(position) -> Dict:
        """Serialize position to dict"""
        return {
            "symbol": position.symbol,
            "exchange": position.exchange,
            "quantity": str(position.quantity),
            "average_entry_price": str(position.average_entry_price),
            "realized_pnl": str(position.realized_pnl),
            "unrealized_pnl": str(position.unrealized_pnl),
            "total_cost_basis": str(getattr(position, "total_cost_basis", "0")),
            "total_fees": str(getattr(position, "total_fees", "0")),
            "first_entry": position.first_entry.isoformat() if hasattr(position, "first_entry") and position.first_entry else None,
            "last_update": position.last_update.isoformat() if position.last_update else None,
        }

    @staticmethod
    def deserialize_position(data: Dict):
        """Deserialize position from dict"""
        from .position import PositionState

        return PositionState(
            symbol=data["symbol"],
            exchange=data["exchange"],
            quantity=Decimal(data["quantity"]),
            average_entry_price=Decimal(data["average_entry_price"]),
            realized_pnl=Decimal(data["realized_pnl"]),
            unrealized_pnl=Decimal(data["unrealized_pnl"]),
            total_cost_basis=Decimal(data.get("total_cost_basis", "0")),
            total_fees=Decimal(data.get("total_fees", "0")),
            first_entry=datetime.fromisoformat(data["first_entry"]) if data.get("first_entry") else None,
            last_update=datetime.fromisoformat(data["last_update"]) if data.get("last_update") else None,
        )


class PortfolioPersistence:
    """
    Save and load portfolio state

    Supports:
    - JSON format (human-readable)
    - Parquet format (efficient)
    """

    def __init__(self, base_path: str = "./data/portfolio"):
        """
        Initialize persistence

        Args:
            base_path: Base directory for portfolio files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized PortfolioPersistence: {base_path}")

    def save_state_json(
        self,
        positions: Dict,
        trades: List[Dict],
        initial_capital: Decimal,
        filename: str = "portfolio_state.json"
    ) -> None:
        """
        Save portfolio state to JSON

        Args:
            positions: Position dictionary
            trades: Trade history
            initial_capital: Initial capital
            filename: Output filename
        """

        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "initial_capital": str(initial_capital),
            "positions": {
                symbol: PortfolioStateSerializer.serialize_position(pos)
                for symbol, pos in positions.items()
            },
            "trades": trades,  # Already serializable
        }

        filepath = self.base_path / filename

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Saved portfolio state to {filepath}")

    def load_state_json(
        self,
        filename: str = "portfolio_state.json"
    ) -> Optional[Dict]:
        """
        Load portfolio state from JSON

        Args:
            filename: Input filename

        Returns:
            State dict or None if file doesn't exist
        """

        filepath = self.base_path / filename

        if not filepath.exists():
            logger.warning(f"Portfolio state file not found: {filepath}")
            return None

        with open(filepath, 'r') as f:
            state = json.load(f)

        # Deserialize positions
        positions = {}
        for symbol, pos_data in state.get("positions", {}).items():
            positions[symbol] = PortfolioStateSerializer.deserialize_position(pos_data)

        result = {
            "timestamp": datetime.fromisoformat(state["timestamp"]),
            "initial_capital": Decimal(state["initial_capital"]),
            "positions": positions,
            "trades": state.get("trades", []),
        }

        logger.info(f"Loaded portfolio state from {filepath}")

        return result

    def save_positions_csv(
        self,
        positions: Dict,
        filename: str = "positions.csv"
    ) -> None:
        """
        Export positions to CSV

        Args:
            positions: Position dictionary
            filename: Output filename
        """

        import csv

        filepath = self.base_path / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Symbol", "Exchange", "Quantity", "AvgEntryPrice",
                "RealizedPnL", "UnrealizedPnL", "TotalFees", "LastUpdate"
            ])

            # Positions
            for symbol, pos in positions.items():
                writer.writerow([
                    symbol,
                    pos.exchange,
                    str(pos.quantity),
                    str(pos.average_entry_price),
                    str(pos.realized_pnl),
                    str(pos.unrealized_pnl),
                    str(getattr(pos, "total_fees", "0")),
                    pos.last_update.isoformat() if pos.last_update else ""
                ])

        logger.info(f"Exported positions to {filepath}")

    def load_positions_csv(
        self,
        filename: str = "positions.csv"
    ) -> Dict:
        """
        Import positions from CSV

        Args:
            filename: Input filename

        Returns:
            Position dictionary
        """

        import csv
        from .position import PositionState

        filepath = self.base_path / filename

        if not filepath.exists():
            logger.warning(f"Positions file not found: {filepath}")
            return {}

        positions = {}

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                position = PositionState(
                    symbol=row["Symbol"],
                    exchange=row["Exchange"],
                    quantity=Decimal(row["Quantity"]),
                    average_entry_price=Decimal(row["AvgEntryPrice"]),
                    realized_pnl=Decimal(row["RealizedPnL"]),
                    unrealized_pnl=Decimal(row["UnrealizedPnL"]),
                    total_fees=Decimal(row.get("TotalFees", "0")),
                    last_update=datetime.fromisoformat(row["LastUpdate"]) if row.get("LastUpdate") else None
                )

                positions[row["Symbol"]] = position

        logger.info(f"Loaded {len(positions)} positions from {filepath}")

        return positions

    def save_equity_curve_csv(
        self,
        equity_curve: List,
        filename: str = "equity_curve.csv"
    ) -> None:
        """
        Export equity curve to CSV

        Args:
            equity_curve: List of (timestamp, equity) tuples
            filename: Output filename
        """

        import csv

        filepath = self.base_path / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Equity"])

            for timestamp, equity in equity_curve:
                writer.writerow([
                    timestamp.isoformat(),
                    str(equity)
                ])

        logger.info(f"Exported equity curve to {filepath}")


def seed_portfolio_from_file(
    portfolio_manager,
    filepath: str,
    format: str = "json"
) -> None:
    """
    Seed portfolio manager with positions from file

    Args:
        portfolio_manager: PortfolioManager instance
        filepath: Path to positions file
        format: File format ('json' or 'csv')
    """

    persistence = PortfolioPersistence()

    if format == "json":
        state = persistence.load_state_json(filepath)
        if state:
            # Restore positions
            portfolio_manager.position_tracker.positions = state["positions"]

            # Update initial capital if specified
            if "initial_capital" in state:
                portfolio_manager.initial_capital = state["initial_capital"]
                portfolio_manager.pnl_calculator.initial_capital = state["initial_capital"]

            logger.info(f"Seeded portfolio with {len(state['positions'])} positions from {filepath}")

    elif format == "csv":
        positions = persistence.load_positions_csv(filepath)
        portfolio_manager.position_tracker.positions = positions

        logger.info(f"Seeded portfolio with {len(positions)} positions from {filepath}")

    else:
        logger.error(f"Unknown format: {format}")
