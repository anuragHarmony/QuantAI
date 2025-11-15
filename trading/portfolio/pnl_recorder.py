"""
PnL Recorder

Automatically records P&L metrics to file at regular intervals.

Features:
- Subscribes to portfolio value events
- Dumps PNL snapshots to CSV every N seconds
- Creates timestamped files per session
- Buffers in memory for performance
- Supports CSV and JSON formats

Usage:
    recorder = PnLRecorder(
        event_bus=event_bus,
        dump_interval=10.0,  # Dump every 10 seconds
        output_dir="./data/pnl"
    )

    await recorder.start()
    # ... trading runs ...
    await recorder.stop()
"""

import asyncio
import csv
import json
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
from decimal import Decimal
from loguru import logger

from ..events.bus import IEventBus
from ..events.position import PortfolioValueEvent


class PnLRecorder:
    """
    Records P&L metrics to file at regular intervals

    Subscribes to portfolio value events and dumps snapshots
    to CSV/JSON files periodically for analysis and monitoring.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        dump_interval: float = 10.0,  # Seconds between dumps
        output_dir: str = "./data/pnl",
        format: str = "csv",  # 'csv' or 'json'
        session_name: Optional[str] = None,
    ):
        """
        Initialize PnL Recorder

        Args:
            event_bus: Event bus for subscribing to portfolio events
            dump_interval: Seconds between file dumps (default 10.0)
            output_dir: Directory to save PNL files
            format: Output format ('csv' or 'json')
            session_name: Optional session name (auto-generated if None)
        """
        self.event_bus = event_bus
        self.dump_interval = dump_interval
        self.output_dir = Path(output_dir)
        self.format = format.lower()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate session name and filename
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        self.session_name = session_name

        if self.format == "csv":
            self.filepath = self.output_dir / f"{session_name}_pnl.csv"
        elif self.format == "json":
            self.filepath = self.output_dir / f"{session_name}_pnl.jsonl"
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")

        # In-memory buffer for PNL snapshots
        self.snapshots: List[Dict] = []

        # Latest snapshot (updated on every portfolio value event)
        self.latest_snapshot: Optional[Dict] = None

        # State
        self.is_running = False
        self._dump_task: Optional[asyncio.Task] = None

        # Initialize file with headers
        self._initialize_file()

        logger.info(
            f"Initialized PnLRecorder: "
            f"interval={dump_interval}s, format={format}, "
            f"file={self.filepath}"
        )

    def _initialize_file(self) -> None:
        """Initialize output file with headers"""

        if self.format == "csv":
            # Write CSV header
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "cash",
                    "equity",
                    "total_value",
                    "unrealized_pnl",
                    "realized_pnl",
                    "total_pnl",
                    "return_pct",
                    "initial_value",
                ])

            logger.debug(f"Initialized CSV file: {self.filepath}")

        elif self.format == "json":
            # JSON Lines format - no header needed
            # Each line is a JSON object
            self.filepath.touch()
            logger.debug(f"Initialized JSONL file: {self.filepath}")

    async def start(self) -> None:
        """Start the PnL recorder"""

        if self.is_running:
            logger.warning("PnLRecorder already running")
            return

        self.is_running = True

        # Subscribe to portfolio value events
        await self.event_bus.subscribe("portfolio_value", self._handle_portfolio_value)

        # Start periodic dump task
        self._dump_task = asyncio.create_task(self._dump_loop())

        logger.info(f"PnLRecorder started: {self.session_name}")

    async def stop(self) -> None:
        """Stop the PnL recorder and perform final dump"""

        self.is_running = False

        # Cancel dump task
        if self._dump_task:
            self._dump_task.cancel()
            try:
                await self._dump_task
            except asyncio.CancelledError:
                pass

        # Final dump
        await self._dump_to_file()

        logger.info(
            f"PnLRecorder stopped: {self.session_name} | "
            f"Total snapshots: {len(self.snapshots)}"
        )

    async def _handle_portfolio_value(self, event: PortfolioValueEvent) -> None:
        """Handle portfolio value event - update latest snapshot"""

        snapshot = {
            "timestamp": datetime.now(),
            "cash": event.cash,
            "equity": event.equity,
            "total_value": event.total_value,
            "unrealized_pnl": event.unrealized_pnl,
            "realized_pnl": event.realized_pnl,
            "total_pnl": event.total_pnl,
            "return_pct": event.return_pct,
            "initial_value": event.initial_value,
        }

        self.latest_snapshot = snapshot

    async def _dump_loop(self) -> None:
        """Periodic dump loop"""

        while self.is_running:
            try:
                await asyncio.sleep(self.dump_interval)

                # Dump latest snapshot to file
                await self._dump_to_file()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in PnL dump loop: {e}", exc_info=True)

    async def _dump_to_file(self) -> None:
        """Dump latest snapshot to file"""

        if self.latest_snapshot is None:
            logger.debug("No PnL snapshot to dump yet")
            return

        try:
            if self.format == "csv":
                await self._dump_csv()
            elif self.format == "json":
                await self._dump_json()

            # Add to history
            self.snapshots.append(self.latest_snapshot.copy())

            logger.debug(
                f"Dumped PnL snapshot: "
                f"equity={self.latest_snapshot['total_value']}, "
                f"pnl={self.latest_snapshot['total_pnl']}, "
                f"return={self.latest_snapshot['return_pct']}%"
            )

        except Exception as e:
            logger.error(f"Failed to dump PnL snapshot: {e}", exc_info=True)

    async def _dump_csv(self) -> None:
        """Dump snapshot to CSV file"""

        snapshot = self.latest_snapshot

        # Append row to CSV
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                snapshot["timestamp"].isoformat(),
                str(snapshot["cash"]),
                str(snapshot["equity"]),
                str(snapshot["total_value"]),
                str(snapshot["unrealized_pnl"]),
                str(snapshot["realized_pnl"]),
                str(snapshot["total_pnl"]),
                str(snapshot["return_pct"]),
                str(snapshot["initial_value"]),
            ])

    async def _dump_json(self) -> None:
        """Dump snapshot to JSON Lines file"""

        snapshot = self.latest_snapshot.copy()

        # Convert Decimal to string for JSON serialization
        serializable = {
            k: str(v) if isinstance(v, Decimal) else v.isoformat() if isinstance(v, datetime) else v
            for k, v in snapshot.items()
        }

        # Append JSON line
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(serializable) + '\n')

    def get_snapshots(self) -> List[Dict]:
        """Get all recorded snapshots"""
        return self.snapshots.copy()

    def get_latest_snapshot(self) -> Optional[Dict]:
        """Get latest snapshot"""
        return self.latest_snapshot.copy() if self.latest_snapshot else None

    def get_filepath(self) -> Path:
        """Get output file path"""
        return self.filepath

    def get_statistics(self) -> Dict:
        """Get recording statistics"""

        return {
            "session_name": self.session_name,
            "format": self.format,
            "dump_interval": self.dump_interval,
            "filepath": str(self.filepath),
            "snapshots_recorded": len(self.snapshots),
            "is_running": self.is_running,
            "latest_snapshot_time": self.latest_snapshot["timestamp"] if self.latest_snapshot else None,
        }


# Convenience function for quick setup
def create_pnl_recorder(
    event_bus: IEventBus,
    dump_interval: float = 10.0,
    output_dir: str = "./data/pnl",
    format: str = "csv",
) -> PnLRecorder:
    """
    Create and return a PnL recorder

    Args:
        event_bus: Event bus
        dump_interval: Seconds between dumps (default 10.0)
        output_dir: Output directory (default "./data/pnl")
        format: Output format - 'csv' or 'json' (default 'csv')

    Returns:
        Configured PnLRecorder instance
    """

    return PnLRecorder(
        event_bus=event_bus,
        dump_interval=dump_interval,
        output_dir=output_dir,
        format=format,
    )
