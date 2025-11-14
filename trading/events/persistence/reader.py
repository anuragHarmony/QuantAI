"""
Event persistence reader

Reads events from Parquet files for replay/simulation.
Supports time-accurate replay with delays.

Features:
- Read by date range
- Filter by symbol/exchange
- Time-accurate replay (preserves original delays)
- Fast-forward support
"""
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, AsyncIterator, Callable
import asyncio
from loguru import logger
from decimal import Decimal

from ..base import BaseEvent
from .. import (
    TickEvent, TradeEvent, OrderBookEvent, BarEvent, QuoteEvent,
    OrderSubmittedEvent, OrderAcceptedEvent, OrderFilledEvent,
    OrderCancelledEvent, OrderRejectedEvent,
    PositionOpenedEvent, PositionModifiedEvent, PositionClosedEvent,
)


# Map event types to classes
EVENT_TYPE_MAP = {
    "tick": TickEvent,
    "trade": TradeEvent,
    "order_book": OrderBookEvent,
    "bar": BarEvent,
    "quote": QuoteEvent,
    "order_submitted": OrderSubmittedEvent,
    "order_accepted": OrderAcceptedEvent,
    "order_filled": OrderFilledEvent,
    "order_cancelled": OrderCancelledEvent,
    "order_rejected": OrderRejectedEvent,
    "position_opened": PositionOpenedEvent,
    "position_modified": PositionModifiedEvent,
    "position_closed": PositionClosedEvent,
}


class EventReader:
    """
    Reads events from Parquet files for replay

    Supports:
    - Time range queries
    - Symbol/exchange filtering
    - Time-accurate replay
    - Speedup factor
    """

    def __init__(
        self,
        base_path: str = "./data/events",
        speedup: float = 1.0
    ):
        """
        Initialize reader

        Args:
            base_path: Base directory for event files
            speedup: Replay speed multiplier (1.0 = real-time, 10.0 = 10x faster)
        """
        self.base_path = Path(base_path)
        self.speedup = speedup

        if not self.base_path.exists():
            logger.warning(f"Event directory does not exist: {base_path}")

        logger.info(f"Initialized EventReader at {base_path} (speedup: {speedup}x)")

    async def read_range(
        self,
        start: datetime,
        end: datetime,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        event_types: Optional[list[str]] = None
    ) -> AsyncIterator[BaseEvent]:
        """
        Read events in date range

        Args:
            start: Start datetime
            end: End datetime
            symbol: Filter by symbol (optional)
            exchange: Filter by exchange (optional)
            event_types: Filter by event types (optional)

        Yields:
            Events in time order
        """
        # Get all matching files
        files = self._find_files(
            start.date(),
            end.date(),
            exchange=exchange,
            event_types=event_types
        )

        if not files:
            logger.warning(f"No files found for range {start} to {end}")
            return

        # Read and merge events from all files
        all_events = []
        for file_path in files:
            events = await self._read_file(file_path)
            all_events.extend(events)

        # Sort by timestamp
        all_events.sort(key=lambda e: e["timestamp"])

        # Filter by time range and symbol
        for event_dict in all_events:
            event_time = datetime.fromisoformat(event_dict["timestamp"])

            # Check time range
            if not (start <= event_time <= end):
                continue

            # Check symbol filter
            if symbol and event_dict.get("symbol") != symbol:
                continue

            # Convert back to event object
            event = self._dict_to_event(event_dict)
            yield event

    async def replay(
        self,
        start: datetime,
        end: datetime,
        callback: Callable[[BaseEvent], None],
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        preserve_timing: bool = True
    ) -> None:
        """
        Replay events with time-accurate delays

        Args:
            start: Start datetime
            end: End datetime
            callback: Function to call for each event
            symbol: Filter by symbol (optional)
            exchange: Filter by exchange (optional)
            event_types: Filter by event types (optional)
            preserve_timing: Preserve original delays between events
        """
        prev_timestamp: Optional[datetime] = None

        async for event in self.read_range(start, end, symbol, exchange, event_types):
            # Calculate delay
            if preserve_timing and prev_timestamp:
                delay = (event.timestamp - prev_timestamp).total_seconds()
                delay_with_speedup = delay / self.speedup

                if delay_with_speedup > 0:
                    await asyncio.sleep(delay_with_speedup)

            # Call callback
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

            prev_timestamp = event.timestamp

    def _find_files(
        self,
        start_date: date,
        end_date: date,
        exchange: Optional[str] = None,
        event_types: Optional[list[str]] = None
    ) -> list[Path]:
        """
        Find all parquet files in date range

        Args:
            start_date: Start date
            end_date: End date
            exchange: Filter by exchange
            event_types: Filter by event types

        Returns:
            List of file paths
        """
        files = []

        # Iterate through dates
        current_date = start_date
        while current_date <= end_date:
            # Get directory for this date
            date_dir = (
                self.base_path
                / str(current_date.year)
                / f"{current_date.month:02d}"
                / f"{current_date.day:02d}"
            )

            if date_dir.exists():
                # Find matching parquet files
                for file_path in date_dir.glob("*.parquet"):
                    # Parse filename: {event_type}_{exchange}.parquet
                    name_parts = file_path.stem.split("_")
                    if len(name_parts) >= 2:
                        file_event_type = "_".join(name_parts[:-1])
                        file_exchange = name_parts[-1]

                        # Check filters
                        if event_types and file_event_type not in event_types:
                            continue

                        if exchange and file_exchange != exchange:
                            continue

                        files.append(file_path)

            current_date += timedelta(days=1)

        logger.info(f"Found {len(files)} files for date range")
        return files

    async def _read_file(self, file_path: Path) -> list[dict]:
        """
        Read events from parquet file

        Args:
            file_path: Path to parquet file

        Returns:
            List of event dictionaries
        """
        try:
            import pandas as pd

            # Read parquet file
            df = pd.read_parquet(file_path)

            # Convert to list of dicts
            events = df.to_dict(orient="records")

            logger.debug(f"Read {len(events)} events from {file_path}")
            return events

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []

    def _dict_to_event(self, event_dict: dict) -> BaseEvent:
        """
        Convert dictionary back to event object

        Args:
            event_dict: Event dictionary

        Returns:
            Event object
        """
        # Get event type
        event_type = event_dict.get("event_type")

        # Get event class
        event_class = EVENT_TYPE_MAP.get(event_type, BaseEvent)

        # Convert timestamp strings back to datetime
        if "timestamp" in event_dict:
            event_dict["timestamp"] = datetime.fromisoformat(event_dict["timestamp"])

        if "exchange_timestamp" in event_dict and event_dict["exchange_timestamp"]:
            event_dict["exchange_timestamp"] = datetime.fromisoformat(
                event_dict["exchange_timestamp"]
            )

        if "fill_timestamp" in event_dict and event_dict["fill_timestamp"]:
            event_dict["fill_timestamp"] = datetime.fromisoformat(
                event_dict["fill_timestamp"]
            )

        # Convert Decimal strings back to Decimal
        decimal_fields = [
            "bid", "ask", "last", "price", "quantity", "volume",
            "bid_volume", "ask_volume", "open", "high", "low", "close",
            "bid_price", "ask_price", "bid_quantity", "ask_quantity",
            "fill_price", "fill_quantity", "total_filled", "average_fill_price",
            "entry_price", "current_price", "unrealized_pnl", "realized_pnl"
        ]

        for field in decimal_fields:
            if field in event_dict and event_dict[field] is not None:
                if isinstance(event_dict[field], str):
                    event_dict[field] = Decimal(event_dict[field])

        # Create event object
        try:
            event = event_class(**event_dict)
            return event
        except Exception as e:
            logger.error(f"Error creating event from dict: {e}")
            logger.debug(f"Event dict: {event_dict}")
            # Fallback to BaseEvent
            return BaseEvent(**event_dict)

    def get_date_range(self) -> tuple[Optional[date], Optional[date]]:
        """
        Get the date range of available data

        Returns:
            Tuple of (earliest_date, latest_date) or (None, None)
        """
        if not self.base_path.exists():
            return None, None

        # Find all year directories
        year_dirs = sorted([d for d in self.base_path.iterdir() if d.is_dir()])
        if not year_dirs:
            return None, None

        # Get earliest date
        earliest_year = year_dirs[0]
        earliest_date = self._get_earliest_date_in_year(earliest_year)

        # Get latest date
        latest_year = year_dirs[-1]
        latest_date = self._get_latest_date_in_year(latest_year)

        return earliest_date, latest_date

    def _get_earliest_date_in_year(self, year_dir: Path) -> Optional[date]:
        """Get earliest date with data in a year directory"""
        month_dirs = sorted([d for d in year_dir.iterdir() if d.is_dir()])
        if not month_dirs:
            return None

        for month_dir in month_dirs:
            day_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()])
            if day_dirs:
                day_dir = day_dirs[0]
                year = int(year_dir.name)
                month = int(month_dir.name)
                day = int(day_dir.name)
                return date(year, month, day)

        return None

    def _get_latest_date_in_year(self, year_dir: Path) -> Optional[date]:
        """Get latest date with data in a year directory"""
        month_dirs = sorted([d for d in year_dir.iterdir() if d.is_dir()], reverse=True)
        if not month_dirs:
            return None

        for month_dir in month_dirs:
            day_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()], reverse=True)
            if day_dirs:
                day_dir = day_dirs[0]
                year = int(year_dir.name)
                month = int(month_dir.name)
                day = int(day_dir.name)
                return date(year, month, day)

        return None
