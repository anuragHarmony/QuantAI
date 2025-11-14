"""
Event Log Storage Backends

Multiple storage backends for event logs:
- FileLogStorage: Store events to files (JSON or CSV)
- MemoryLogStorage: Store events in memory (for testing/replay)
- DatabaseLogStorage: Store events to database (future implementation)
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from abc import ABC, abstractmethod
from loguru import logger
import json
import csv
import asyncio
from collections import defaultdict

from .event_logger import LoggedEvent, EventType


class LogStorage(ABC):
    """Abstract base class for log storage backends"""

    @abstractmethod
    async def store(self, event: LoggedEvent) -> None:
        """Store a logged event"""
        pass

    @abstractmethod
    async def query(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[LoggedEvent]:
        """Query stored events"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close storage backend"""
        pass


class FileLogStorage(LogStorage):
    """
    File-based log storage

    Stores events to JSON or CSV files, organized by date and event type.
    """

    def __init__(
        self,
        base_path: str = "./logs/events",
        format: str = "json",  # "json" or "csv"
        rotate_daily: bool = True,
        buffer_size: int = 100
    ):
        """
        Initialize file storage

        Args:
            base_path: Base directory for log files
            format: File format ("json" or "csv")
            rotate_daily: Create new file each day
            buffer_size: Number of events to buffer before flushing
        """
        self.base_path = Path(base_path)
        self.format = format
        self.rotate_daily = rotate_daily
        self.buffer_size = buffer_size

        # Create directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Buffers for batch writing
        self.buffers: Dict[str, List[LoggedEvent]] = defaultdict(list)
        self.file_handles: Dict[str, Any] = {}

        # Statistics
        self.events_written = 0
        self.files_created = 0

        logger.info(f"File log storage initialized: {self.base_path} (format: {format})")

    async def store(self, event: LoggedEvent) -> None:
        """Store event to file"""

        # Determine file path
        filepath = self._get_filepath(event)

        # Add to buffer
        self.buffers[filepath].append(event)

        # Flush if buffer is full
        if len(self.buffers[filepath]) >= self.buffer_size:
            await self._flush_buffer(filepath)

    async def _flush_buffer(self, filepath: str) -> None:
        """Flush buffer to file"""

        if filepath not in self.buffers or not self.buffers[filepath]:
            return

        events = self.buffers[filepath]
        self.buffers[filepath] = []

        try:
            if self.format == "json":
                await self._write_json(filepath, events)
            elif self.format == "csv":
                await self._write_csv(filepath, events)

            self.events_written += len(events)

        except Exception as e:
            logger.error(f"Failed to write events to {filepath}: {e}")

    async def _write_json(self, filepath: str, events: List[LoggedEvent]) -> None:
        """Write events to JSON file"""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing file or create new one
        existing_data = []
        if path.exists():
            try:
                with open(path, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []

        # Add new events
        new_data = existing_data + [event.to_dict() for event in events]

        # Write back
        with open(path, 'w') as f:
            json.dump(new_data, f, indent=2, default=str)

    async def _write_csv(self, filepath: str, events: List[LoggedEvent]) -> None:
        """Write events to CSV file"""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we need headers
        file_exists = path.exists()

        with open(path, 'a', newline='') as f:
            # Define CSV fields
            fieldnames = [
                'event_id', 'event_type', 'timestamp', 'strategy_id',
                'exchange', 'symbol', 'session_id', 'data'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header if new file
            if not file_exists:
                writer.writeheader()
                self.files_created += 1

            # Write events
            for event in events:
                event_dict = event.to_dict()
                # Serialize data field as JSON string
                event_dict['data'] = json.dumps(event_dict['data'])
                writer.writerow(event_dict)

    def _get_filepath(self, event: LoggedEvent) -> str:
        """Determine file path for event"""

        # Base path
        path_parts = [str(self.base_path)]

        # Add date if daily rotation
        if self.rotate_daily:
            date_str = event.timestamp.strftime("%Y%m%d")
            path_parts.append(date_str)

        # Add event type
        path_parts.append(event.event_type.value)

        # Create filename
        if self.rotate_daily:
            filename = f"{event.event_type.value}.{self.format}"
        else:
            filename = f"{event.event_type.value}_{event.timestamp.strftime('%Y%m%d')}.{self.format}"

        path_parts.append(filename)

        return str(Path(*path_parts))

    async def query(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[LoggedEvent]:
        """Query events from files"""

        # TODO: Implement file querying
        # This would involve scanning files and filtering events
        logger.warning("File querying not yet implemented")
        return []

    async def close(self) -> None:
        """Close storage and flush all buffers"""

        logger.info(f"Closing file storage. Flushing {len(self.buffers)} buffers...")

        # Flush all remaining buffers
        for filepath in list(self.buffers.keys()):
            await self._flush_buffer(filepath)

        # Close all file handles
        for handle in self.file_handles.values():
            try:
                handle.close()
            except:
                pass

        logger.info(
            f"File storage closed. "
            f"Events written: {self.events_written}, "
            f"Files created: {self.files_created}"
        )


class MemoryLogStorage(LogStorage):
    """
    In-memory log storage

    Stores events in memory for fast access and replay.
    Useful for testing and short-term storage.
    """

    def __init__(self, max_events: int = 100000):
        """
        Initialize memory storage

        Args:
            max_events: Maximum number of events to store
        """
        self.max_events = max_events
        self.events: List[LoggedEvent] = []

        logger.info(f"Memory log storage initialized (max: {max_events} events)")

    async def store(self, event: LoggedEvent) -> None:
        """Store event in memory"""

        self.events.append(event)

        # Remove oldest events if over limit
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    async def query(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[LoggedEvent]:
        """Query events from memory"""

        results = self.events

        # Filter by event type
        if event_types:
            results = [e for e in results if e.event_type in event_types]

        # Filter by time range
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        # Filter by symbol
        if symbol:
            results = [e for e in results if e.symbol == symbol]

        # Filter by exchange
        if exchange:
            results = [e for e in results if e.exchange == exchange]

        # Filter by strategy
        if strategy_id:
            results = [e for e in results if e.strategy_id == strategy_id]

        # Limit results
        return results[-limit:]

    async def close(self) -> None:
        """Close storage (clear memory)"""
        logger.info(f"Closing memory storage ({len(self.events)} events stored)")
        self.events.clear()

    def get_all_events(self) -> List[LoggedEvent]:
        """Get all stored events"""
        return self.events.copy()

    def clear(self) -> None:
        """Clear all events"""
        self.events.clear()


class DatabaseLogStorage(LogStorage):
    """
    Database log storage

    Stores events to a database for persistent storage and complex queries.
    (Future implementation - would support PostgreSQL, MongoDB, etc.)
    """

    def __init__(self, connection_string: str):
        """
        Initialize database storage

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        logger.info("Database log storage initialized (not yet implemented)")

    async def store(self, event: LoggedEvent) -> None:
        """Store event to database"""
        # TODO: Implement database storage
        pass

    async def query(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[LoggedEvent]:
        """Query events from database"""
        # TODO: Implement database querying
        return []

    async def close(self) -> None:
        """Close database connection"""
        pass


logger.info("Log storage backends module loaded")
