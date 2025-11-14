"""
Event persistence writer

Writes events to Parquet files for later replay.
Partitioned by date, event type, and exchange for efficient queries.

Directory structure:
data/events/
  2024/
    01/
      15/
        tick_binance.parquet
        tick_okex.parquet
        order_filled_binance.parquet
"""
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Any
from collections import defaultdict
import asyncio
from loguru import logger

from ..base import BaseEvent


class ParquetEventWriter:
    """
    Writes events to Parquet files with partitioning

    Features:
    - Automatic partitioning by date/event_type/exchange
    - Batch writing for performance
    - Background flushing
    - Compression
    """

    def __init__(
        self,
        base_path: str = "./data/events",
        batch_size: int = 1000,
        flush_interval: float = 5.0,  # seconds
        compression: str = "snappy"
    ):
        """
        Initialize writer

        Args:
            base_path: Base directory for event files
            batch_size: Write after this many events
            flush_interval: Flush every N seconds
            compression: Parquet compression (snappy, gzip, lz4)
        """
        self.base_path = Path(base_path)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.compression = compression

        # Buffered events by partition key
        self.buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._closed = False

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ParquetEventWriter at {base_path}")

    def start(self) -> None:
        """Start background flush task"""
        if not self._flush_task:
            self._flush_task = asyncio.create_task(self._flush_loop())
            logger.info("Started background flush task")

    async def write_event(self, event: BaseEvent) -> None:
        """
        Write single event

        Args:
            event: Event to write
        """
        if self._closed:
            logger.warning("Attempted to write to closed writer")
            return

        # Get partition key
        partition_key = self._get_partition_key(event)

        # Convert to dict and add to buffer
        event_dict = self._event_to_dict(event)
        self.buffers[partition_key].append(event_dict)

        # Check if buffer should be flushed
        if len(self.buffers[partition_key]) >= self.batch_size:
            await self._flush_partition(partition_key)

    async def write_batch(self, events: list[BaseEvent]) -> None:
        """
        Write multiple events

        Args:
            events: Events to write
        """
        for event in events:
            await self.write_event(event)

    def _get_partition_key(self, event: BaseEvent) -> tuple:
        """
        Get partition key for event

        Returns:
            Tuple of (date, event_type, exchange)
        """
        event_date = event.timestamp.date()
        event_type = event.event_type
        exchange = event.exchange or "system"

        return (event_date, event_type, exchange)

    def _get_partition_path(self, partition_key: tuple) -> Path:
        """
        Get file path for partition

        Args:
            partition_key: (date, event_type, exchange)

        Returns:
            Path to parquet file
        """
        event_date, event_type, exchange = partition_key

        # data/events/2024/01/15/tick_binance.parquet
        file_path = (
            self.base_path
            / str(event_date.year)
            / f"{event_date.month:02d}"
            / f"{event_date.day:02d}"
            / f"{event_type}_{exchange}.parquet"
        )

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path

    def _event_to_dict(self, event: BaseEvent) -> dict[str, Any]:
        """
        Convert event to dictionary for storage

        Args:
            event: Event to convert

        Returns:
            Dictionary representation
        """
        event_dict = event.dict()

        # Convert datetime to ISO string
        if "timestamp" in event_dict:
            event_dict["timestamp"] = event_dict["timestamp"].isoformat()

        # Convert exchange_timestamp if present
        if "exchange_timestamp" in event_dict:
            event_dict["exchange_timestamp"] = event_dict["exchange_timestamp"].isoformat()

        # Convert fill_timestamp if present
        if "fill_timestamp" in event_dict:
            event_dict["fill_timestamp"] = event_dict["fill_timestamp"].isoformat()

        # Convert Decimals to strings
        for key, value in event_dict.items():
            if hasattr(value, "__class__") and value.__class__.__name__ == "Decimal":
                event_dict[key] = str(value)

        return event_dict

    async def _flush_partition(self, partition_key: tuple) -> None:
        """
        Flush events in partition to disk

        Args:
            partition_key: Partition to flush
        """
        if partition_key not in self.buffers or not self.buffers[partition_key]:
            return

        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Get events from buffer
            events = self.buffers[partition_key]
            self.buffers[partition_key] = []  # Clear buffer

            # Convert to DataFrame
            df = pd.DataFrame(events)

            # Get file path
            file_path = self._get_partition_path(partition_key)

            # Append or create file
            if file_path.exists():
                # Append to existing file
                existing_table = pq.read_table(file_path)
                new_table = pa.Table.from_pandas(df)
                combined_table = pa.concat_tables([existing_table, new_table])

                pq.write_table(
                    combined_table,
                    file_path,
                    compression=self.compression
                )
            else:
                # Create new file
                pq.write_table(
                    pa.Table.from_pandas(df),
                    file_path,
                    compression=self.compression
                )

            logger.debug(f"Flushed {len(events)} events to {file_path}")

        except Exception as e:
            logger.error(f"Error flushing partition {partition_key}: {e}")
            # Put events back in buffer
            self.buffers[partition_key].extend(events)

    async def _flush_loop(self) -> None:
        """Background task to flush buffers periodically"""
        while not self._closed:
            try:
                await asyncio.sleep(self.flush_interval)

                # Flush all partitions
                for partition_key in list(self.buffers.keys()):
                    if self.buffers[partition_key]:
                        await self._flush_partition(partition_key)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    async def flush(self) -> None:
        """Flush all buffers immediately"""
        for partition_key in list(self.buffers.keys()):
            await self._flush_partition(partition_key)

    async def close(self) -> None:
        """Close writer and flush remaining events"""
        self._closed = True

        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()

        logger.info("Closed ParquetEventWriter")

    def get_statistics(self) -> dict[str, Any]:
        """Get writer statistics"""
        buffered_events = sum(len(events) for events in self.buffers.values())

        return {
            "buffered_events": buffered_events,
            "num_partitions": len(self.buffers),
            "closed": self._closed,
            "base_path": str(self.base_path)
        }
