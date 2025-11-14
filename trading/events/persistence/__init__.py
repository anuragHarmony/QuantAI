"""
Event persistence for recording and replay

Components:
- ParquetEventWriter: Record events to Parquet files
- EventReader: Read and replay events

Usage (Recording):
    writer = ParquetEventWriter()
    writer.start()  # Start background flush
    await bus.subscribe("*", writer.write_event)  # Record all events

Usage (Replay):
    reader = EventReader(speedup=10.0)  # 10x speed
    async for event in reader.read_range(start, end):
        await bus.publish(event)
"""

from .writer import ParquetEventWriter
from .reader import EventReader, EVENT_TYPE_MAP

__all__ = [
    "ParquetEventWriter",
    "EventReader",
    "EVENT_TYPE_MAP",
]
