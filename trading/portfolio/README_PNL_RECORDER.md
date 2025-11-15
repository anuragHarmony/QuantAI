# PnL Recorder

Automatic P&L file recording for live trading strategies.

## Overview

The `PnLRecorder` automatically records portfolio P&L metrics to files at regular intervals (default: 10 seconds). This is essential for:
- Real-time monitoring of trading performance
- Post-trade analysis and debugging
- Performance tracking across sessions
- Creating equity curves and visualizations

## Features

✓ **Event-driven**: Subscribes to `PortfolioValueEvent` - no manual intervention needed
✓ **Configurable intervals**: Dump every N seconds (default 10s)
✓ **Multiple formats**: CSV (human-readable) or JSON Lines (programmatic)
✓ **Timestamped sessions**: Auto-generates unique filenames per session
✓ **Buffered writes**: Efficient in-memory buffering before disk writes
✓ **Graceful shutdown**: Final dump on stop to capture last state

## Quick Start

```python
from trading.events.bus import InMemoryEventBus
from trading.portfolio import PortfolioManager, PnLRecorder

# Setup event bus and portfolio
event_bus = InMemoryEventBus()
portfolio = PortfolioManager(event_bus, initial_capital=10000)

# Create PnL recorder (dumps every 10 seconds to CSV)
pnl_recorder = PnLRecorder(
    event_bus=event_bus,
    dump_interval=10.0,
    output_dir="./data/pnl",
    format="csv"
)

# Start both
await portfolio.start()
await pnl_recorder.start()

# ... trading happens ...

# Stop (performs final dump)
await pnl_recorder.stop()
await portfolio.stop()
```

## Configuration Options

### Constructor Parameters

```python
PnLRecorder(
    event_bus: IEventBus,          # Event bus for subscribing
    dump_interval: float = 10.0,   # Seconds between dumps
    output_dir: str = "./data/pnl",# Output directory
    format: str = "csv",           # 'csv' or 'json'
    session_name: Optional[str] = None  # Auto-generated if None
)
```

### Dump Intervals

| Interval | Use Case |
|----------|----------|
| 1-5s | High-frequency strategies, detailed monitoring |
| 10s | **Default** - good balance for most strategies |
| 30-60s | Slower strategies, reduced I/O overhead |

## Output Formats

### CSV Format

**File**: `session_YYYYMMDD_HHMMSS_pnl.csv`

```csv
timestamp,cash,equity,total_value,unrealized_pnl,realized_pnl,total_pnl,return_pct,initial_value
2025-11-15T11:10:17.788392,100000,0,100000,0,0,0,0,100000
2025-11-15T11:10:27.788392,100000,0,102500,0,2500,2500,2.5,100000
2025-11-15T11:10:37.788392,100000,0,105200,0,5200,5200,5.2,100000
```

**Pros**:
- Human-readable
- Easy to import into Excel/Google Sheets
- Simple analysis with pandas: `pd.read_csv('file.csv')`

### JSON Lines Format

**File**: `session_YYYYMMDD_HHMMSS_pnl.jsonl`

```json
{"timestamp": "2025-11-15T11:10:17.788392", "cash": "100000", "total_value": "100000", "total_pnl": "0", "return_pct": "0"}
{"timestamp": "2025-11-15T11:10:27.788392", "cash": "100000", "total_value": "102500", "total_pnl": "2500", "return_pct": "2.5"}
```

**Pros**:
- Streaming-friendly (one JSON object per line)
- Easy parsing: `json.loads(line)` per line
- Smaller file size for large datasets

## Recorded Metrics

Each snapshot contains:

| Field | Description |
|-------|-------------|
| `timestamp` | Exact time of snapshot (ISO 8601) |
| `cash` | Available cash balance |
| `equity` | Market value of open positions |
| `total_value` | Total portfolio value (cash + equity) |
| `unrealized_pnl` | P&L from open positions |
| `realized_pnl` | P&L from closed positions |
| `total_pnl` | Total P&L (realized + unrealized) |
| `return_pct` | Return percentage vs initial capital |
| `initial_value` | Starting capital |

## Advanced Usage

### Custom Session Names

```python
pnl_recorder = PnLRecorder(
    event_bus=event_bus,
    dump_interval=10.0,
    session_name="btc_momentum_live_v2"
)
# Output: btc_momentum_live_v2_pnl.csv
```

### Convenience Function

```python
from trading.portfolio import create_pnl_recorder

pnl_recorder = create_pnl_recorder(
    event_bus=event_bus,
    dump_interval=5.0,
    format="json"
)
```

### Query Snapshots

```python
# Get all recorded snapshots
snapshots = pnl_recorder.get_snapshots()
print(f"Recorded {len(snapshots)} snapshots")

# Get latest snapshot
latest = pnl_recorder.get_latest_snapshot()
print(f"Current PnL: {latest['total_pnl']}")

# Get file path
filepath = pnl_recorder.get_filepath()
print(f"Data saved to: {filepath}")

# Get statistics
stats = pnl_recorder.get_statistics()
print(stats)
# Output:
# {
#   'session_name': 'session_20251115_110916',
#   'format': 'csv',
#   'dump_interval': 10.0,
#   'filepath': 'data/pnl/session_20251115_110916_pnl.csv',
#   'snapshots_recorded': 42,
#   'is_running': True,
#   'latest_snapshot_time': datetime(2025, 11, 15, 11, 16, 37)
# }
```

## Integration with Trading Systems

### Live Trading Bot

```python
async def run_trading_bot():
    event_bus = InMemoryEventBus()

    # Portfolio
    portfolio = PortfolioManager(event_bus, initial_capital=50000)

    # PnL Recording (every 10 seconds)
    pnl_recorder = PnLRecorder(
        event_bus=event_bus,
        dump_interval=10.0,
        session_name=f"live_bot_{datetime.now().strftime('%Y%m%d')}"
    )

    # Order management, market data, etc.
    oms = OrderManagementSystem(...)
    market_data = MarketDataStream(...)

    # Start everything
    await portfolio.start()
    await pnl_recorder.start()
    await oms.start()
    await market_data.start()

    try:
        # Run strategy
        await strategy.run()
    finally:
        # Graceful shutdown - ensures final PnL dump
        await pnl_recorder.stop()
        await portfolio.stop()
```

### Multiple Strategies

```python
# Record PnL for multiple strategies independently
recorders = {}

for strategy_name in ["momentum", "mean_reversion", "arbitrage"]:
    recorder = PnLRecorder(
        event_bus=event_bus,
        dump_interval=10.0,
        session_name=f"{strategy_name}_live",
        output_dir=f"./data/pnl/{strategy_name}"
    )
    await recorder.start()
    recorders[strategy_name] = recorder
```

## Post-Processing Recorded Data

### Analyze with Pandas

```python
import pandas as pd

# Load CSV
df = pd.read_csv('data/pnl/session_20251115_110916_pnl.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Calculate statistics
print(f"Final Return: {df['return_pct'].iloc[-1]:.2f}%")
print(f"Max Drawdown: {df['total_pnl'].min()}")
print(f"Peak PnL: {df['total_pnl'].max()}")

# Plot equity curve
df['total_value'].plot(title='Equity Curve')
```

### Load JSON Lines

```python
import json

snapshots = []
with open('data/pnl/session_pnl.jsonl', 'r') as f:
    for line in f:
        snapshot = json.loads(line)
        snapshots.append(snapshot)

# Convert to DataFrame
df = pd.DataFrame(snapshots)
df['total_pnl'] = df['total_pnl'].astype(float)
```

## Performance Considerations

### I/O Overhead

- CSV append: ~0.1-0.5ms per dump (negligible)
- JSON append: ~0.05-0.2ms per dump
- Recommended: Keep `dump_interval` ≥ 1 second

### Memory Usage

- Snapshots are buffered in memory via `get_snapshots()`
- Each snapshot: ~200 bytes
- 1 hour at 10s interval = 360 snapshots = ~72KB
- **Safe for long-running strategies**

### Disk Space

| Interval | Session Duration | Approx Size (CSV) |
|----------|------------------|-------------------|
| 10s | 1 hour | 3-5 KB |
| 10s | 24 hours | 70-100 KB |
| 10s | 1 month | 2-3 MB |

## Error Handling

The PnLRecorder is designed to be robust:

```python
# Errors in dump loop are logged but don't crash the recorder
# This ensures trading continues even if disk I/O fails

2025-11-15 11:10:28 | ERROR | Failed to dump PnL snapshot: [Errno 28] No space left on device
# ... recorder continues, attempts next dump in 10s
```

**Best practices**:
- Monitor disk space in production
- Set up log alerts for repeated dump failures
- Use log rotation for long-running sessions

## Examples

See `examples/pnl_recorder_example.py` for complete working examples:

```bash
# Run all examples
PYTHONPATH=/home/user/QuantAI python examples/pnl_recorder_example.py

# Output files will be in ./data/pnl/
```

## Architecture

```
┌─────────────────┐
│  PortfolioMgr   │
│   (calculates   │
│      P&L)       │
└────────┬────────┘
         │ publishes
         ▼
┌─────────────────┐
│   EventBus      │ ◄─── subscribes
└────────┬────────┘
         │ PortfolioValueEvent
         ▼
┌─────────────────┐
│  PnLRecorder    │
│                 │
│  ┌───────────┐  │
│  │ In-Memory │  │
│  │  Buffer   │  │
│  └─────┬─────┘  │
│        │ every 10s
│        ▼         │
│  ┌───────────┐  │
│  │ CSV/JSON  │  │
│  │   File    │  │
│  └───────────┘  │
└─────────────────┘
```

## Design Rationale

**Why not use logger for PnL dumps?**
- Mixing application logs with data files creates parsing complexity
- Loguru rotation is time-based, not data-structure-aware
- PnL data needs structured format (CSV/JSON), not log entries

**Why event-driven instead of polling?**
- More efficient - reacts to changes vs constant polling
- Decoupled - PnLRecorder doesn't need direct reference to Portfolio
- Scalable - multiple recorders can subscribe independently

**Why separate class instead of built into Portfolio?**
- Single Responsibility Principle
- Optional feature - not all strategies need file recording
- Easier to test and maintain

## FAQ

**Q: Can I change dump interval while running?**
A: No - set it on initialization. Stop and start new recorder if needed.

**Q: What happens if I don't call stop()?**
A: Last snapshot since previous dump won't be saved. Always call `stop()`.

**Q: Can I dump to database instead of files?**
A: Not built-in. Create custom class subscribing to `PortfolioValueEvent`.

**Q: Does it slow down my strategy?**
A: No - dumps happen asynchronously every N seconds, not on every trade.

**Q: Can I append to existing file across sessions?**
A: No - each session creates new timestamped file. Merge files in post-processing.

## See Also

- `trading/portfolio/manager.py` - PortfolioManager implementation
- `trading/portfolio/persistence.py` - Portfolio state persistence
- `examples/pnl_recorder_example.py` - Working examples
