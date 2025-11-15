# Backtesting Module

Professional-grade vectorized backtesting engine for trading strategies with comprehensive metrics and visualization.

## Features

✅ **Vectorized Execution** - Fast backtesting using pandas
✅ **Realistic Simulation** - Commission, slippage, stop losses
✅ **Comprehensive Metrics** - 20+ performance indicators
✅ **Dynamic Strategy Loading** - Load strategies from code strings
✅ **Professional Visualization** - Beautiful charts and reports
✅ **Long & Short** - Support for both long and short positions
✅ **Risk Management** - Stop loss, take profit, position sizing

## Quick Start

```python
import asyncio
from backtesting import BacktestEngine, load_strategy_from_code

# Your strategy code
strategy_code = """
class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        # Add your indicators
        return data

    def generate_signals(self, data):
        # Generate buy/sell signals
        data['signal'] = 0  # 1=buy, -1=sell, 0=hold
        return data
"""

async def run_backtest():
    # Load strategy
    strategy = load_strategy_from_code(strategy_code)

    # Create engine
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )

    # Run backtest
    result = await engine.run_backtest(
        strategy_class=strategy,
        data=your_ohlcv_data,
        symbol="BTC/USDT",
        timeframe="1h"
    )

    # View results
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Total Return: {result.metrics.total_return*100:.2f}%")
    print(f"Win Rate: {result.metrics.win_rate*100:.1f}%")

asyncio.run(run_backtest())
```

## Components

### 1. Backtest Engine

The core backtesting engine simulates trading:

```python
from backtesting import BacktestEngine

engine = BacktestEngine(
    initial_capital=10000.0,      # Starting capital
    commission=0.001,              # 0.1% per trade
    slippage=0.0005,               # 0.05% slippage
    position_size_pct=1.0,         # 100% of capital per position
    max_positions=1,               # Max simultaneous positions
    enable_shorting=True           # Allow short selling
)

result = await engine.run_backtest(
    strategy_class=MyStrategy,
    data=ohlcv_dataframe,
    strategy_parameters={'rsi_period': 14},
    symbol="BTC/USDT",
    timeframe="1h"
)
```

### 2. Performance Metrics

Comprehensive performance analysis:

```python
from backtesting import calculate_comprehensive_metrics

metrics = result.metrics

# Returns
print(f"Total Return: {metrics.total_return*100:.2f}%")
print(f"Annual Return: {metrics.annualized_return*100:.2f}%")

# Risk-adjusted returns
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")

# Risk metrics
print(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
print(f"Volatility: {metrics.volatility*100:.2f}%")

# Trading metrics
print(f"Total Trades: {metrics.total_trades}")
print(f"Win Rate: {metrics.win_rate*100:.1f}%")
print(f"Profit Factor: {metrics.profit_factor:.2f}")

# Profit metrics
print(f"Avg Win: ${metrics.avg_win:.2f}")
print(f"Avg Loss: ${metrics.avg_loss:.2f}")
print(f"Expectancy: ${metrics.expectancy:.2f}")
```

### 3. Strategy Loader

Load strategies dynamically from code:

```python
from backtesting import load_strategy_from_code

# From code string
strategy_class = load_strategy_from_code(code_string)

# From file
from backtesting import StrategyLoader
strategy_class = StrategyLoader.load_from_file("my_strategy.py")

# Validate strategy
StrategyLoader.validate_strategy(strategy_class)
```

### 4. Visualization

Create professional charts:

```python
from backtesting.visualization import BacktestVisualizer

visualizer = BacktestVisualizer()

# Full report with multiple charts
visualizer.create_full_report(
    result,
    save_path="backtest_report.png"
)

# Just equity curve
visualizer.create_equity_curve_only(
    result,
    save_path="equity_curve.png"
)
```

The full report includes:
- Equity curve with trade markers
- Drawdown chart
- Returns distribution
- Monthly returns heatmap
- Cumulative P&L by trade
- Metrics table

## Strategy Development

### Base Strategy Class

All strategies must inherit from `BaseStrategy`:

```python
class BaseStrategy:
    """Base class for trading strategies"""

    def __init__(self, name: str = "Strategy", parameters: dict = None):
        self.name = name
        self.parameters = parameters or {}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        raise NotImplementedError

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals (1=buy, -1=sell, 0=hold)"""
        raise NotImplementedError
```

### Example Strategy

```python
import pandas as pd
import numpy as np

class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy

    Entry: RSI < 30 (oversold)
    Exit: RSI > 50 or stop loss
    """

    def __init__(self, parameters: dict = None):
        super().__init__(name="RSI_MeanReversion", parameters=parameters)

        # Default parameters
        self.parameters.setdefault('rsi_period', 14)
        self.parameters.setdefault('rsi_oversold', 30)
        self.parameters.setdefault('rsi_exit', 50)
        self.parameters.setdefault('stop_loss_pct', 2.0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI"""
        df = data.copy()

        period = self.parameters['rsi_period']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals"""
        df = self.calculate_indicators(data)

        df['signal'] = 0

        # Buy when RSI is oversold
        df.loc[df['rsi'] < self.parameters['rsi_oversold'], 'signal'] = 1

        # Sell when RSI returns to neutral
        df.loc[df['rsi'] > self.parameters['rsi_exit'], 'signal'] = -1

        return df
```

## How Backtesting Works

### 1. Signal Generation

The strategy generates signals for each candle:
- `signal = 1`: Buy signal (enter long)
- `signal = -1`: Sell signal (exit long or enter short)
- `signal = 0`: No action (hold)

### 2. Position Management

The engine manages positions:
- Opens positions when signal is 1 (long) or -1 (short)
- Closes positions on opposite signals
- Applies stop losses and take profits
- Only one position at a time (configurable)

### 3. Transaction Costs

Realistic costs are applied:
- **Commission**: Percentage fee per trade (e.g., 0.1%)
- **Slippage**: Price impact (e.g., 0.05%)
- Total cost = (commission + slippage) × trade_value

### 4. Stop Loss & Take Profit

Automatic risk management:
```python
strategy_parameters = {
    'stop_loss_pct': 2.0,      # Stop if loss > 2%
    'take_profit_pct': 5.0     # Take profit if gain > 5%
}
```

The engine checks intra-candle extremes (high/low) to detect stop/TP hits.

### 5. Equity Calculation

Equity is tracked at each candle:
```
Equity = Cash + Unrealized P&L
```

Unrealized P&L updates with current price for open positions.

## Performance Metrics Explained

### Returns

- **Total Return**: Overall profit/loss percentage
- **Annualized Return**: Return adjusted to yearly basis

### Risk-Adjusted Returns

- **Sharpe Ratio**: Excess return per unit of risk (volatility)
  - `> 1.0` = Good
  - `> 2.0` = Very good
  - `> 3.0` = Excellent

- **Sortino Ratio**: Like Sharpe, but only penalizes downside volatility
  - Generally higher than Sharpe
  - Better for asymmetric strategies

- **Calmar Ratio**: Return divided by max drawdown
  - Higher is better
  - Good for comparing drawdown risk

### Risk Metrics

- **Max Drawdown**: Largest peak-to-trough decline
  - `< 20%` = Good
  - `< 10%` = Very good

- **Volatility**: Annualized standard deviation of returns
- **Downside Deviation**: Volatility of negative returns only

### Trading Metrics

- **Win Rate**: Percentage of profitable trades
  - `> 50%` = Above random
  - `> 60%` = Good
  - Quality matters more than quantity

- **Profit Factor**: Total wins / Total losses
  - `> 1.0` = Profitable
  - `> 2.0` = Very profitable

- **Expectancy**: Average profit per trade
  - Must be positive for profitability

## Data Format

The engine expects OHLCV data as a pandas DataFrame:

```python
import pandas as pd

data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.DatetimeIndex([...]))
```

Requirements:
- ✅ DatetimeIndex (timestamps)
- ✅ Columns: open, high, low, close, volume
- ✅ No missing values (forward fill if needed)
- ✅ Sorted chronologically

## Testing

Run the comprehensive test suite:

```bash
python examples/test_backtesting.py
```

This tests:
1. Strategy loading from code
2. Backtest execution
3. Metrics calculation
4. Visualization generation
5. Parameter optimization

## Integration with AI Agent

The backtesting engine is fully integrated with the AI research agent:

```python
from research.agent.agent_factory import AgentFactory

# Create agent (now uses REAL backtesting!)
agent = await AgentFactory.create_simple_agent()

# Run autonomous research
session = await agent.run_research_loop(
    objective="Discover profitable strategies",
    max_iterations=5
)

# Agent will:
# 1. Generate strategy hypothesis
# 2. Convert to Python code
# 3. Run REAL backtest (not mock!)
# 4. Analyze results
# 5. Learn and iterate
```

## Best Practices

### 1. Use Enough Data

- Minimum: 3-6 months
- Recommended: 1-2 years
- More data = more reliable results

### 2. Consider Transaction Costs

- Always include commission and slippage
- Real trading has costs
- Small edges disappear with costs

### 3. Avoid Overfitting

- Don't over-optimize parameters
- Test on out-of-sample data
- Simple strategies often work better

### 4. Validate Metrics

- Sharpe > 1.0 minimum
- At least 30 trades for significance
- Max drawdown < 20% ideally

### 5. Check Win Rate vs. Expectancy

- Can be profitable with < 50% win rate if winners > losers
- High win rate + low expectancy = likely curve-fit

## Advanced Usage

### Walk-Forward Analysis

```python
# Split data into training and testing periods
train_data = data['2023-01-01':'2023-06-30']
test_data = data['2023-07-01':'2023-12-31']

# Backtest on training
train_result = await engine.run_backtest(strategy_class, train_data)

# Validate on testing
test_result = await engine.run_backtest(strategy_class, test_data)

# Compare metrics
print(f"Train Sharpe: {train_result.metrics.sharpe_ratio:.2f}")
print(f"Test Sharpe: {test_result.metrics.sharpe_ratio:.2f}")
```

### Parameter Optimization

```python
best_sharpe = -999
best_params = None

for rsi_period in [10, 14, 20]:
    for rsi_oversold in [20, 30, 40]:
        result = await engine.run_backtest(
            strategy_class,
            data,
            strategy_parameters={
                'rsi_period': rsi_period,
                'rsi_oversold': rsi_oversold
            }
        )

        if result.metrics.sharpe_ratio > best_sharpe:
            best_sharpe = result.metrics.sharpe_ratio
            best_params = {'rsi_period': rsi_period, 'rsi_oversold': rsi_oversold}

print(f"Best parameters: {best_params}")
print(f"Best Sharpe: {best_sharpe:.2f}")
```

## Limitations

- **No market impact**: Assumes you can trade any size
- **Perfect execution**: Assumes orders fill at expected prices
- **Historical data**: Past performance ≠ future results
- **No regime changes**: Market conditions change
- **Simplified slippage**: Real slippage varies

Always paper trade before live trading!

## Dependencies

```bash
pip install pandas numpy loguru matplotlib scipy
```

Optional for visualization:
```bash
pip install matplotlib seaborn
```

## License

See main project LICENSE file.
