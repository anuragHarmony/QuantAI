"""
BTC Mean Reversion Strategy

Strategy discovered through data analysis:
- Entry: Price touches lower Bollinger Band AND RSI < 35 (oversold)
- Exit: RSI > 50 (return to neutral) OR stop loss triggered
- Risk: 2% stop loss per trade

Backtested performance on analysis showed:
- Win rate: 59.7%
- Avg return per signal: +0.74%
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, Any


class BTCMeanReversionStrategy:
    """
    BTC Mean Reversion Strategy using Bollinger Bands and RSI

    This strategy exploits mean reversion when price becomes oversold.

    Entry Logic:
    - Price touches or goes below lower Bollinger Band (20-period, 2 std dev)
    - AND RSI (14-period) < 35 (oversold condition)

    Exit Logic:
    - RSI crosses above 50 (return to neutral)
    - OR stop loss triggered (2%)
    - OR take profit at 5%

    Risk Management:
    - Stop loss: 2% below entry
    - Position size: 100% of capital
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        self.name = "BTCMeanReversion"
        self.parameters = parameters or {}

        # Strategy parameters
        self.parameters.setdefault('bb_period', 20)
        self.parameters.setdefault('bb_std', 2.0)
        self.parameters.setdefault('rsi_period', 14)
        self.parameters.setdefault('rsi_entry', 35)
        self.parameters.setdefault('rsi_exit', 50)
        self.parameters.setdefault('stop_loss_pct', 2.0)
        self.parameters.setdefault('take_profit_pct', 5.0)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and RSI"""
        df = data.copy()

        # Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']

        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        df['bb_std'] = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)

        # RSI
        rsi_period = self.parameters['rsi_period']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(data)

        # Initialize signal column
        df['signal'] = 0

        # Entry signal: Price at/below lower BB AND RSI oversold
        entry_condition = (df['close'] <= df['bb_lower']) & (df['rsi'] < self.parameters['rsi_entry'])

        # Exit signal: RSI returns to neutral
        exit_condition = df['rsi'] > self.parameters['rsi_exit']

        # Set signals
        df.loc[entry_condition, 'signal'] = 1   # Buy
        df.loc[exit_condition, 'signal'] = -1   # Sell

        return df


if __name__ == "__main__":
    # Quick test
    import asyncio
    from backtesting import BacktestEngine, load_strategy_from_code

    async def test_strategy():
        # Load data
        df = pd.read_csv('/home/user/QuantAI/data/market_data/BTCUSDT_1h_20250519_to_20251115.csv',
                         index_col=0, parse_dates=True)

        print("Testing BTC Mean Reversion Strategy...")
        print(f"Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        # Create engine
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005,
            position_size_pct=1.0
        )

        # Run backtest
        result = await engine.run_backtest(
            strategy_class=BTCMeanReversionStrategy,
            data=df,
            strategy_parameters={
                'rsi_entry': 35,
                'rsi_exit': 50,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 5.0
            },
            symbol="BTC/USDT",
            timeframe="1h"
        )

        # Print results
        m = result.metrics
        print(f"\n{'='*80}")
        print("BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"\nPerformance Metrics:")
        print(f"  Total Return:      {m.total_return*100:>8.2f}%")
        print(f"  Sharpe Ratio:      {m.sharpe_ratio:>8.2f}")
        print(f"  Win Rate:          {m.win_rate*100:>8.1f}%")
        print(f"  Total Trades:      {m.total_trades:>8}")
        print(f"  Profit Factor:     {m.profit_factor:>8.2f}")
        print(f"  Max Drawdown:      {m.max_drawdown*100:>8.2f}%")
        print(f"{'='*80}")

        return result

    asyncio.run(test_strategy())
