# QuantAI - Production-Ready Quantitative Trading Platform

A sophisticated, event-driven quantitative trading platform built with Python, supporting multiple exchanges, backtesting, and live trading through a unified interface.

## 🚀 Quick Start

### Run Live Trading (Today)
```bash
python run.py 2025-11-14 --config configs/strategies/my_strategy.yaml
```

### Run Backtest (Past Date)
```bash
python run.py 2024-01-01 --end-date 2024-12-31 --capital 100000
```

**That's it!** The system automatically detects whether to run live or backtest based on the date.

---

## ✨ Key Features

### 🏗️ Architecture
- **Event-Driven**: All components communicate via typed events
- **WebSocket-First**: Real-time market data for maximum stability
- **Async/Await**: Non-blocking, high-performance execution
- **SOLID Principles**: Clean, maintainable, extensible code
- **Type-Safe**: Comprehensive type hints throughout

### 📊 Exchange Support
- **Binance**: Spot, Futures, Perpetual (WebSocket + REST)
- **OKX**: Spot, Futures, Swap, Options (V5 API)
- **Bybit**: Spot, Linear, Inverse, Options (V5 API - 2025)
- **Simulated**: Realistic backtesting with configurable slippage

### 🎯 Trading Features
- **Universe System**: Organize instruments with tickers and filters
- **Multi-Strategy**: Run multiple strategies simultaneously
- **Risk Management**: 6 built-in risk checks
- **Portfolio Tracking**: Real-time P&L, positions, metrics
- **Order Management**: Smart routing with rate limiting

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🎓 Usage Example

```python
from trading.strategy.base import BaseStrategy
from trading.events import TickEvent

class MyStrategy(BaseStrategy):
    async def on_tick(self, tick: TickEvent):
        if self.should_buy(tick):
            await self.place_market_order(
                symbol=tick.symbol,
                side="buy",
                quantity=1.0,
                exchange="binance"
            )
```

**Config:**
```yaml
name: "My Strategy"
strategy_id: "my_strategy_v1"
mode: live_trading  # or simulation

subscriptions:
  symbols: [BTC/USDT]
  exchanges: [binance]

parameters:
  my_param: 42
```

**Run:**
```bash
python run.py 2025-11-14 --config configs/strategies/my_strategy.yaml
```

---

## 🏛️ Architecture

```
Strategy → EventBus → Portfolio Manager
                   ↓
                  OMS (Risk + Routing)
                   ↓
              Exchanges (Binance/OKX/Bybit)
```

---

## 📊 Performance Metrics

- Total P&L, Return %, Max Drawdown
- Sharpe, Sortino, Calmar ratios
- Win rate, Profit factor
- Transaction costs tracking

---

## 🛡️ Risk Management

6 built-in checks:
1. Position limits
2. Order size limits
3. Max loss limits
4. Concentration limits
5. Daily loss limits
6. Capital requirements

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

**Built with ❤️ for quantitative trading**
