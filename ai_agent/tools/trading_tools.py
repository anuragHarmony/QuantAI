"""
Example trading tools for AI agents
"""
from typing import Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
import numpy as np
from loguru import logger

from ai_agent.tools.base import BaseTool, ToolParameter, ToolParameterType, ToolResult


class MarketDataResult(BaseModel):
    """Market data result"""
    symbol: str
    data: dict[str, Any]  # Will contain OHLCV data
    start_date: str
    end_date: str
    num_bars: int


class GetMarketDataTool(BaseTool):
    """Tool to fetch historical market data"""

    @property
    def name(self) -> str:
        return "get_market_data"

    @property
    def description(self) -> str:
        return "Fetch historical market data (OHLCV) for a given symbol and date range"

    @property
    def category(self) -> Optional[str]:
        return "market_data"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type=ToolParameterType.STRING,
                description="Stock symbol (e.g., AAPL, MSFT)",
                required=True
            ),
            ToolParameter(
                name="start_date",
                type=ToolParameterType.STRING,
                description="Start date in YYYY-MM-DD format",
                required=True
            ),
            ToolParameter(
                name="end_date",
                type=ToolParameterType.STRING,
                description="End date in YYYY-MM-DD format",
                required=False
            ),
            ToolParameter(
                name="interval",
                type=ToolParameterType.STRING,
                description="Data interval (1d, 1h, 1m)",
                required=False,
                default="1d",
                enum=["1m", "5m", "15m", "1h", "1d"]
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult[MarketDataResult]:
        """Fetch market data using yfinance"""
        try:
            import yfinance as yf

            symbol = kwargs["symbol"]
            start_date = kwargs["start_date"]
            end_date = kwargs.get("end_date", datetime.now().strftime("%Y-%m-%d"))
            interval = kwargs.get("interval", "1d")

            logger.info(f"Fetching market data for {symbol} from {start_date} to {end_date}")

            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                return ToolResult(
                    success=False,
                    error=f"No data found for {symbol}"
                )

            # Convert to dict for serialization
            data_dict = {
                "open": df["Open"].tolist(),
                "high": df["High"].tolist(),
                "low": df["Low"].tolist(),
                "close": df["Close"].tolist(),
                "volume": df["Volume"].tolist(),
                "dates": [d.strftime("%Y-%m-%d") for d in df.index]
            }

            result = MarketDataResult(
                symbol=symbol,
                data=data_dict,
                start_date=start_date,
                end_date=end_date,
                num_bars=len(df)
            )

            logger.info(f"Fetched {len(df)} bars for {symbol}")

            return ToolResult(
                success=True,
                result=result,
                metadata={"num_bars": len(df)}
            )

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return ToolResult(success=False, error=str(e))


class CalculateIndicatorTool(BaseTool):
    """Tool to calculate technical indicators"""

    @property
    def name(self) -> str:
        return "calculate_indicator"

    @property
    def description(self) -> str:
        return "Calculate technical indicators (SMA, EMA, RSI, MACD, etc.) on price data"

    @property
    def category(self) -> Optional[str]:
        return "technical_analysis"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="indicator",
                type=ToolParameterType.STRING,
                description="Indicator type",
                required=True,
                enum=["SMA", "EMA", "RSI", "MACD", "BBANDS", "ATR"]
            ),
            ToolParameter(
                name="prices",
                type=ToolParameterType.ARRAY,
                description="Price data array",
                required=True,
                items={"type": "number"}
            ),
            ToolParameter(
                name="period",
                type=ToolParameterType.INTEGER,
                description="Indicator period/window",
                required=False,
                default=14
            ),
            ToolParameter(
                name="params",
                type=ToolParameterType.OBJECT,
                description="Additional indicator-specific parameters",
                required=False
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult[dict[str, Any]]:
        """Calculate technical indicator"""
        try:
            indicator = kwargs["indicator"].upper()
            prices = np.array(kwargs["prices"])
            period = kwargs.get("period", 14)
            params = kwargs.get("params", {})

            logger.info(f"Calculating {indicator} with period {period}")

            if indicator == "SMA":
                result = self._calculate_sma(prices, period)
            elif indicator == "EMA":
                result = self._calculate_ema(prices, period)
            elif indicator == "RSI":
                result = self._calculate_rsi(prices, period)
            elif indicator == "MACD":
                result = self._calculate_macd(prices, **params)
            elif indicator == "BBANDS":
                result = self._calculate_bbands(prices, period, **params)
            elif indicator == "ATR":
                highs = np.array(params.get("highs", prices))
                lows = np.array(params.get("lows", prices))
                result = self._calculate_atr(highs, lows, prices, period)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported indicator: {indicator}"
                )

            return ToolResult(
                success=True,
                result={"indicator": indicator, "values": result.tolist()}
            )

        except Exception as e:
            logger.error(f"Failed to calculate indicator: {e}")
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        if down == 0:
            return np.full_like(prices, 100.0)

        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]

            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            rs = up / down if down != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs) if rs != 0 else 100.0

        return rsi

    @staticmethod
    def _calculate_macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> np.ndarray:
        """Calculate MACD"""
        ema_fast = pd.Series(prices).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return (macd - signal).values

    @staticmethod
    def _calculate_bbands(
        prices: np.ndarray,
        period: int,
        num_std: float = 2.0
    ) -> np.ndarray:
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        # Return middle band
        return sma.values

    @staticmethod
    def _calculate_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Average True Range"""
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))

        ranges = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(ranges).rolling(window=period).mean().values


class GenerateSignalsTool(BaseTool):
    """Tool to generate trading signals based on strategy rules"""

    @property
    def name(self) -> str:
        return "generate_signals"

    @property
    def description(self) -> str:
        return "Generate buy/sell signals based on technical indicators and strategy rules"

    @property
    def category(self) -> Optional[str]:
        return "signal_generation"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="strategy",
                type=ToolParameterType.STRING,
                description="Strategy type",
                required=True,
                enum=["MA_CROSSOVER", "RSI_REVERSAL", "MACD_CROSSOVER", "BBANDS_BREAKOUT"]
            ),
            ToolParameter(
                name="prices",
                type=ToolParameterType.ARRAY,
                description="Price data",
                required=True,
                items={"type": "number"}
            ),
            ToolParameter(
                name="params",
                type=ToolParameterType.OBJECT,
                description="Strategy parameters",
                required=False
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult[dict[str, Any]]:
        """Generate trading signals"""
        try:
            strategy = kwargs["strategy"]
            prices = np.array(kwargs["prices"])
            params = kwargs.get("params", {})

            logger.info(f"Generating signals for strategy: {strategy}")

            if strategy == "MA_CROSSOVER":
                signals = self._ma_crossover(prices, **params)
            elif strategy == "RSI_REVERSAL":
                signals = self._rsi_reversal(prices, **params)
            elif strategy == "MACD_CROSSOVER":
                signals = self._macd_crossover(prices, **params)
            elif strategy == "BBANDS_BREAKOUT":
                signals = self._bbands_breakout(prices, **params)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported strategy: {strategy}"
                )

            num_buy = np.sum(signals == 1)
            num_sell = np.sum(signals == -1)

            return ToolResult(
                success=True,
                result={
                    "signals": signals.tolist(),
                    "num_buy": int(num_buy),
                    "num_sell": int(num_sell)
                },
                metadata={
                    "strategy": strategy,
                    "num_signals": int(num_buy + num_sell)
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return ToolResult(success=False, error=str(e))

    def _ma_crossover(
        self,
        prices: np.ndarray,
        fast_period: int = 10,
        slow_period: int = 50
    ) -> np.ndarray:
        """Moving Average Crossover Strategy"""
        fast_ma = pd.Series(prices).rolling(window=fast_period).mean().values
        slow_ma = pd.Series(prices).rolling(window=slow_period).mean().values

        signals = np.zeros(len(prices))

        # Buy when fast crosses above slow
        signals[1:] = np.where(
            (fast_ma[1:] > slow_ma[1:]) & (fast_ma[:-1] <= slow_ma[:-1]),
            1,  # Buy signal
            np.where(
                (fast_ma[1:] < slow_ma[1:]) & (fast_ma[:-1] >= slow_ma[:-1]),
                -1,  # Sell signal
                0
            )
        )

        return signals

    def _rsi_reversal(
        self,
        prices: np.ndarray,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70
    ) -> np.ndarray:
        """RSI Reversal Strategy"""
        rsi = CalculateIndicatorTool._calculate_rsi(prices, period)

        signals = np.zeros(len(prices))

        # Buy when RSI crosses above oversold
        signals[1:] = np.where(
            (rsi[1:] > oversold) & (rsi[:-1] <= oversold),
            1,  # Buy signal
            np.where(
                (rsi[1:] < overbought) & (rsi[:-1] >= overbought),
                -1,  # Sell signal
                0
            )
        )

        return signals

    def _macd_crossover(self, prices: np.ndarray, **params: Any) -> np.ndarray:
        """MACD Crossover Strategy"""
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        signal = params.get("signal_period", 9)

        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - macd_signal

        signals = np.zeros(len(prices))

        # Buy when MACD crosses above signal
        signals[1:] = np.where(
            (histogram.values[1:] > 0) & (histogram.values[:-1] <= 0),
            1,
            np.where(
                (histogram.values[1:] < 0) & (histogram.values[:-1] >= 0),
                -1,
                0
            )
        )

        return signals

    def _bbands_breakout(
        self,
        prices: np.ndarray,
        period: int = 20,
        num_std: float = 2.0
    ) -> np.ndarray:
        """Bollinger Bands Breakout Strategy"""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        signals = np.zeros(len(prices))

        # Buy when price crosses above lower band
        signals[1:] = np.where(
            (prices[1:] > lower.values[1:]) & (prices[:-1] <= lower.values[:-1]),
            1,
            np.where(
                (prices[1:] < upper.values[1:]) & (prices[:-1] >= upper.values[:-1]),
                -1,
                0
            )
        )

        return signals


class CalculateMetricsTool(BaseTool):
    """Tool to calculate backtest performance metrics"""

    @property
    def name(self) -> str:
        return "calculate_metrics"

    @property
    def description(self) -> str:
        return "Calculate performance metrics (Sharpe, returns, drawdown, etc.) from returns series"

    @property
    def category(self) -> Optional[str]:
        return "performance_analysis"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="returns",
                type=ToolParameterType.ARRAY,
                description="Returns series",
                required=True,
                items={"type": "number"}
            ),
            ToolParameter(
                name="risk_free_rate",
                type=ToolParameterType.NUMBER,
                description="Annual risk-free rate",
                required=False,
                default=0.02
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult[dict[str, Any]]:
        """Calculate performance metrics"""
        try:
            returns = np.array(kwargs["returns"])
            risk_free_rate = kwargs.get("risk_free_rate", 0.02)

            # Remove NaN values
            returns = returns[~np.isnan(returns)]

            if len(returns) == 0:
                return ToolResult(success=False, error="No valid returns data")

            metrics = {
                "total_return": float(np.prod(1 + returns) - 1),
                "annualized_return": float(np.mean(returns) * 252),  # Assuming daily returns
                "volatility": float(np.std(returns) * np.sqrt(252)),
                "sharpe_ratio": self._calculate_sharpe(returns, risk_free_rate),
                "max_drawdown": self._calculate_max_drawdown(returns),
                "win_rate": float(np.sum(returns > 0) / len(returns)),
                "num_trades": int(len(returns)),
            }

            logger.info(f"Calculated metrics: Sharpe={metrics['sharpe_ratio']:.2f}, "
                       f"Return={metrics['total_return']:.2%}")

            return ToolResult(success=True, result=metrics)

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
