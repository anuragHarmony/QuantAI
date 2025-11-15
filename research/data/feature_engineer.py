"""
Feature Engineering Implementation

Calculates technical indicators and features from market data.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .interface import IFeatureEngineer


class TechnicalFeatureEngineer(IFeatureEngineer):
    """
    Calculate technical indicators

    Single Responsibility: Transform raw OHLCV into features
    """

    def __init__(self):
        logger.info("Initialized TechnicalFeatureEngineer")

    async def calculate_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate technical indicators

        Available indicators:
        - SMA, EMA
        - RSI
        - MACD
        - Bollinger Bands
        - ATR
        - ADX
        - Stochastic
        - OBV, VWAP
        """
        df = data.copy()

        # If no indicators specified, calculate all
        if indicators is None:
            indicators = [
                "sma", "ema", "rsi", "macd", "bollinger",
                "atr", "adx", "stochastic", "obv", "vwap"
            ]

        # Calculate each indicator
        for indicator in indicators:
            try:
                if indicator == "sma":
                    df = self._add_sma(df)
                elif indicator == "ema":
                    df = self._add_ema(df)
                elif indicator == "rsi":
                    df = self._add_rsi(df)
                elif indicator == "macd":
                    df = self._add_macd(df)
                elif indicator == "bollinger":
                    df = self._add_bollinger_bands(df)
                elif indicator == "atr":
                    df = self._add_atr(df)
                elif indicator == "adx":
                    df = self._add_adx(df)
                elif indicator == "stochastic":
                    df = self._add_stochastic(df)
                elif indicator == "obv":
                    df = self._add_obv(df)
                elif indicator == "vwap":
                    df = self._add_vwap(df)

            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")

        logger.info(f"Calculated {len(indicators)} indicators")
        return df

    def _add_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    def _add_ema(self, df: pd.DataFrame, periods: List[int] = [12, 26, 50]) -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Add MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def _add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: int = 2
    ) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        df[f'bb_middle_{period}'] = sma
        df[f'bb_upper_{period}'] = sma + (std * std_dev)
        df[f'bb_lower_{period}'] = sma - (std * std_dev)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma

        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        return df

    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index"""
        # Calculate +DM and -DM
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        # Calculate smoothed values
        atr = pd.Series(true_range).rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df[f'adx_{period}'] = dx.rolling(window=period).mean()

        df[f'plus_di_{period}'] = plus_di
        df[f'minus_di_{period}'] = minus_di

        return df

    def _add_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df[f'stoch_k_{k_period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_d_{k_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()

        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df

    async def calculate_market_microstructure(
        self,
        tick_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate microstructure features from tick data

        Note: Requires tick-by-tick data with bid/ask
        """
        df = tick_data.copy()

        if 'bid' in df.columns and 'ask' in df.columns:
            # Bid-ask spread
            df['spread'] = df['ask'] - df['bid']
            df['spread_bps'] = 10000 * df['spread'] / ((df['bid'] + df['ask']) / 2)

            # Order flow imbalance (if trade direction available)
            if 'side' in df.columns:
                df['buy_volume'] = df['volume'].where(df['side'] == 'buy', 0)
                df['sell_volume'] = df['volume'].where(df['side'] == 'sell', 0)

                # Rolling imbalance
                window = 100
                df['order_flow_imbalance'] = (
                    (df['buy_volume'].rolling(window).sum() - df['sell_volume'].rolling(window).sum()) /
                    (df['buy_volume'].rolling(window).sum() + df['sell_volume'].rolling(window).sum())
                )

        logger.info("Calculated microstructure features")
        return df

    async def calculate_regime_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate market regime indicators
        """
        df = data.copy()

        # Volatility (rolling standard deviation)
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        df['volatility_50'] = df['close'].pct_change().rolling(50).std()

        # Trend strength (ADX if not already calculated)
        if 'adx_14' not in df.columns:
            df = self._add_adx(df)

        # Mean reversion score (distance from moving average)
        if 'sma_20' not in df.columns:
            df = self._add_sma(df, [20])

        df['mean_reversion_score'] = (df['close'] - df['sma_20']) / df['sma_20']

        # Autocorrelation (simple lag-1)
        df['autocorr_1'] = df['close'].pct_change().rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        )

        logger.info("Calculated regime features")
        return df


logger.info("Technical feature engineer loaded")
