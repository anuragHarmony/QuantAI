"""
Pattern Detection Implementation

Detects patterns and regimes in market data.
"""
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from .interface import IPatternDetector, Pattern, MarketRegime


class SimplePatternDetector(IPatternDetector):
    """
    Simple pattern detector using statistical methods

    Single Responsibility: Identify patterns in market data
    """

    def __init__(self):
        logger.info("Initialized SimplePatternDetector")

    async def detect_trends(
        self,
        data: pd.DataFrame
    ) -> List[Pattern]:
        """
        Detect trends using multiple methods
        """
        patterns = []

        # Method 1: Linear regression slope
        if len(data) < 20:
            return patterns

        # Calculate trend over last 20, 50, 100 periods
        for window in [20, 50, 100]:
            if len(data) < window:
                continue

            recent_data = data.tail(window)
            x = np.arange(len(recent_data))
            y = recent_data['close'].values

            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)

            # Normalize slope by price
            slope_pct = (slope / y[0]) * 100

            # Determine direction and strength
            if abs(slope_pct) > 0.5:  # > 0.5% slope
                direction = "up" if slope > 0 else "down"
                strength = min(abs(slope_pct) / 5.0, 1.0)  # Normalize

                pattern = Pattern(
                    pattern_type="trend",
                    symbol=data.index.name if hasattr(data.index, 'name') else "unknown",
                    timeframe=f"{window}period",
                    start_time=recent_data.index[0],
                    end_time=recent_data.index[-1],
                    confidence=strength,
                    properties={
                        "direction": direction,
                        "slope": float(slope),
                        "slope_pct": float(slope_pct),
                        "method": "linear_regression"
                    }
                )

                patterns.append(pattern)

        logger.debug(f"Detected {len(patterns)} trend patterns")
        return patterns

    async def detect_regimes(
        self,
        data: pd.DataFrame
    ) -> List[MarketRegime]:
        """
        Detect market regimes using volatility and trend
        """
        regimes = []

        if len(data) < 50:
            return regimes

        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()

        # Calculate trend strength (ADX proxy)
        # Simple trend strength: correlation with linear trend
        def calc_trend_strength(window_data):
            if len(window_data) < 10:
                return 0
            x = np.arange(len(window_data))
            correlation = np.corrcoef(x, window_data)[0, 1]
            return abs(correlation)

        trend_strength = data['close'].rolling(20).apply(calc_trend_strength, raw=True)

        # Current regime (last 20 periods)
        recent_vol = volatility.iloc[-20:].mean() if len(volatility) > 20 else 0
        recent_trend = trend_strength.iloc[-20:].mean() if len(trend_strength) > 20 else 0

        # Classify regime
        if recent_vol > volatility.quantile(0.7):
            regime_type = "volatile"
        elif recent_vol < volatility.quantile(0.3):
            regime_type = "low_volatility"
        elif recent_trend > 0.7:
            regime_type = "trending"
        elif recent_trend < 0.3:
            regime_type = "ranging"
        else:
            regime_type = "mixed"

        regime = MarketRegime(
            regime_type=regime_type,
            symbol=data.index.name if hasattr(data.index, 'name') else "unknown",
            start_time=data.index[-20] if len(data) > 20 else data.index[0],
            end_time=None,  # Current regime
            confidence=0.7,  # Simple heuristic
            characteristics={
                "volatility": float(recent_vol),
                "trend_strength": float(recent_trend),
                "method": "statistical"
            }
        )

        regimes.append(regime)

        logger.debug(f"Current regime: {regime_type}")
        return regimes

    async def detect_support_resistance(
        self,
        data: pd.DataFrame
    ) -> List[Pattern]:
        """
        Detect support and resistance levels

        Uses local minima/maxima and price clustering
        """
        patterns = []

        if len(data) < 50:
            return patterns

        # Find local minima (support) and maxima (resistance)
        from scipy.signal import argrelextrema

        # Support levels (local minima in 'low')
        support_indices = argrelextrema(data['low'].values, np.less, order=5)[0]

        for idx in support_indices:
            level = data['low'].iloc[idx]

            pattern = Pattern(
                pattern_type="support",
                symbol=data.index.name if hasattr(data.index, 'name') else "unknown",
                timeframe="detected",
                start_time=data.index[idx],
                end_time=data.index[idx],
                confidence=0.6,
                properties={
                    "level": float(level),
                    "method": "local_minima"
                }
            )

            patterns.append(pattern)

        # Resistance levels (local maxima in 'high')
        resistance_indices = argrelextrema(data['high'].values, np.greater, order=5)[0]

        for idx in resistance_indices:
            level = data['high'].iloc[idx]

            pattern = Pattern(
                pattern_type="resistance",
                symbol=data.index.name if hasattr(data.index, 'name') else "unknown",
                timeframe="detected",
                start_time=data.index[idx],
                end_time=data.index[idx],
                confidence=0.6,
                properties={
                    "level": float(level),
                    "method": "local_maxima"
                }
            )

            patterns.append(pattern)

        logger.debug(f"Detected {len(patterns)} support/resistance levels")
        return patterns

    async def detect_seasonality(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns (time of day, day of week, etc.)
        """
        seasonality = {}

        if len(data) < 100:
            return seasonality

        # Add time features if not present
        if not isinstance(data.index, pd.DatetimeIndex):
            return seasonality

        df = data.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day

        # Returns by hour
        df['returns'] = df['close'].pct_change()

        # Hour of day pattern
        hour_stats = df.groupby('hour')['returns'].agg(['mean', 'std', 'count'])
        if not hour_stats.empty:
            seasonality['hour_of_day'] = {
                'best_hours': hour_stats['mean'].nlargest(3).index.tolist(),
                'worst_hours': hour_stats['mean'].nsmallest(3).index.tolist(),
                'stats': hour_stats.to_dict()
            }

        # Day of week pattern
        dow_stats = df.groupby('day_of_week')['returns'].agg(['mean', 'std', 'count'])
        if not dow_stats.empty:
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            seasonality['day_of_week'] = {
                'best_days': [days[i] for i in dow_stats['mean'].nlargest(2).index.tolist()],
                'worst_days': [days[i] for i in dow_stats['mean'].nsmallest(2).index.tolist()],
                'stats': dow_stats.to_dict()
            }

        logger.debug("Detected seasonality patterns")
        return seasonality


logger.info("Simple pattern detector loaded")
