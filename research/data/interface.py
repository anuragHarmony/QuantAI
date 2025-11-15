"""
Data Analytics Interfaces

Abstract interfaces for market data operations following SOLID principles.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal
import pandas as pd


@dataclass
class MarketData:
    """Market data container"""
    symbol: str
    timeframe: str
    data: pd.DataFrame  # OHLCV data
    start_date: datetime
    end_date: datetime
    source: str


@dataclass
class Pattern:
    """Detected pattern in market data"""
    pattern_type: str  # "trend", "support", "resistance", "breakout", etc.
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    confidence: float  # 0-1
    properties: Dict[str, Any]


@dataclass
class MarketRegime:
    """Market regime identification"""
    regime_type: str  # "trending", "ranging", "volatile", etc.
    symbol: str
    start_time: datetime
    end_time: Optional[datetime]
    confidence: float
    characteristics: Dict[str, Any]  # volatility, trend_strength, etc.


class IDataProvider(ABC):
    """
    Interface for market data providers

    Single Responsibility: Fetches and stores market data
    """

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        exchange: str = "binance"
    ) -> MarketData:
        """Fetch OHLCV candlestick data"""
        pass

    @abstractmethod
    async def fetch_tick_data(
        self,
        symbol: str,
        date: datetime,
        exchange: str = "binance"
    ) -> pd.DataFrame:
        """Fetch tick-by-tick data for a single day"""
        pass

    @abstractmethod
    async def get_available_symbols(
        self,
        exchange: str = "binance"
    ) -> List[str]:
        """Get list of available trading symbols"""
        pass

    @abstractmethod
    async def get_latest_price(
        self,
        symbol: str,
        exchange: str = "binance"
    ) -> Decimal:
        """Get latest price for a symbol"""
        pass


class IFeatureEngineer(ABC):
    """
    Interface for feature engineering

    Single Responsibility: Transforms raw data into features
    """

    @abstractmethod
    async def calculate_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate technical indicators

        Args:
            data: OHLCV DataFrame
            indicators: List of indicators to calculate (None = all)

        Returns:
            DataFrame with added indicator columns
        """
        pass

    @abstractmethod
    async def calculate_market_microstructure(
        self,
        tick_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate microstructure features from tick data

        Features:
        - Bid-ask spread
        - Order flow imbalance
        - Trade intensity
        - Price impact
        """
        pass

    @abstractmethod
    async def calculate_regime_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate market regime indicators

        Features:
        - Volatility regime
        - Trend strength
        - Mean reversion score
        """
        pass


class IPatternDetector(ABC):
    """
    Interface for pattern detection

    Single Responsibility: Identifies patterns in market data
    """

    @abstractmethod
    async def detect_trends(
        self,
        data: pd.DataFrame
    ) -> List[Pattern]:
        """
        Detect trends in price data

        Returns list of trend patterns with direction, strength, confidence
        """
        pass

    @abstractmethod
    async def detect_regimes(
        self,
        data: pd.DataFrame
    ) -> List[MarketRegime]:
        """
        Detect market regimes

        Returns list of market regimes (trending, ranging, volatile, etc.)
        """
        pass

    @abstractmethod
    async def detect_support_resistance(
        self,
        data: pd.DataFrame
    ) -> List[Pattern]:
        """
        Detect support and resistance levels

        Returns list of price levels with strength
        """
        pass

    @abstractmethod
    async def detect_seasonality(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns

        Returns:
        - Time of day patterns
        - Day of week patterns
        - Monthly patterns
        """
        pass


class IStatisticalAnalyzer(ABC):
    """
    Interface for statistical analysis

    Single Responsibility: Performs statistical tests and analysis
    """

    @abstractmethod
    async def test_stationarity(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """
        Test if series is stationary

        Returns:
        - is_stationary: bool
        - p_value: float
        - test_statistic: float
        - method: str (ADF, KPSS, etc.)
        """
        pass

    @abstractmethod
    async def test_autocorrelation(
        self,
        series: pd.Series,
        max_lag: int = 40
    ) -> Dict[str, Any]:
        """
        Test for autocorrelation

        Returns:
        - acf: array of autocorrelation values
        - pacf: array of partial autocorrelation values
        - significant_lags: list of significant lag values
        """
        pass

    @abstractmethod
    async def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Dict[str, Any]:
        """
        Test for cointegration between two series

        Returns:
        - is_cointegrated: bool
        - p_value: float
        - hedge_ratio: float
        """
        pass

    @abstractmethod
    async def calculate_correlations(
        self,
        data: pd.DataFrame,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix

        Args:
            data: DataFrame with multiple columns
            method: "pearson", "spearman", or "kendall"

        Returns:
            Correlation matrix
        """
        pass
