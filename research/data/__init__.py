"""
Data Analytics Module

Provides market data fetching, feature engineering, and pattern detection.
"""

from .interface import (
    IDataProvider,
    IFeatureEngineer,
    IPatternDetector,
    IStatisticalAnalyzer,
    MarketData,
    Pattern,
    MarketRegime,
)

from .binance_provider import BinanceDataProvider
from .feature_engineer import TechnicalFeatureEngineer
from .pattern_detector import SimplePatternDetector

__all__ = [
    # Interfaces
    "IDataProvider",
    "IFeatureEngineer",
    "IPatternDetector",
    "IStatisticalAnalyzer",

    # Data types
    "MarketData",
    "Pattern",
    "MarketRegime",

    # Implementations
    "BinanceDataProvider",
    "TechnicalFeatureEngineer",
    "SimplePatternDetector",
]
