"""
Product Forecasting System

Customer-based B2B forecasting with production-grade statistical analysis.
Uses 6+ years of historical data for accurate predictions.
"""

from .core.pattern_analyzer import PatternAnalyzer, CustomerProductPattern
from .core.customer_predictor import CustomerPredictor, CustomerPrediction
from .core.product_aggregator import ProductAggregator
from .core.forecast_engine import ForecastEngine

__all__ = [
    'PatternAnalyzer',
    'CustomerProductPattern',
    'CustomerPredictor',
    'CustomerPrediction',
    'ProductAggregator',
    'ForecastEngine',
]

__version__ = '1.0.0'
