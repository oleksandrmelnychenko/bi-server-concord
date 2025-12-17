"""
Pytest fixtures for forecasting tests.
"""
import pytest
import os
import sys
from unittest.mock import Mock, MagicMock

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.forecasting.core.product_aggregator import ProductForecast


def make_mock_forecast(
    product_id: int = 12345,
    training_customers: int = 100,
    historical_avg: float = 50.0,
    weekly_predictions: list = None
) -> ProductForecast:
    """
    Create a mock ProductForecast for testing.

    Args:
        product_id: Product ID for the forecast
        training_customers: Number of customers used to train the model (for low-volume detection)
        historical_avg: Historical average quantity (used for capping)
        weekly_predictions: List of dicts with qty, orders, lower, upper for each week

    Returns:
        ProductForecast object configured for testing
    """
    weekly_data = [
        # Historical week (1 week of actual data)
        {
            'week_start': '2025-07-21',
            'week_end': '2025-07-25',
            'data_type': 'actual',
            'quantity': historical_avg,
            'orders': 5,
            'predicted_quantity': 0.0,
            'predicted_orders': 0.0,
            'confidence_lower': 0.0,
            'confidence_upper': 0.0,
            'expected_customers': []
        }
    ]

    # Add prediction weeks (defaults to 4 weeks with qty=100 each)
    if weekly_predictions is None:
        weekly_predictions = [
            {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
            {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
            {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
            {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
        ]

    for i, wp in enumerate(weekly_predictions):
        weekly_data.append({
            'week_start': f'2025-08-{4+i*7:02d}',
            'week_end': f'2025-08-{8+i*7:02d}',
            'data_type': 'predicted',
            'quantity': 0.0,
            'orders': 0,
            'predicted_quantity': float(wp.get('qty', 100)),
            'predicted_orders': float(wp.get('orders', 10)),
            'confidence_lower': float(wp.get('lower', 80)),
            'confidence_upper': float(wp.get('upper', 120)),
            'predicted_revenue': None,
            'expected_customers': [
                {'customer_id': 1, 'expected_quantity': 50.0},
                {'customer_id': 2, 'expected_quantity': 50.0}
            ]
        })

    total_qty = sum(wp.get('qty', 100) for wp in weekly_predictions)
    total_orders = sum(wp.get('orders', 10) for wp in weekly_predictions)

    return ProductForecast(
        product_id=product_id,
        product_name='Test Product',
        forecast_period_weeks=4,
        historical_weeks=1,
        summary={
            'total_predicted_quantity': float(total_qty),
            'total_predicted_orders': float(total_orders),
            'total_predicted_revenue': None,
            'average_weekly_quantity': float(total_qty / 4),
            'historical_average': float(historical_avg),
            'active_customers': training_customers,
            'at_risk_customers': 0
        },
        weekly_data=weekly_data,
        top_customers_by_volume=[
            {'customer_id': 1, 'predicted_quantity': 200.0}
        ],
        at_risk_customers=[],
        model_metadata={
            'model_type': 'hybrid_arima',
            'training_customers': training_customers,
            'forecast_accuracy_estimate': 0.75,
            'seasonality_detected': False,
            'statistical_methods': ['bayesian_inference']
        }
    )


class MockForecastEngine:
    """
    Minimal mock of ForecastEngine that only has _apply_bias_correction.
    This allows testing bias correction without needing a DB connection.
    """

    def __init__(self, bias_uplift=1.27, low_volume_order_cap=5, low_volume_clamp_multiplier=1.5):
        self.bias_uplift = bias_uplift
        self.low_volume_order_cap = low_volume_order_cap
        self.low_volume_clamp_multiplier = low_volume_clamp_multiplier
        self.forecast_weeks = 4

    def _apply_bias_correction(self, forecast: ProductForecast) -> ProductForecast:
        """
        Apply systematic uplift to predicted quantities/orders and clamp low-volume items.
        This is a copy of the real implementation for testing without DB.
        """
        try:
            uplift = max(self.bias_uplift, 0.0)
        except Exception:
            uplift = 1.0

        if uplift <= 0 or abs(uplift - 1.0) < 1e-3:
            forecast.model_metadata['bias_correction'] = 1.0
            forecast.model_metadata['bias_low_volume_cap'] = False
            return forecast

        # Use training_customers (product's 3-year customer count) for low-volume detection
        training_customers = forecast.model_metadata.get('training_customers', 0)
        historical_avg = float(forecast.summary.get('historical_average', 0.0) or 0.0)
        low_volume = training_customers <= self.low_volume_order_cap

        total_orig_qty = 0.0
        total_new_qty = 0.0
        total_orig_orders = 0.0
        total_new_orders = 0.0

        for week in forecast.weekly_data:
            if week.get('data_type') != 'predicted':
                continue

            orig_qty = float(week.get('predicted_quantity', 0.0) or 0.0)
            orig_orders = float(week.get('predicted_orders', 0.0) or 0.0)
            orig_lower = float(week.get('confidence_lower', 0.0) or 0.0)
            orig_upper = float(week.get('confidence_upper', 0.0) or 0.0)
            orig_revenue = week.get('predicted_revenue')

            scaled_qty = orig_qty * uplift
            scaled_orders = orig_orders * uplift
            scaled_lower = orig_lower * uplift
            scaled_upper = orig_upper * uplift
            scaled_revenue = None if orig_revenue is None else orig_revenue * uplift

            ratio = 1.0
            if low_volume and historical_avg > 0:
                # Only cap low-volume products when we have meaningful historical baseline
                cap = max(orig_qty, historical_avg * self.low_volume_clamp_multiplier)
                if cap > 0 and scaled_qty > cap:
                    ratio = cap / scaled_qty
                    scaled_qty = cap

            if ratio != 1.0:
                scaled_orders *= ratio
                scaled_lower *= ratio
                scaled_upper *= ratio
                if scaled_revenue is not None:
                    scaled_revenue *= ratio

            week['predicted_quantity'] = round(scaled_qty, 1)
            week['predicted_orders'] = round(scaled_orders, 1)
            week['confidence_lower'] = round(min(scaled_lower, scaled_qty), 1)
            week['confidence_upper'] = round(max(scaled_upper, scaled_qty), 1)
            if 'predicted_revenue' in week:
                week['predicted_revenue'] = None if scaled_revenue is None else round(scaled_revenue, 2)

            for cust in week.get('expected_customers', []):
                if 'expected_quantity' in cust:
                    cust['expected_quantity'] = round(
                        cust['expected_quantity'] * uplift * ratio, 1
                    )

            total_orig_qty += orig_qty
            total_new_qty += scaled_qty
            total_orig_orders += orig_orders
            total_new_orders += scaled_orders

        # Scale top customers proportionally
        total_orig_qty = max(total_orig_qty, 1e-9)
        qty_ratio = total_new_qty / total_orig_qty
        for cust in forecast.top_customers_by_volume:
            if 'predicted_quantity' in cust:
                cust['predicted_quantity'] = round(cust['predicted_quantity'] * qty_ratio, 1)

        # Recompute summary totals
        forecast.summary['total_predicted_quantity'] = round(total_new_qty, 1)
        forecast.summary['total_predicted_orders'] = round(total_new_orders, 1)
        forecast.summary['average_weekly_quantity'] = round(
            total_new_qty / max(1, self.forecast_weeks), 1
        )

        if 'total_predicted_revenue' in forecast.summary:
            forecast.summary['total_predicted_revenue'] = None

        forecast.model_metadata['bias_correction'] = round(uplift, 3)
        forecast.model_metadata['bias_low_volume_cap'] = low_volume

        return forecast


@pytest.fixture
def mock_forecast():
    """Provide a default mock forecast for testing."""
    return make_mock_forecast()


@pytest.fixture
def engine():
    """Provide a MockForecastEngine with default settings (1.27x uplift)."""
    return MockForecastEngine(bias_uplift=1.27, low_volume_order_cap=5, low_volume_clamp_multiplier=1.5)


@pytest.fixture
def engine_no_uplift():
    """Provide a MockForecastEngine with uplift disabled."""
    return MockForecastEngine(bias_uplift=1.0, low_volume_order_cap=5, low_volume_clamp_multiplier=1.5)


@pytest.fixture
def engine_high_cap():
    """Provide a MockForecastEngine with high low-volume threshold."""
    return MockForecastEngine(bias_uplift=1.27, low_volume_order_cap=100, low_volume_clamp_multiplier=1.5)
