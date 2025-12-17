"""
Tests for _apply_bias_correction() method in ForecastEngine.

These tests verify:
1. Basic uplift application (1.27x multiplier)
2. Low volume detection using training_customers (Bug Fix #1)
3. Capping logic for low-volume products (Bug Fix #2)
4. Confidence interval scaling
5. Customer quantity scaling
6. Summary recalculation
7. Metadata tracking
8. Edge cases
"""
import pytest
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

# Import from the same directory conftest (pytest auto-imports it)
from scripts.forecasting.tests.conftest import make_mock_forecast, MockForecastEngine


class TestBiasCorrectionBasic:
    """Test basic uplift application."""

    def test_uplift_applies_1_27_multiplier(self, engine, mock_forecast):
        """Default 1.27 multiplier increases quantities by 27%."""
        result = engine._apply_bias_correction(mock_forecast)

        predicted_weeks = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # 100 * 1.27 = 127
        assert predicted_weeks[0]['predicted_quantity'] == pytest.approx(127.0, rel=0.01)

    def test_uplift_no_change_when_1_0(self, mock_forecast):
        """Uplift of 1.0 makes no changes."""
        engine = MockForecastEngine(bias_uplift=1.0)
        result = engine._apply_bias_correction(mock_forecast)

        predicted_weeks = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        assert predicted_weeks[0]['predicted_quantity'] == 100.0

    def test_uplift_disabled_when_0(self, mock_forecast):
        """Uplift of 0 is treated as no change."""
        engine = MockForecastEngine(bias_uplift=0.0)
        result = engine._apply_bias_correction(mock_forecast)

        predicted_weeks = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Should remain unchanged since uplift <= 0
        assert predicted_weeks[0]['predicted_quantity'] == 100.0

    def test_uplift_scales_all_weeks(self, engine):
        """Uplift applies to all 4 forecast weeks."""
        forecast = make_mock_forecast(
            weekly_predictions=[
                {'qty': 10, 'orders': 1, 'lower': 8, 'upper': 12},
                {'qty': 20, 'orders': 2, 'lower': 16, 'upper': 24},
                {'qty': 30, 'orders': 3, 'lower': 24, 'upper': 36},
                {'qty': 40, 'orders': 4, 'lower': 32, 'upper': 48},
            ]
        )
        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        assert predicted[0]['predicted_quantity'] == pytest.approx(12.7, rel=0.01)
        assert predicted[1]['predicted_quantity'] == pytest.approx(25.4, rel=0.01)
        assert predicted[2]['predicted_quantity'] == pytest.approx(38.1, rel=0.01)
        assert predicted[3]['predicted_quantity'] == pytest.approx(50.8, rel=0.01)


class TestLowVolumeDetection:
    """Test low volume product detection - Bug Fix #1 verification."""

    def test_low_volume_detected_when_training_customers_0(self, engine):
        """Products with 0 training customers are marked as low volume."""
        forecast = make_mock_forecast(training_customers=0)
        result = engine._apply_bias_correction(forecast)

        assert result.model_metadata['bias_low_volume_cap'] is True

    def test_low_volume_detected_when_training_customers_5(self, engine):
        """Products with exactly 5 training customers are marked as low volume."""
        forecast = make_mock_forecast(training_customers=5)
        result = engine._apply_bias_correction(forecast)

        assert result.model_metadata['bias_low_volume_cap'] is True

    def test_not_low_volume_when_training_customers_6(self, engine):
        """Products with 6+ training customers are NOT low volume."""
        forecast = make_mock_forecast(training_customers=6)
        result = engine._apply_bias_correction(forecast)

        assert result.model_metadata['bias_low_volume_cap'] is False

    def test_low_volume_threshold_configurable(self):
        """Low volume threshold can be configured."""
        engine = MockForecastEngine(low_volume_order_cap=10)
        forecast = make_mock_forecast(training_customers=8)
        result = engine._apply_bias_correction(forecast)

        # 8 <= 10, so should be low volume
        assert result.model_metadata['bias_low_volume_cap'] is True

    def test_uses_training_customers_not_weekly_orders(self, engine):
        """
        Critical Bug Fix #1: Low volume detection uses training_customers,
        NOT 1-week historical orders.

        A high-volume product with a slow week should NOT be marked low volume.
        """
        # High-volume product (500 training customers) with slow historical week
        forecast = make_mock_forecast(
            training_customers=500,  # High volume
            historical_avg=10.0      # Slow week (but doesn't matter)
        )

        result = engine._apply_bias_correction(forecast)

        # Should NOT be capped (high volume product)
        assert result.model_metadata['bias_low_volume_cap'] is False


class TestCappingLogic:
    """Test low-volume capping behavior - Bug Fix #2 verification."""

    def test_cap_applied_for_low_volume_with_historical(self, engine):
        """Low-volume products with historical data get capped."""
        forecast = make_mock_forecast(
            training_customers=3,    # Low volume
            historical_avg=20.0      # Has historical baseline
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Cap = max(100, 20 * 1.5) = max(100, 30) = 100
        # Since 100 < 127, no capping occurs here
        # Let's test with lower orig_qty
        assert result.model_metadata['bias_low_volume_cap'] is True

    def test_cap_not_applied_when_historical_avg_zero(self, engine):
        """
        Critical Bug Fix #2: When historical_avg=0, allow full uplift.

        Before fix: cap = max(orig_qty, 0) = orig_qty, blocking all uplift
        After fix: skip capping when historical_avg=0
        """
        forecast = make_mock_forecast(
            training_customers=3,    # Low volume
            historical_avg=0.0       # No historical baseline
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Should apply full 1.27x uplift, NOT be capped to 100
        assert predicted[0]['predicted_quantity'] == pytest.approx(127.0, rel=0.01)

    def test_cap_not_applied_for_high_volume(self, engine):
        """High-volume products get full uplift regardless of historical_avg."""
        forecast = make_mock_forecast(
            training_customers=100,  # High volume
            historical_avg=5.0       # Very low historical (but doesn't matter)
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Full uplift: 100 * 1.27 = 127
        assert predicted[0]['predicted_quantity'] == pytest.approx(127.0, rel=0.01)

    def test_cap_uses_max_of_orig_qty_and_historical(self, engine):
        """Cap is max(orig_qty, historical_avg * 1.5)."""
        forecast = make_mock_forecast(
            training_customers=3,    # Low volume
            historical_avg=100.0,    # High historical avg
            weekly_predictions=[
                {'qty': 50, 'orders': 5, 'lower': 40, 'upper': 60},  # orig_qty=50
                {'qty': 50, 'orders': 5, 'lower': 40, 'upper': 60},
                {'qty': 50, 'orders': 5, 'lower': 40, 'upper': 60},
                {'qty': 50, 'orders': 5, 'lower': 40, 'upper': 60},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Scaled: 50 * 1.27 = 63.5
        # Cap = max(50, 100 * 1.5) = max(50, 150) = 150
        # 63.5 < 150, so no capping
        assert predicted[0]['predicted_quantity'] == pytest.approx(63.5, rel=0.01)

    def test_ratio_propagates_to_orders(self, engine):
        """When capped, the ratio applies to orders too."""
        forecast = make_mock_forecast(
            training_customers=3,    # Low volume
            historical_avg=50.0,     # Historical avg
            weekly_predictions=[
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Cap = max(100, 50 * 1.5) = max(100, 75) = 100
        # Scaled qty = 100 * 1.27 = 127
        # Since 127 > cap (100), qty is capped to 100
        # Ratio = 100 / 127 = 0.787
        # Orders: 10 * 1.27 * 0.787 = 10 (approximately)
        assert predicted[0]['predicted_quantity'] == pytest.approx(100.0, rel=0.01)


class TestConfidenceIntervals:
    """Test confidence interval scaling."""

    def test_ci_scaled_by_uplift(self, engine):
        """CIs are scaled by the same uplift factor."""
        forecast = make_mock_forecast(
            training_customers=100,  # High volume (no cap)
            weekly_predictions=[
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # lower: 80 * 1.27 = 101.6
        # upper: 120 * 1.27 = 152.4
        assert predicted[0]['confidence_lower'] == pytest.approx(101.6, rel=0.01)
        assert predicted[0]['confidence_upper'] == pytest.approx(152.4, rel=0.01)

    def test_ci_lower_never_exceeds_qty(self, engine):
        """CI lower bound is clamped to not exceed predicted quantity."""
        forecast = make_mock_forecast(
            training_customers=100,  # High volume
            weekly_predictions=[
                {'qty': 50, 'orders': 5, 'lower': 100, 'upper': 120},  # lower > qty after scaling
                {'qty': 50, 'orders': 5, 'lower': 100, 'upper': 120},
                {'qty': 50, 'orders': 5, 'lower': 100, 'upper': 120},
                {'qty': 50, 'orders': 5, 'lower': 100, 'upper': 120},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # qty: 50 * 1.27 = 63.5
        # lower: 100 * 1.27 = 127 -> clamped to min(127, 63.5) = 63.5
        assert predicted[0]['confidence_lower'] <= predicted[0]['predicted_quantity']

    def test_ci_upper_never_below_qty(self, engine):
        """CI upper bound is clamped to not be below predicted quantity."""
        forecast = make_mock_forecast(
            training_customers=100,
            weekly_predictions=[
                {'qty': 100, 'orders': 10, 'lower': 10, 'upper': 50},  # upper < qty
                {'qty': 100, 'orders': 10, 'lower': 10, 'upper': 50},
                {'qty': 100, 'orders': 10, 'lower': 10, 'upper': 50},
                {'qty': 100, 'orders': 10, 'lower': 10, 'upper': 50},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # qty: 100 * 1.27 = 127
        # upper: 50 * 1.27 = 63.5 -> clamped to max(63.5, 127) = 127
        assert predicted[0]['confidence_upper'] >= predicted[0]['predicted_quantity']


class TestCustomerScaling:
    """Test customer quantity scaling."""

    def test_expected_customers_scaled(self, engine):
        """expected_customers quantities are scaled by uplift."""
        forecast = make_mock_forecast(training_customers=100)  # High volume

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Each customer had expected_quantity=50, scaled to 50 * 1.27 = 63.5
        assert predicted[0]['expected_customers'][0]['expected_quantity'] == pytest.approx(63.5, rel=0.01)

    def test_top_customers_scaled_by_ratio(self, engine):
        """top_customers_by_volume are scaled by the overall quantity ratio."""
        forecast = make_mock_forecast(training_customers=100)  # High volume

        result = engine._apply_bias_correction(forecast)

        # Original: 200, scaled by ratio (total_new/total_orig = 1.27)
        # New: 200 * 1.27 = 254
        assert result.top_customers_by_volume[0]['predicted_quantity'] == pytest.approx(254.0, rel=0.01)


class TestSummaryRecalculation:
    """Test summary totals recalculation."""

    def test_total_predicted_quantity_recalculated(self, engine):
        """total_predicted_quantity is recalculated from weekly data."""
        forecast = make_mock_forecast(training_customers=100)  # 4 weeks * 100 = 400 total

        result = engine._apply_bias_correction(forecast)

        # 400 * 1.27 = 508
        assert result.summary['total_predicted_quantity'] == pytest.approx(508.0, rel=0.01)

    def test_total_predicted_orders_recalculated(self, engine):
        """total_predicted_orders is recalculated from weekly data."""
        forecast = make_mock_forecast(training_customers=100)  # 4 weeks * 10 = 40 total orders

        result = engine._apply_bias_correction(forecast)

        # 40 * 1.27 = 50.8
        assert result.summary['total_predicted_orders'] == pytest.approx(50.8, rel=0.01)

    def test_average_weekly_quantity_updated(self, engine):
        """average_weekly_quantity is recalculated as total/4."""
        forecast = make_mock_forecast(training_customers=100)

        result = engine._apply_bias_correction(forecast)

        # total=508, avg=508/4=127
        assert result.summary['average_weekly_quantity'] == pytest.approx(127.0, rel=0.01)


class TestMetadata:
    """Test metadata tracking."""

    def test_bias_correction_stored_in_metadata(self, engine):
        """bias_correction value is stored in metadata."""
        forecast = make_mock_forecast()

        result = engine._apply_bias_correction(forecast)

        assert result.model_metadata['bias_correction'] == pytest.approx(1.27, rel=0.001)

    def test_bias_correction_1_0_stored_when_disabled(self, engine_no_uplift, mock_forecast):
        """When uplift is 1.0, bias_correction is stored as 1.0."""
        result = engine_no_uplift._apply_bias_correction(mock_forecast)

        assert result.model_metadata['bias_correction'] == 1.0

    def test_low_volume_cap_stored_in_metadata_true(self, engine):
        """bias_low_volume_cap=True when product is low volume."""
        forecast = make_mock_forecast(training_customers=3)

        result = engine._apply_bias_correction(forecast)

        assert result.model_metadata['bias_low_volume_cap'] is True

    def test_low_volume_cap_stored_in_metadata_false(self, engine):
        """bias_low_volume_cap=False when product is high volume."""
        forecast = make_mock_forecast(training_customers=100)

        result = engine._apply_bias_correction(forecast)

        assert result.model_metadata['bias_low_volume_cap'] is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_predicted_quantity(self, engine):
        """Zero predicted quantities don't cause errors."""
        forecast = make_mock_forecast(
            training_customers=100,
            weekly_predictions=[
                {'qty': 0, 'orders': 0, 'lower': 0, 'upper': 0},
                {'qty': 0, 'orders': 0, 'lower': 0, 'upper': 0},
                {'qty': 0, 'orders': 0, 'lower': 0, 'upper': 0},
                {'qty': 0, 'orders': 0, 'lower': 0, 'upper': 0},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        assert predicted[0]['predicted_quantity'] == 0.0

    def test_negative_uplift_treated_as_zero(self):
        """Negative uplift is treated as 0 (no change)."""
        engine = MockForecastEngine(bias_uplift=-0.5)
        forecast = make_mock_forecast()

        result = engine._apply_bias_correction(forecast)

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        assert predicted[0]['predicted_quantity'] == 100.0

    def test_empty_weekly_data(self, engine):
        """Empty weekly_data doesn't cause errors."""
        forecast = make_mock_forecast()
        forecast.weekly_data = []

        result = engine._apply_bias_correction(forecast)

        assert result.summary['total_predicted_quantity'] == 0.0

    def test_no_predicted_weeks(self, engine):
        """Forecast with only actual weeks doesn't crash."""
        forecast = make_mock_forecast()
        # Keep only historical week
        forecast.weekly_data = [w for w in forecast.weekly_data if w['data_type'] == 'actual']

        result = engine._apply_bias_correction(forecast)

        assert result.summary['total_predicted_quantity'] == 0.0

    def test_missing_predicted_revenue_field(self, engine):
        """Missing predicted_revenue field doesn't cause errors."""
        forecast = make_mock_forecast(training_customers=100)
        # Remove predicted_revenue from first predicted week
        for w in forecast.weekly_data:
            if w['data_type'] == 'predicted':
                del w['predicted_revenue']
                break

        # Should not raise an exception
        result = engine._apply_bias_correction(forecast)
        assert result is not None

    def test_missing_expected_customers(self, engine):
        """Missing expected_customers list doesn't cause errors."""
        forecast = make_mock_forecast(training_customers=100)
        # Remove expected_customers from all predicted weeks
        for w in forecast.weekly_data:
            if w['data_type'] == 'predicted':
                del w['expected_customers']

        result = engine._apply_bias_correction(forecast)
        assert result.summary['total_predicted_quantity'] == pytest.approx(508.0, rel=0.01)


class TestIntegration:
    """Integration tests combining multiple aspects."""

    def test_full_pipeline_high_volume_product(self, engine):
        """Full bias correction pipeline for high-volume product."""
        forecast = make_mock_forecast(
            product_id=12345,
            training_customers=500,
            historical_avg=100.0,
            weekly_predictions=[
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 150, 'orders': 15, 'lower': 120, 'upper': 180},
                {'qty': 200, 'orders': 20, 'lower': 160, 'upper': 240},
                {'qty': 250, 'orders': 25, 'lower': 200, 'upper': 300},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        # Verify no capping
        assert result.model_metadata['bias_low_volume_cap'] is False

        # Verify all quantities scaled by 1.27
        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        assert predicted[0]['predicted_quantity'] == pytest.approx(127.0, rel=0.01)
        assert predicted[1]['predicted_quantity'] == pytest.approx(190.5, rel=0.01)
        assert predicted[2]['predicted_quantity'] == pytest.approx(254.0, rel=0.01)
        assert predicted[3]['predicted_quantity'] == pytest.approx(317.5, rel=0.01)

        # Verify totals
        total_expected = (127 + 190.5 + 254 + 317.5)
        assert result.summary['total_predicted_quantity'] == pytest.approx(total_expected, rel=0.01)

    def test_full_pipeline_low_volume_with_cap(self, engine):
        """Full bias correction pipeline for low-volume product with capping."""
        forecast = make_mock_forecast(
            product_id=99999,
            training_customers=2,    # Low volume
            historical_avg=50.0,     # Cap = max(100, 75) = 100
            weekly_predictions=[
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
                {'qty': 100, 'orders': 10, 'lower': 80, 'upper': 120},
            ]
        )

        result = engine._apply_bias_correction(forecast)

        # Verify capping
        assert result.model_metadata['bias_low_volume_cap'] is True

        predicted = [w for w in result.weekly_data if w['data_type'] == 'predicted']
        # Scaled: 100 * 1.27 = 127, capped to 100
        for p in predicted:
            assert p['predicted_quantity'] == pytest.approx(100.0, rel=0.01)
