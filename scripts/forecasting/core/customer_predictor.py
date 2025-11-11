#!/usr/bin/env python3

import numpy as np
from scipy import stats
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .pattern_analyzer import CustomerProductPattern

logger = logging.getLogger(__name__)

@dataclass
class CustomerPrediction:
    customer_id: int
    product_id: int

    expected_order_date: datetime
    date_confidence_lower: datetime
    date_confidence_upper: datetime
    date_stddev_days: float

    expected_quantity: float
    quantity_confidence_lower: float
    quantity_confidence_upper: float
    quantity_stddev: float

    probability_orders_this_period: float
    weekly_probabilities: List[Tuple[datetime, float]]

    status: str
    churn_probability: float
    prediction_confidence: float

    reorder_cycle_days: float
    consistency_score: float
    days_since_last_order: int

class CustomerPredictor:

    def __init__(self, forecast_horizon_days: int = 90):
        self.forecast_horizon_days = forecast_horizon_days
        logger.info(f"CustomerPredictor initialized with {forecast_horizon_days} day horizon")

    def predict_next_order(
        self,
        pattern: CustomerProductPattern,
        as_of_date: str
    ) -> Optional[CustomerPrediction]:
        try:
            as_of_dt = datetime.fromisoformat(as_of_date)

            if not self._is_predictable(pattern):
                logger.debug(f"Customer {pattern.customer_id} not predictable")
                return None

            expected_date, date_lower, date_upper, date_stddev = self._predict_order_date(
                pattern, as_of_dt
            )

            expected_qty, qty_lower, qty_upper, qty_stddev = self._predict_order_quantity(
                pattern
            )

            prob_orders = self._calculate_order_probability(
                pattern, as_of_dt, expected_date, date_stddev
            )

            weekly_probs = self._generate_weekly_probabilities(
                as_of_dt, expected_date, date_stddev
            )

            pred_confidence = self._calculate_prediction_confidence(
                pattern, prob_orders, date_stddev
            )

            return CustomerPrediction(
                customer_id=pattern.customer_id,
                product_id=pattern.product_id,
                expected_order_date=expected_date,
                date_confidence_lower=date_lower,
                date_confidence_upper=date_upper,
                date_stddev_days=date_stddev,
                expected_quantity=expected_qty,
                quantity_confidence_lower=qty_lower,
                quantity_confidence_upper=qty_upper,
                quantity_stddev=qty_stddev,
                probability_orders_this_period=prob_orders,
                weekly_probabilities=weekly_probs,
                status=pattern.status,
                churn_probability=pattern.churn_probability,
                prediction_confidence=pred_confidence,
                reorder_cycle_days=pattern.reorder_cycle_median,
                consistency_score=pattern.consistency_score,
                days_since_last_order=pattern.days_since_last_order
            )

        except Exception as e:
            logger.error(f"Error predicting for customer {pattern.customer_id}: {e}")
            return None

    def _is_predictable(self, pattern: CustomerProductPattern) -> bool:
        if pattern.total_orders < 2:
            return False

        if pattern.reorder_cycle_median is None:
            return False

        if pattern.status == 'churned' and pattern.days_since_last_order > 365:
            return False

        return True

    def _predict_order_date(
        self,
        pattern: CustomerProductPattern,
        as_of_date: datetime
    ) -> Tuple[datetime, datetime, datetime, float]:

        base_expected_days = pattern.reorder_cycle_median

        if pattern.trend_direction == 'growing' and pattern.trend_pvalue < 0.05:

            trend_adjustment = -0.1 * base_expected_days
        elif pattern.trend_direction == 'declining' and pattern.trend_pvalue < 0.05:

            trend_adjustment = 0.15 * base_expected_days
        else:
            trend_adjustment = 0

        expected_cycle = base_expected_days + trend_adjustment

        if pattern.rfm_frequency_score > 0.7:

            expected_cycle *= 0.95
        elif pattern.rfm_frequency_score < 0.3:

            expected_cycle *= 1.05

        if pattern.velocity_trend == 'accelerating':

            velocity_adjustment = abs(pattern.order_velocity) * 0.5
            expected_cycle *= (1 - velocity_adjustment)
            expected_cycle = max(expected_cycle, pattern.reorder_cycle_median * 0.7)
        elif pattern.velocity_trend == 'decelerating':

            velocity_adjustment = abs(pattern.order_velocity) * 0.5
            expected_cycle *= (1 + velocity_adjustment)
            expected_cycle = min(expected_cycle, pattern.reorder_cycle_median * 1.5)

        if pattern.status == 'at_risk':

            churn_adjustment = 0.2 * expected_cycle * pattern.churn_probability
            expected_cycle += churn_adjustment
        elif pattern.status == 'churned':

            expected_cycle *= 1.5

        expected_date = pattern.last_order_date + timedelta(days=expected_cycle)

        if pattern.reorder_cycle_iqr and pattern.reorder_cycle_iqr > 0:
            stddev_days = pattern.reorder_cycle_iqr / 1.35
        else:

            stddev_days = expected_cycle * pattern.reorder_cycle_cv

        uncertainty_multiplier = 1 + (1 - pattern.consistency_score)
        stddev_days *= uncertainty_multiplier

        rfm_consistency_factor = 1 - (0.3 * pattern.rfm_consistency_score)
        stddev_days *= rfm_consistency_factor

        if pattern.seasonality_detected and pattern.seasonality_period_days:

            days_since_last = (as_of_date - pattern.last_order_date).days
            seasonal_period = pattern.seasonality_period_days
            cycles_since_last = days_since_last / seasonal_period
            next_seasonal_peak = (np.ceil(cycles_since_last) * seasonal_period)

            days_to_expected = (expected_date - as_of_date).days
            days_to_peak = next_seasonal_peak

            if abs(days_to_expected - days_to_peak) < seasonal_period * 0.3:

                seasonal_weight = pattern.seasonality_strength * 0.5
                expected_cycle = (
                    (1 - seasonal_weight) * (expected_date - pattern.last_order_date).days +
                    seasonal_weight * next_seasonal_peak
                )
                expected_date = pattern.last_order_date + timedelta(days=expected_cycle)

        ci_multiplier = 1.96
        date_lower = expected_date - timedelta(days=ci_multiplier * stddev_days)
        date_upper = expected_date + timedelta(days=ci_multiplier * stddev_days)

        if date_lower < as_of_date:
            date_lower = as_of_date

        return (expected_date, date_lower, date_upper, stddev_days)

    def _predict_order_quantity(
        self,
        pattern: CustomerProductPattern
    ) -> Tuple[float, float, float, float]:
        expected_qty = pattern.avg_quantity
        stddev = pattern.quantity_stddev

        if pattern.quantity_trend == 'increasing':

            expected_qty *= 1.1
        elif pattern.quantity_trend == 'decreasing':

            expected_qty *= 0.9

        if pattern.rfm_monetary_score > 0.8:

            expected_qty *= 1.05
        elif pattern.rfm_monetary_score < 0.3:

            expected_qty *= 0.95

        if pattern.status == 'at_risk':
            expected_qty *= (1 - 0.2 * pattern.churn_probability)
        elif pattern.status == 'churned':
            expected_qty *= 0.5

        rfm_quantity_confidence = (
            0.6 * pattern.rfm_monetary_score +
            0.4 * pattern.rfm_consistency_score
        )
        quantity_uncertainty_factor = 1 - (0.3 * rfm_quantity_confidence)
        stddev *= quantity_uncertainty_factor

        ci_multiplier = 1.96
        qty_lower = max(1, expected_qty - ci_multiplier * stddev)
        qty_upper = expected_qty + ci_multiplier * stddev

        return (
            round(expected_qty, 2),
            round(qty_lower, 2),
            round(qty_upper, 2),
            round(stddev, 2)
        )

    def _calculate_order_probability(
        self,
        pattern: CustomerProductPattern,
        as_of_date: datetime,
        expected_date: datetime,
        date_stddev: float
    ) -> float:

        forecast_end = as_of_date + timedelta(days=self.forecast_horizon_days)

        days_to_expected = (expected_date - pattern.last_order_date).days

        days_to_forecast_end = (forecast_end - pattern.last_order_date).days

        if date_stddev > 0:
            z_score = (days_to_forecast_end - days_to_expected) / date_stddev
            prob = stats.norm.cdf(z_score)
        else:

            prob = 1.0 if days_to_forecast_end >= days_to_expected else 0.0

        prob_active = 1 - pattern.churn_probability
        prob *= prob_active

        prob *= pattern.pattern_confidence

        return round(min(1.0, max(0.0, prob)), 3)

    def _generate_weekly_probabilities(
        self,
        as_of_date: datetime,
        expected_date: datetime,
        date_stddev: float
    ) -> List[Tuple[datetime, float]]:
        weekly_probs = []

        num_weeks = self.forecast_horizon_days // 7

        for week_idx in range(num_weeks):
            week_start = as_of_date + timedelta(days=week_idx * 7)
            week_end = week_start + timedelta(days=7)

            days_to_week_start = (week_start - expected_date).days
            days_to_week_end = (week_end - expected_date).days

            if date_stddev > 0:
                prob_by_week_end = stats.norm.cdf(days_to_week_end / date_stddev)
                prob_by_week_start = stats.norm.cdf(days_to_week_start / date_stddev)
                week_prob = prob_by_week_end - prob_by_week_start
            else:

                if week_start <= expected_date < week_end:
                    week_prob = 1.0
                else:
                    week_prob = 0.0

            weekly_probs.append((week_start, round(max(0.0, week_prob), 4)))

        total_prob = sum(p for _, p in weekly_probs)
        if total_prob > 0:
            weekly_probs = [
                (date, round(prob / total_prob, 4))
                for date, prob in weekly_probs
            ]

        return weekly_probs

    def _calculate_prediction_confidence(
        self,
        pattern: CustomerProductPattern,
        prob_orders: float,
        date_stddev: float
    ) -> float:

        pattern_conf = pattern.pattern_confidence

        prob_conf = prob_orders

        if pattern.reorder_cycle_median > 0:
            relative_stddev = date_stddev / pattern.reorder_cycle_median
            precision_conf = 1 / (1 + relative_stddev)
        else:
            precision_conf = 0.5

        if pattern.status == 'churned':
            status_conf = 0.3
        elif pattern.status == 'at_risk':
            status_conf = 0.7
        else:
            status_conf = 1.0

        confidence = (
            0.3 * pattern_conf +
            0.3 * prob_conf +
            0.2 * precision_conf +
            0.2 * status_conf
        )

        return round(confidence, 3)
