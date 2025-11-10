#!/usr/bin/env python3
"""
Customer Predictor - Bayesian Next-Order Prediction

Predicts when and how much a customer will order next using:
- Bayesian inference with Gaussian distributions
- Confidence intervals (Monte Carlo simulation)
- Seasonal adjustments
- Churn-aware predictions
- Multi-scenario modeling

Uses pattern analysis results to make probabilistic forecasts.
"""

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
    """
    Complete prediction for customer's next order

    Contains:
    - Expected order date with confidence intervals
    - Expected quantity with confidence intervals
    - Probability of ordering in forecast window
    - Weekly probabilities for bucketing
    - Risk and confidence metrics
    """
    customer_id: int
    product_id: int

    # Date prediction
    expected_order_date: datetime
    date_confidence_lower: datetime  # 95% CI lower bound
    date_confidence_upper: datetime  # 95% CI upper bound
    date_stddev_days: float

    # Quantity prediction
    expected_quantity: float
    quantity_confidence_lower: float  # 95% CI
    quantity_confidence_upper: float  # 95% CI
    quantity_stddev: float

    # Probabilities
    probability_orders_this_period: float  # 0-1 (3 month window)
    weekly_probabilities: List[Tuple[datetime, float]]  # List of (week_start, probability)

    # Risk metrics
    status: str  # 'active', 'at_risk', 'churned', 'new'
    churn_probability: float
    prediction_confidence: float  # 0-1 (overall confidence in prediction)

    # Pattern reference
    reorder_cycle_days: float
    consistency_score: float
    days_since_last_order: int


class CustomerPredictor:
    """
    Production-grade Bayesian predictor for customer orders

    Approach:
    - Uses pattern statistics as prior distribution
    - Applies Bayesian updates based on recency
    - Generates confidence intervals via analytical methods
    - Handles edge cases (churn, seasonality, new customers)
    - Produces weekly probability distributions

    Optimized for B2B scenarios with established patterns.
    """

    def __init__(self, forecast_horizon_days: int = 90):
        """
        Initialize predictor

        Args:
            forecast_horizon_days: How far ahead to predict (default 90 = 3 months)
        """
        self.forecast_horizon_days = forecast_horizon_days
        logger.info(f"CustomerPredictor initialized with {forecast_horizon_days} day horizon")

    def predict_next_order(
        self,
        pattern: CustomerProductPattern,
        as_of_date: str
    ) -> Optional[CustomerPrediction]:
        """
        Predict customer's next order with confidence intervals

        Steps:
        1. Validate pattern has sufficient data
        2. Calculate expected order date (Bayesian)
        3. Calculate expected quantity (Bayesian)
        4. Generate confidence intervals (analytical)
        5. Calculate weekly probabilities
        6. Assess prediction confidence
        7. Return complete prediction

        Args:
            pattern: Customer-product pattern from PatternAnalyzer
            as_of_date: Prediction reference date (ISO format)

        Returns:
            CustomerPrediction or None if insufficient data
        """
        try:
            as_of_dt = datetime.fromisoformat(as_of_date)

            # 1. Validate pattern
            if not self._is_predictable(pattern):
                logger.debug(f"Customer {pattern.customer_id} not predictable")
                return None

            # 2. Predict next order date
            expected_date, date_lower, date_upper, date_stddev = self._predict_order_date(
                pattern, as_of_dt
            )

            # 3. Predict order quantity
            expected_qty, qty_lower, qty_upper, qty_stddev = self._predict_order_quantity(
                pattern
            )

            # 4. Calculate probability of ordering in forecast window
            prob_orders = self._calculate_order_probability(
                pattern, as_of_dt, expected_date, date_stddev
            )

            # 5. Generate weekly probability distribution
            weekly_probs = self._generate_weekly_probabilities(
                as_of_dt, expected_date, date_stddev
            )

            # 6. Overall prediction confidence
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
        """
        Check if pattern has sufficient data for prediction

        Requirements:
        - At least 2 orders (need 1 reorder cycle)
        - Not churned completely (unless recent)
        - Has reorder cycle data
        """
        if pattern.total_orders < 2:
            return False

        if pattern.reorder_cycle_median is None:
            return False

        # Allow churned if recently active
        if pattern.status == 'churned' and pattern.days_since_last_order > 365:
            return False

        return True

    def _predict_order_date(
        self,
        pattern: CustomerProductPattern,
        as_of_date: datetime
    ) -> Tuple[datetime, datetime, datetime, float]:
        """
        Predict next order date using Bayesian approach

        Prior: reorder_cycle_median ± IQR
        Likelihood: weighted by recency and consistency
        Posterior: Bayesian combination

        Returns: (expected_date, lower_95ci, upper_95ci, stddev_days)
        """
        # Base prediction: last order + median cycle
        base_expected_days = pattern.reorder_cycle_median

        # Adjust for trend
        if pattern.trend_direction == 'growing' and pattern.trend_pvalue < 0.05:
            # Ordering more frequently → shorter cycle
            trend_adjustment = -0.1 * base_expected_days
        elif pattern.trend_direction == 'declining' and pattern.trend_pvalue < 0.05:
            # Ordering less frequently → longer cycle
            trend_adjustment = 0.15 * base_expected_days
        else:
            trend_adjustment = 0

        expected_cycle = base_expected_days + trend_adjustment

        # Adjust for churn risk
        if pattern.status == 'at_risk':
            # Likely to order later than expected
            churn_adjustment = 0.2 * expected_cycle * pattern.churn_probability
            expected_cycle += churn_adjustment
        elif pattern.status == 'churned':
            # May take much longer or not order at all
            expected_cycle *= 1.5

        # Calculate expected date
        expected_date = pattern.last_order_date + timedelta(days=expected_cycle)

        # Standard deviation (use IQR / 1.35 as robust stddev estimator)
        if pattern.reorder_cycle_iqr and pattern.reorder_cycle_iqr > 0:
            stddev_days = pattern.reorder_cycle_iqr / 1.35
        else:
            # Fallback: use CV
            stddev_days = expected_cycle * pattern.reorder_cycle_cv

        # Inflate uncertainty for low consistency
        uncertainty_multiplier = 1 + (1 - pattern.consistency_score)
        stddev_days *= uncertainty_multiplier

        # Seasonality adjustment
        if pattern.seasonality_detected and pattern.seasonality_period_days:
            # Find nearest seasonal peak
            days_since_last = (as_of_date - pattern.last_order_date).days
            seasonal_period = pattern.seasonality_period_days
            cycles_since_last = days_since_last / seasonal_period
            next_seasonal_peak = (np.ceil(cycles_since_last) * seasonal_period)

            # Nudge expected date toward seasonal peak if close
            days_to_expected = (expected_date - as_of_date).days
            days_to_peak = next_seasonal_peak

            if abs(days_to_expected - days_to_peak) < seasonal_period * 0.3:
                # Within 30% of seasonal cycle → adjust toward peak
                seasonal_weight = pattern.seasonality_strength * 0.5
                expected_cycle = (
                    (1 - seasonal_weight) * (expected_date - pattern.last_order_date).days +
                    seasonal_weight * next_seasonal_peak
                )
                expected_date = pattern.last_order_date + timedelta(days=expected_cycle)

        # 95% confidence interval (±1.96 * stddev for normal distribution)
        ci_multiplier = 1.96
        date_lower = expected_date - timedelta(days=ci_multiplier * stddev_days)
        date_upper = expected_date + timedelta(days=ci_multiplier * stddev_days)

        # Ensure dates are not in the past
        if date_lower < as_of_date:
            date_lower = as_of_date

        return (expected_date, date_lower, date_upper, stddev_days)

    def _predict_order_quantity(
        self,
        pattern: CustomerProductPattern
    ) -> Tuple[float, float, float, float]:
        """
        Predict order quantity using Bayesian approach

        Prior: avg_quantity ± stddev
        Adjust for trend

        Returns: (expected_qty, lower_95ci, upper_95ci, stddev)
        """
        expected_qty = pattern.avg_quantity
        stddev = pattern.quantity_stddev

        # Adjust for quantity trend
        if pattern.quantity_trend == 'increasing':
            # Expect slightly higher quantity
            expected_qty *= 1.1
        elif pattern.quantity_trend == 'decreasing':
            # Expect slightly lower quantity
            expected_qty *= 0.9

        # Adjust for churn risk (at-risk customers may order less)
        if pattern.status == 'at_risk':
            expected_qty *= (1 - 0.2 * pattern.churn_probability)
        elif pattern.status == 'churned':
            expected_qty *= 0.5  # If they order, likely smaller amount

        # Confidence intervals (95%)
        ci_multiplier = 1.96
        qty_lower = max(1, expected_qty - ci_multiplier * stddev)  # At least 1
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
        """
        Calculate probability customer orders within forecast window

        Uses cumulative distribution function (CDF) of normal distribution
        """
        # Days until end of forecast window
        forecast_end = as_of_date + timedelta(days=self.forecast_horizon_days)

        # Days from last order to expected order
        days_to_expected = (expected_date - pattern.last_order_date).days

        # Days from last order to forecast end
        days_to_forecast_end = (forecast_end - pattern.last_order_date).days

        # Probability of ordering by forecast end (CDF of normal)
        if date_stddev > 0:
            z_score = (days_to_forecast_end - days_to_expected) / date_stddev
            prob = stats.norm.cdf(z_score)
        else:
            # No variance → deterministic
            prob = 1.0 if days_to_forecast_end >= days_to_expected else 0.0

        # Adjust for churn probability
        prob_active = 1 - pattern.churn_probability
        prob *= prob_active

        # Adjust for pattern confidence
        prob *= pattern.pattern_confidence

        return round(min(1.0, max(0.0, prob)), 3)

    def _generate_weekly_probabilities(
        self,
        as_of_date: datetime,
        expected_date: datetime,
        date_stddev: float
    ) -> List[Tuple[datetime, float]]:
        """
        Generate probability distribution across weekly buckets

        Uses probability density function (PDF) of normal distribution
        Returns list of (week_start_date, probability)
        """
        weekly_probs = []

        # Generate weeks for forecast horizon
        num_weeks = self.forecast_horizon_days // 7

        for week_idx in range(num_weeks):
            week_start = as_of_date + timedelta(days=week_idx * 7)
            week_end = week_start + timedelta(days=7)

            # Calculate probability of ordering in this week
            # Using CDF: P(week_start <= order_date < week_end)

            days_to_week_start = (week_start - expected_date).days
            days_to_week_end = (week_end - expected_date).days

            if date_stddev > 0:
                prob_by_week_end = stats.norm.cdf(days_to_week_end / date_stddev)
                prob_by_week_start = stats.norm.cdf(days_to_week_start / date_stddev)
                week_prob = prob_by_week_end - prob_by_week_start
            else:
                # Deterministic: probability = 1 if expected date falls in week
                if week_start <= expected_date < week_end:
                    week_prob = 1.0
                else:
                    week_prob = 0.0

            weekly_probs.append((week_start, round(max(0.0, week_prob), 4)))

        # Normalize to sum to 1.0 (ensure valid probability distribution)
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
        """
        Overall confidence in this prediction

        Combines:
        - Pattern confidence (sample size + consistency)
        - Order probability (likelihood of ordering)
        - Prediction precision (low stddev = high confidence)
        """
        # Component 1: Pattern confidence (from analyzer)
        pattern_conf = pattern.pattern_confidence

        # Component 2: Order probability (high prob = high confidence)
        prob_conf = prob_orders

        # Component 3: Precision (low relative stddev = high confidence)
        if pattern.reorder_cycle_median > 0:
            relative_stddev = date_stddev / pattern.reorder_cycle_median
            precision_conf = 1 / (1 + relative_stddev)
        else:
            precision_conf = 0.5

        # Component 4: Status penalty
        if pattern.status == 'churned':
            status_conf = 0.3
        elif pattern.status == 'at_risk':
            status_conf = 0.7
        else:
            status_conf = 1.0

        # Weighted combination
        confidence = (
            0.3 * pattern_conf +
            0.3 * prob_conf +
            0.2 * precision_conf +
            0.2 * status_conf
        )

        return round(confidence, 3)
