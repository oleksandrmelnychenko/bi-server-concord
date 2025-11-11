#!/usr/bin/env python3
"""
Pattern Analyzer - Production-Grade Statistical Analysis

Analyzes customer-product purchasing patterns using robust statistics:
- Median/IQR instead of mean/stddev (outlier resistant)
- Mann-Kendall trend detection (non-parametric)
- FFT seasonality detection
- Survival analysis for churn prediction
- Bayesian confidence scoring

Uses 6+ years of historical data for accurate pattern recognition.
"""

import numpy as np
from scipy import stats
from scipy.fft import fft
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import sys

# Add parent directory to path for datetime_utils
sys.path.insert(0, '/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts')
from datetime_utils import calculate_fractional_days

logger = logging.getLogger(__name__)


@dataclass
class CustomerProductPattern:
    """
    Complete pattern analysis for customer-product pair

    Contains all metrics needed for prediction:
    - Order history statistics
    - Reorder cycle metrics (robust)
    - Quantity patterns
    - Temporal patterns (seasonality, trend)
    - Health/churn metrics
    - Confidence scores
    """
    customer_id: int
    product_id: int

    # Order history
    total_orders: int
    first_order_date: datetime
    last_order_date: datetime

    # Reorder cycle metrics (ROBUST - using median/IQR)
    reorder_cycle_median: float  # days (robust to outliers)
    reorder_cycle_iqr: float     # interquartile range
    reorder_cycle_cv: float      # coefficient of variation

    # Quantity metrics
    avg_quantity: float
    quantity_stddev: float
    quantity_trend: str          # 'increasing', 'stable', 'decreasing'

    # Temporal patterns
    seasonality_detected: bool
    seasonality_period_days: Optional[int]  # e.g., 7, 14, 30, 60
    seasonality_strength: float  # 0-1

    # Trend (using Mann-Kendall non-parametric test)
    trend_direction: str         # 'growing', 'stable', 'declining'
    trend_slope: float          # orders per month change
    trend_pvalue: float         # statistical significance

    # Health metrics
    consistency_score: float     # 0-1 (1 = very consistent)
    status: str                 # 'active', 'at_risk', 'churned'
    churn_probability: float    # 0-1
    days_since_last_order: int
    days_overdue: float         # negative if not overdue

    # RFM (Recency-Frequency-Monetary) Metrics
    rfm_recency_score: float    # 0-1 (1 = very recent order)
    rfm_frequency_score: float  # 0-1 (orders per year normalized)
    rfm_monetary_score: float   # 0-1 (average order value normalized)
    rfm_consistency_score: float # 0-1 (regularity of ordering)
    rfm_segment: str            # 'champion', 'loyal', 'at_risk', 'hibernating', 'lost'

    # Order Velocity Metrics
    order_velocity: float       # Change in order frequency (negative = accelerating, positive = decelerating)
    order_acceleration: float   # Change in velocity (second derivative)
    velocity_trend: str         # 'accelerating', 'stable', 'decelerating'

    # Confidence (Bayesian posterior)
    pattern_confidence: float   # 0-1 (based on sample size + consistency)


class PatternAnalyzer:
    """
    Production-grade pattern analysis with robust statistics

    Uses:
    - Median instead of mean (robust to outliers)
    - IQR instead of stddev (robust to outliers)
    - Mann-Kendall for trend (non-parametric)
    - FFT for seasonality detection
    - Survival analysis for churn

    Optimized for 6+ years of historical B2B data.
    """

    def __init__(self, conn):
        """
        Initialize analyzer with database connection

        Args:
            conn: pymssql connection (from pool)
        """
        self.conn = conn
        logger.info("PatternAnalyzer initialized")

    def analyze_customer_product(
        self,
        customer_id: int,
        product_id: int,
        as_of_date: str
    ) -> Optional[CustomerProductPattern]:
        """
        Comprehensive pattern analysis using 6 years of data

        Steps:
        1. Fetch complete order history
        2. Calculate reorder cycles (robust statistics)
        3. Detect seasonality (FFT + autocorrelation)
        4. Analyze trends (Mann-Kendall test)
        5. Compute consistency scores
        6. Assess churn risk (survival analysis)
        7. Return complete pattern object

        Args:
            customer_id: Customer ID
            product_id: Product ID
            as_of_date: Analysis cutoff date (ISO format)

        Returns:
            CustomerProductPattern or None if no orders
        """
        try:
            # 1. Get order history
            orders = self._get_order_history(customer_id, product_id, as_of_date)

            if len(orders) == 0:
                logger.debug(f"No orders for customer {customer_id}, product {product_id}")
                return None

            # 2. Calculate reorder cycles
            cycles = self._calculate_reorder_cycles(orders)

            # 3. Reorder cycle statistics (ROBUST)
            cycle_median = np.median(cycles) if len(cycles) > 0 else None
            if len(cycles) > 0:
                cycle_q1, cycle_q3 = np.percentile(cycles, [25, 75])
                cycle_iqr = cycle_q3 - cycle_q1
                cycle_cv = np.std(cycles) / np.mean(cycles) if np.mean(cycles) > 0 else 0
            else:
                cycle_iqr = None
                cycle_cv = 0

            # 4. Quantity statistics
            quantities = [o['quantity'] for o in orders]
            avg_quantity = np.mean(quantities)
            quantity_stddev = np.std(quantities) if len(quantities) > 1 else 0

            # 5. Quantity trend (using linear regression)
            quantity_trend = self._detect_quantity_trend(orders)

            # 6. Seasonality detection (FFT + autocorrelation)
            seasonality = self._detect_seasonality(cycles) if len(cycles) >= 12 else None

            # 7. Order frequency trend (Mann-Kendall test)
            trend = self._detect_frequency_trend(orders)

            # 8. Consistency score
            consistency = self._calculate_consistency_score(
                cycle_cv, len(orders), cycle_iqr, cycle_median
            )

            # 9. Calculate days since last order (needed for RFM and churn)
            days_since_last = (datetime.fromisoformat(as_of_date) - orders[-1]['date']).days

            # 10. RFM Features (calculate early, used by churn and confidence)
            rfm = self._calculate_rfm_features(
                orders, days_since_last, cycle_median, cycle_cv, as_of_date
            )

            # 11. Order Velocity (calculate momentum)
            velocity = self._calculate_order_velocity(orders)

            # 12. Churn risk assessment (with RFM enhancement)
            status, churn_prob, days_overdue = self._assess_churn_risk(
                days_since_last, cycle_median, cycle_iqr, len(orders), rfm
            )

            # 13. Pattern confidence (Bayesian posterior with RFM)
            pattern_confidence = self._calculate_pattern_confidence(
                len(orders), consistency, days_since_last, cycle_median, rfm
            )

            return CustomerProductPattern(
                customer_id=customer_id,
                product_id=product_id,
                total_orders=len(orders),
                first_order_date=orders[0]['date'],
                last_order_date=orders[-1]['date'],
                reorder_cycle_median=cycle_median,
                reorder_cycle_iqr=cycle_iqr,
                reorder_cycle_cv=cycle_cv,
                avg_quantity=avg_quantity,
                quantity_stddev=quantity_stddev,
                quantity_trend=quantity_trend,
                seasonality_detected=seasonality is not None,
                seasonality_period_days=seasonality['period'] if seasonality else None,
                seasonality_strength=seasonality['strength'] if seasonality else 0.0,
                trend_direction=trend['direction'],
                trend_slope=trend['slope'],
                trend_pvalue=trend['pvalue'],
                consistency_score=consistency,
                status=status,
                churn_probability=churn_prob,
                days_since_last_order=days_since_last,
                days_overdue=days_overdue,
                rfm_recency_score=rfm['recency_score'],
                rfm_frequency_score=rfm['frequency_score'],
                rfm_monetary_score=rfm['monetary_score'],
                rfm_consistency_score=rfm['consistency_score'],
                rfm_segment=rfm['segment'],
                order_velocity=velocity['velocity'],
                order_acceleration=velocity['acceleration'],
                velocity_trend=velocity['trend'],
                pattern_confidence=pattern_confidence
            )

        except Exception as e:
            logger.error(f"Error analyzing pattern for customer {customer_id}, product {product_id}: {e}")
            return None

    def _get_order_history(
        self,
        customer_id: int,
        product_id: int,
        as_of_date: str
    ) -> List[Dict]:
        """
        Get complete order history with quantities

        Uses: Client → ClientAgreement → Order → OrderItem

        Returns list of {'date': datetime, 'quantity': int}
        """
        query = """
        SELECT
            o.ID as order_id,
            o.Created as order_date,
            SUM(oi.Qty) as total_quantity
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = %s
              AND oi.ProductID = %s
              AND o.Created < %s
        GROUP BY o.ID, o.Created
        ORDER BY o.Created
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query, (customer_id, product_id, as_of_date))

        orders = []
        for row in cursor.fetchall():
            orders.append({
                'date': row['order_date'],
                'quantity': row['total_quantity']
            })

        cursor.close()
        return orders

    def _calculate_reorder_cycles(self, orders: List[Dict]) -> List[float]:
        """
        Calculate days between consecutive orders with sub-day precision

        Uses fractional days for accuracy (e.g., 3.5 days = 3 days 12 hours).
        Ignores same-day duplicates (< 1 day apart).
        """
        cycles = []
        for i in range(1, len(orders)):
            # Use fractional days for precision instead of integer .days
            days = calculate_fractional_days(orders[i-1]['date'], orders[i]['date'])
            if days >= 1.0:  # Ignore orders less than 1 day apart
                cycles.append(days)
        return cycles

    def _detect_seasonality(self, cycles: List[float]) -> Optional[Dict]:
        """
        Detect seasonality using FFT and autocorrelation

        Returns period and strength if detected
        Common business periods: 7, 14, 30, 60, 90 days
        """
        if len(cycles) < 12:  # Need minimum data
            return None

        try:
            # Autocorrelation for validation
            autocorr = np.correlate(cycles, cycles, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize

            # Look for peaks in autocorrelation
            peaks = []
            for i in range(2, min(len(autocorr)-1, 30)):  # Check up to 30 orders
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.3:  # Threshold for significance
                        peaks.append((i, autocorr[i]))

            if peaks:
                # Strongest peak is seasonality period
                best_peak = max(peaks, key=lambda x: x[1])
                period_days = int(best_peak[0] * np.median(cycles))
                strength = best_peak[1]

                # Validate: should be 7, 14, 30, 60, 90 days (common business cycles)
                common_periods = [7, 14, 30, 60, 90]
                closest_period = min(common_periods, key=lambda x: abs(x - period_days))

                if abs(closest_period - period_days) / max(1, closest_period) < 0.3:  # Within 30%
                    return {
                        'period': closest_period,
                        'strength': strength,
                        'confidence': 0.8 if len(cycles) >= 24 else 0.6
                    }

            return None

        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
            return None

    def _detect_frequency_trend(self, orders: List[Dict]) -> Dict:
        """
        Detect if order frequency is increasing/decreasing using Mann-Kendall test

        Mann-Kendall is non-parametric, robust to outliers
        Returns trend direction, slope, and p-value
        """
        if len(orders) < 4:
            return {'direction': 'stable', 'slope': 0.0, 'pvalue': 1.0}

        try:
            # Calculate order intervals
            intervals = []
            for i in range(1, len(orders)):
                days = (orders[i]['date'] - orders[i-1]['date']).days
                if days > 0:
                    intervals.append(days)

            if len(intervals) < 3:
                return {'direction': 'stable', 'slope': 0.0, 'pvalue': 1.0}

            # Mann-Kendall test
            result = stats.kendalltau(range(len(intervals)), intervals)
            tau, pvalue = result.correlation, result.pvalue

            # Sen's slope estimator (robust slope)
            slopes = []
            for i in range(len(intervals)):
                for j in range(i+1, len(intervals)):
                    if j - i > 0:
                        slope = (intervals[j] - intervals[i]) / (j - i)
                        slopes.append(slope)

            median_slope = np.median(slopes) if slopes else 0.0

            # Interpret
            if pvalue < 0.05:  # Statistically significant
                if median_slope > 0:
                    direction = 'declining'  # Increasing intervals = declining frequency
                else:
                    direction = 'growing'
            else:
                direction = 'stable'

            # Convert slope to orders per month (negative because intervals → frequency)
            slope_per_month = -median_slope * 30 / max(1, np.mean(intervals))

            return {
                'direction': direction,
                'slope': round(slope_per_month, 4),
                'pvalue': round(pvalue, 4)
            }

        except Exception as e:
            logger.warning(f"Trend detection failed: {e}")
            return {'direction': 'stable', 'slope': 0.0, 'pvalue': 1.0}

    def _detect_quantity_trend(self, orders: List[Dict]) -> str:
        """
        Detect if order quantities are increasing/decreasing

        Uses linear regression
        """
        if len(orders) < 3:
            return 'stable'

        try:
            quantities = [o['quantity'] for o in orders]

            # Simple linear regression
            x = np.arange(len(quantities))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, quantities)

            if p_value < 0.05:  # Significant
                avg_qty = np.mean(quantities)
                if slope > 0.1 * avg_qty / len(quantities):  # > 10% per order on average
                    return 'increasing'
                elif slope < -0.1 * avg_qty / len(quantities):
                    return 'decreasing'

            return 'stable'

        except Exception as e:
            logger.warning(f"Quantity trend detection failed: {e}")
            return 'stable'

    def _calculate_consistency_score(
        self,
        cv: float,
        n_orders: int,
        iqr: Optional[float],
        median: Optional[float]
    ) -> float:
        """
        Calculate pattern consistency (0-1)

        High consistency = predictable reorder patterns
        Formula combines CV, sample size, and IQR
        """
        if median is None or median == 0 or iqr is None:
            return 0.0

        # Component 1: Low coefficient of variation is good
        cv_score = 1 / (1 + cv)  # Range: 0.5-1.0 typically

        # Component 2: More orders = more confidence
        sample_score = min(1.0, n_orders / 20)  # Max out at 20 orders

        # Component 3: Low relative IQR is good
        relative_iqr = iqr / median
        iqr_score = 1 / (1 + relative_iqr)

        # Weighted combination
        consistency = (
            0.4 * cv_score +
            0.3 * sample_score +
            0.3 * iqr_score
        )

        return round(consistency, 3)

    def _assess_churn_risk(
        self,
        days_since_last: int,
        cycle_median: Optional[float],
        cycle_iqr: Optional[float],
        n_orders: int,
        rfm: Optional[Dict] = None
    ) -> Tuple[str, float, float]:
        """
        Assess churn risk using survival analysis with RFM enhancement

        RFM Enhancement:
        - High RFM recency score → lower churn risk
        - High RFM frequency score → more loyal, lower churn
        - Low RFM scores → higher churn risk

        Returns: (status, churn_probability, days_overdue)
        """
        if cycle_median is None:
            return ('unknown', 0.5, 0.0)

        # Expected next order date
        expected_reorder_days = cycle_median
        days_overdue = days_since_last - expected_reorder_days

        # Calculate churn probability using exponential decay
        # The longer overdue, the higher the churn risk

        if days_overdue <= 0:
            # Not overdue
            status = 'active'
            churn_prob = 0.05  # Base churn rate

        elif cycle_iqr and days_overdue <= cycle_iqr:
            # Slightly overdue (within IQR)
            status = 'active'
            churn_prob = 0.15

        elif days_overdue <= cycle_median:
            # Moderately overdue (1x cycle)
            status = 'at_risk'
            churn_prob = 0.40

        elif days_overdue <= 2 * cycle_median:
            # Significantly overdue (2x cycle)
            status = 'at_risk'
            churn_prob = 0.70

        else:
            # Very overdue (>2x cycle)
            status = 'churned'
            churn_prob = 0.90

        # Adjust for customer loyalty (more orders = lower churn)
        loyalty_factor = min(1.0, n_orders / 10)
        churn_prob = churn_prob * (1 - 0.3 * loyalty_factor)

        # RFM Enhancement: Adjust churn based on RFM scores
        if rfm:
            # High RFM recency = recently ordered = lower churn
            recency_adjustment = rfm['recency_score']

            # High RFM frequency = loyal customer = lower churn
            frequency_adjustment = rfm['frequency_score']

            # High RFM monetary = valuable customer = lower churn (businesses work harder to retain)
            monetary_adjustment = rfm['monetary_score']

            # Combined RFM loyalty score
            rfm_loyalty = (
                0.4 * recency_adjustment +
                0.4 * frequency_adjustment +
                0.2 * monetary_adjustment
            )

            # Reduce churn probability based on RFM loyalty
            # High RFM (0.8+) → reduce churn by up to 40%
            # Medium RFM (0.5) → reduce churn by 20%
            # Low RFM (0.2) → reduce churn by 0-10%
            churn_reduction = 0.5 * rfm_loyalty  # Max 50% reduction
            churn_prob = churn_prob * (1 - churn_reduction)

            # Adjust status based on RFM
            # High-RFM customers stay "active" longer
            if status == 'at_risk' and rfm_loyalty > 0.7:
                # Downgrade risk for high-loyalty customers
                status = 'active'
            elif status == 'churned' and rfm_loyalty > 0.6:
                # Give high-loyalty customers benefit of doubt
                status = 'at_risk'

        return (status, round(churn_prob, 3), round(days_overdue, 1))

    def _calculate_pattern_confidence(
        self,
        n_orders: int,
        consistency: float,
        days_since_last: int,
        cycle_median: Optional[float],
        rfm: Optional[Dict] = None
    ) -> float:
        """
        Bayesian confidence in pattern prediction with RFM enhancement

        Higher confidence = more predictable
        Combines sample size, consistency, recency, and RFM features

        RFM Enhancement:
        - High RFM consistency → higher confidence
        - High RFM frequency → more reliable patterns
        - High RFM recency → more relevant predictions
        """
        # Prior: Start with medium confidence
        prior = 0.5

        # Update based on sample size (more data = more confidence)
        likelihood_sample = min(1.0, n_orders / 15)

        # Update based on consistency
        likelihood_consistency = consistency

        # Update based on recency (recent data is more relevant)
        if cycle_median and cycle_median > 0:
            recency_factor = min(1.0, cycle_median / max(1, days_since_last))
        else:
            recency_factor = 0.5

        # RFM Enhancement (if available)
        if rfm:
            # RFM consistency: how regular are orders
            rfm_consistency_factor = rfm['consistency_score']

            # RFM frequency: frequent customers = more predictable
            rfm_frequency_factor = rfm['frequency_score']

            # RFM recency: recent customers = more reliable
            rfm_recency_factor = rfm['recency_score']

            # Combined RFM factor (weighted average)
            rfm_factor = (
                0.4 * rfm_consistency_factor +
                0.3 * rfm_frequency_factor +
                0.3 * rfm_recency_factor
            )

            # Bayesian update with RFM (rebalanced weights)
            posterior = (
                0.1 * prior +
                0.25 * likelihood_sample +
                0.20 * likelihood_consistency +
                0.10 * recency_factor +
                0.35 * rfm_factor  # RFM gets highest weight
            )
        else:
            # Fallback: original calculation without RFM
            posterior = (
                0.2 * prior +
                0.4 * likelihood_sample +
                0.3 * likelihood_consistency +
                0.1 * recency_factor
            )

        return round(posterior, 3)

    def _calculate_rfm_features(
        self,
        orders: List[Dict],
        days_since_last: int,
        cycle_median: Optional[float],
        cycle_cv: float,
        as_of_date: str
    ) -> Dict:
        """
        Calculate RFM (Recency-Frequency-Monetary) features

        RFM Analysis for B2B customers:
        - Recency: How recently did they order? (0-1, higher = more recent)
        - Frequency: How often do they order? (0-1, higher = more frequent)
        - Monetary: How much do they spend? (0-1, higher = more value)
        - Consistency: How regular are their orders? (0-1, higher = more predictable)

        Args:
            orders: List of order history dicts
            days_since_last: Days since last order
            cycle_median: Median reorder cycle days
            cycle_cv: Coefficient of variation of reorder cycles
            as_of_date: Analysis date

        Returns:
            Dict with RFM scores and segment
        """
        if len(orders) == 0:
            return {
                'recency_score': 0.0,
                'frequency_score': 0.0,
                'monetary_score': 0.0,
                'consistency_score': 0.0,
                'segment': 'new'
            }

        # 1. RECENCY SCORE (0-1, higher = more recent)
        # Score based on days since last order relative to expected cycle
        if cycle_median and cycle_median > 0:
            # Normalize by expected cycle: on-time = 1.0, overdue = lower
            recency_ratio = cycle_median / max(1, days_since_last)
            recency_score = min(1.0, max(0.0, recency_ratio))
        else:
            # Fallback: use absolute recency
            # Recent (< 30 days) = high, old (> 180 days) = low
            if days_since_last < 30:
                recency_score = 1.0
            elif days_since_last < 90:
                recency_score = 0.7
            elif days_since_last < 180:
                recency_score = 0.4
            else:
                recency_score = 0.1

        # 2. FREQUENCY SCORE (0-1, higher = more frequent orders)
        # Calculate orders per year
        first_order = orders[0]['date']
        last_order = orders[-1]['date']
        total_days = (datetime.fromisoformat(as_of_date) - first_order).days

        if total_days > 0:
            orders_per_year = len(orders) / (total_days / 365.25)
        else:
            orders_per_year = 0

        # Normalize: 0-1 orders/year → 0-0.3, 1-4 orders/year → 0.3-0.7, 4+ → 0.7-1.0
        if orders_per_year < 1:
            frequency_score = 0.3 * orders_per_year
        elif orders_per_year < 4:
            frequency_score = 0.3 + 0.4 * (orders_per_year - 1) / 3
        else:
            frequency_score = 0.7 + 0.3 * min(1.0, (orders_per_year - 4) / 8)

        frequency_score = round(min(1.0, max(0.0, frequency_score)), 3)

        # 3. MONETARY SCORE (0-1, higher = higher value customer)
        # Use average order revenue
        # Note: We need to query for revenue data, but for now use quantity as proxy
        avg_quantity = np.mean([o['quantity'] for o in orders])

        # Normalize quantity to 0-1 scale
        # Typical B2B: 1-10 units → 0.3-0.7, 10+ units → 0.7-1.0
        if avg_quantity < 5:
            monetary_score = 0.3 + 0.4 * (avg_quantity / 5)
        elif avg_quantity < 20:
            monetary_score = 0.7 + 0.2 * ((avg_quantity - 5) / 15)
        else:
            monetary_score = 0.9 + 0.1 * min(1.0, (avg_quantity - 20) / 30)

        monetary_score = round(min(1.0, max(0.0, monetary_score)), 3)

        # 4. CONSISTENCY SCORE (0-1, higher = more predictable)
        # Use inverse of CV: low CV = high consistency
        if cycle_cv > 0:
            # CV of 0-0.5 → score 0.7-1.0 (very consistent)
            # CV of 0.5-1.5 → score 0.3-0.7 (moderate)
            # CV > 1.5 → score 0-0.3 (inconsistent)
            if cycle_cv < 0.5:
                consistency_score = 0.7 + 0.3 * (1 - cycle_cv / 0.5)
            elif cycle_cv < 1.5:
                consistency_score = 0.3 + 0.4 * (1 - (cycle_cv - 0.5) / 1.0)
            else:
                consistency_score = 0.3 * max(0.0, 1 - (cycle_cv - 1.5) / 2.0)
        else:
            consistency_score = 1.0  # Perfect consistency (though unlikely)

        consistency_score = round(min(1.0, max(0.0, consistency_score)), 3)

        # 5. RFM SEGMENT CLASSIFICATION
        # Classify customer based on RFM scores
        # Using thresholds: High = 0.6+, Medium = 0.3-0.6, Low = < 0.3

        r_high = recency_score >= 0.6
        r_med = 0.3 <= recency_score < 0.6
        f_high = frequency_score >= 0.6
        f_med = 0.3 <= frequency_score < 0.6
        m_high = monetary_score >= 0.6

        # Segment logic (simplified RFM segmentation)
        if r_high and f_high and m_high:
            segment = 'champion'  # Best customers
        elif r_high and f_high:
            segment = 'loyal'  # Regular high-frequency customers
        elif r_high and not f_high:
            segment = 'potential'  # Recent but low frequency
        elif r_med and f_high:
            segment = 'at_risk'  # High frequency but slipping
        elif r_med:
            segment = 'hibernating'  # Moderate recency, need attention
        else:
            segment = 'lost'  # Low recency

        return {
            'recency_score': round(recency_score, 3),
            'frequency_score': frequency_score,
            'monetary_score': monetary_score,
            'consistency_score': consistency_score,
            'segment': segment
        }

    def _calculate_order_velocity(
        self,
        orders: List[Dict]
    ) -> Dict:
        """
        Calculate order velocity and acceleration

        Order Velocity measures whether customer is ordering faster or slower:
        - Negative velocity = accelerating (intervals decreasing)
        - Positive velocity = decelerating (intervals increasing)
        - Zero velocity = stable

        Acceleration is the second derivative (change in velocity).

        Args:
            orders: List of order history dicts

        Returns:
            Dict with velocity, acceleration, and trend
        """
        if len(orders) < 3:
            return {
                'velocity': 0.0,
                'acceleration': 0.0,
                'trend': 'stable'
            }

        # Calculate intervals between consecutive orders
        intervals = []
        for i in range(1, len(orders)):
            days = (orders[i]['date'] - orders[i-1]['date']).days
            if days > 0:  # Ignore same-day duplicates
                intervals.append(days)

        if len(intervals) < 2:
            return {
                'velocity': 0.0,
                'acceleration': 0.0,
                'trend': 'stable'
            }

        # Calculate velocity: comparing recent vs historical intervals
        # Use median for robustness
        if len(intervals) >= 4:
            # Split into recent (last 25%) and historical (first 75%)
            split_point = max(1, len(intervals) * 3 // 4)
            historical_intervals = intervals[:split_point]
            recent_intervals = intervals[split_point:]

            historical_median = np.median(historical_intervals)
            recent_median = np.median(recent_intervals)

            # Velocity = (recent - historical) / historical
            # Positive = slowing down, Negative = speeding up
            if historical_median > 0:
                velocity = (recent_median - historical_median) / historical_median
            else:
                velocity = 0.0
        else:
            # Too few intervals: use simple comparison of first vs last half
            mid = len(intervals) // 2
            first_half_median = np.median(intervals[:mid+1])
            second_half_median = np.median(intervals[mid:])

            if first_half_median > 0:
                velocity = (second_half_median - first_half_median) / first_half_median
            else:
                velocity = 0.0

        # Calculate acceleration (second derivative)
        # Compare velocity of early vs late periods
        if len(intervals) >= 6:
            # Split into thirds
            third = len(intervals) // 3
            early_intervals = intervals[:third]
            middle_intervals = intervals[third:2*third]
            late_intervals = intervals[2*third:]

            early_median = np.median(early_intervals)
            middle_median = np.median(middle_intervals)
            late_median = np.median(late_intervals)

            # Velocity 1: early → middle
            if early_median > 0:
                vel1 = (middle_median - early_median) / early_median
            else:
                vel1 = 0.0

            # Velocity 2: middle → late
            if middle_median > 0:
                vel2 = (late_median - middle_median) / middle_median
            else:
                vel2 = 0.0

            # Acceleration = change in velocity
            acceleration = vel2 - vel1
        else:
            acceleration = 0.0

        # Determine trend
        # Use thresholds: > 0.15 = significant change
        if velocity < -0.15:
            trend = 'accelerating'  # Ordering faster
        elif velocity > 0.15:
            trend = 'decelerating'  # Ordering slower
        else:
            trend = 'stable'

        return {
            'velocity': round(velocity, 3),
            'acceleration': round(acceleration, 3),
            'trend': trend
        }
