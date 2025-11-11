#!/usr/bin/env python3

import numpy as np
from scipy import stats
from scipy.fft import fft
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import sys

sys.path.insert(0, '/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts')
from datetime_utils import calculate_fractional_days

logger = logging.getLogger(__name__)

@dataclass
class CustomerProductPattern:
    customer_id: int
    product_id: int

    total_orders: int
    first_order_date: datetime
    last_order_date: datetime

    reorder_cycle_median: float
    reorder_cycle_iqr: float
    reorder_cycle_cv: float

    avg_quantity: float
    quantity_stddev: float
    quantity_trend: str

    seasonality_detected: bool
    seasonality_period_days: Optional[int]
    seasonality_strength: float

    trend_direction: str
    trend_slope: float
    trend_pvalue: float

    consistency_score: float
    status: str
    churn_probability: float
    days_since_last_order: int
    days_overdue: float

    rfm_recency_score: float
    rfm_frequency_score: float
    rfm_monetary_score: float
    rfm_consistency_score: float
    rfm_segment: str

    order_velocity: float
    order_acceleration: float
    velocity_trend: str

    pattern_confidence: float

class PatternAnalyzer:

    def __init__(self, conn):
        self.conn = conn
        logger.info("PatternAnalyzer initialized")

    def analyze_customer_product(
        self,
        customer_id: int,
        product_id: int,
        as_of_date: str
    ) -> Optional[CustomerProductPattern]:
        try:

            orders = self._get_order_history(customer_id, product_id, as_of_date)

            if len(orders) == 0:
                logger.debug(f"No orders for customer {customer_id}, product {product_id}")
                return None

            cycles = self._calculate_reorder_cycles(orders)

            cycle_median = np.median(cycles) if len(cycles) > 0 else None
            if len(cycles) > 0:
                cycle_q1, cycle_q3 = np.percentile(cycles, [25, 75])
                cycle_iqr = cycle_q3 - cycle_q1
                cycle_cv = np.std(cycles) / np.mean(cycles) if np.mean(cycles) > 0 else 0
            else:
                cycle_iqr = None
                cycle_cv = 0

            quantities = [o['quantity'] for o in orders]
            avg_quantity = np.mean(quantities)
            quantity_stddev = np.std(quantities) if len(quantities) > 1 else 0

            quantity_trend = self._detect_quantity_trend(orders)

            seasonality = self._detect_seasonality(cycles) if len(cycles) >= 12 else None

            trend = self._detect_frequency_trend(orders)

            consistency = self._calculate_consistency_score(
                cycle_cv, len(orders), cycle_iqr, cycle_median
            )

            days_since_last = (datetime.fromisoformat(as_of_date) - orders[-1]['date']).days

            rfm = self._calculate_rfm_features(
                orders, days_since_last, cycle_median, cycle_cv, as_of_date
            )

            velocity = self._calculate_order_velocity(orders)

            status, churn_prob, days_overdue = self._assess_churn_risk(
                days_since_last, cycle_median, cycle_iqr, len(orders), rfm
            )

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
        cycles = []
        for i in range(1, len(orders)):

            days = calculate_fractional_days(orders[i-1]['date'], orders[i]['date'])
            if days >= 1.0:
                cycles.append(days)
        return cycles

    def _detect_seasonality(self, cycles: List[float]) -> Optional[Dict]:
        if len(cycles) < 12:
            return None

        try:

            autocorr = np.correlate(cycles, cycles, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]

            peaks = []
            for i in range(2, min(len(autocorr)-1, 30)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.3:
                        peaks.append((i, autocorr[i]))

            if peaks:

                best_peak = max(peaks, key=lambda x: x[1])
                period_days = int(best_peak[0] * np.median(cycles))
                strength = best_peak[1]

                common_periods = [7, 14, 30, 60, 90]
                closest_period = min(common_periods, key=lambda x: abs(x - period_days))

                if abs(closest_period - period_days) / max(1, closest_period) < 0.3:
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
        if len(orders) < 4:
            return {'direction': 'stable', 'slope': 0.0, 'pvalue': 1.0}

        try:

            intervals = []
            for i in range(1, len(orders)):
                days = (orders[i]['date'] - orders[i-1]['date']).days
                if days > 0:
                    intervals.append(days)

            if len(intervals) < 3:
                return {'direction': 'stable', 'slope': 0.0, 'pvalue': 1.0}

            result = stats.kendalltau(range(len(intervals)), intervals)
            tau, pvalue = result.correlation, result.pvalue

            slopes = []
            for i in range(len(intervals)):
                for j in range(i+1, len(intervals)):
                    if j - i > 0:
                        slope = (intervals[j] - intervals[i]) / (j - i)
                        slopes.append(slope)

            median_slope = np.median(slopes) if slopes else 0.0

            if pvalue < 0.05:
                if median_slope > 0:
                    direction = 'declining'  # Increasing intervals = declining frequency
                else:
                    direction = 'growing'
            else:
                direction = 'stable'

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
        if len(orders) < 3:
            return 'stable'

        try:
            quantities = [o['quantity'] for o in orders]

            x = np.arange(len(quantities))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, quantities)

            if p_value < 0.05:
                avg_qty = np.mean(quantities)
                if slope > 0.1 * avg_qty / len(quantities):
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
        if median is None or median == 0 or iqr is None:
            return 0.0

        cv_score = 1 / (1 + cv)

        sample_score = min(1.0, n_orders / 20)

        relative_iqr = iqr / median
        iqr_score = 1 / (1 + relative_iqr)

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
        if cycle_median is None:
            return ('unknown', 0.5, 0.0)

        expected_reorder_days = cycle_median
        days_overdue = days_since_last - expected_reorder_days

        if days_overdue <= 0:

            status = 'active'
            churn_prob = 0.05

        elif cycle_iqr and days_overdue <= cycle_iqr:

            status = 'active'
            churn_prob = 0.15

        elif days_overdue <= cycle_median:

            status = 'at_risk'
            churn_prob = 0.40

        elif days_overdue <= 2 * cycle_median:

            status = 'at_risk'
            churn_prob = 0.70

        else:

            status = 'churned'
            churn_prob = 0.90

        loyalty_factor = min(1.0, n_orders / 10)
        churn_prob = churn_prob * (1 - 0.3 * loyalty_factor)

        if rfm:

            recency_adjustment = rfm['recency_score']

            frequency_adjustment = rfm['frequency_score']

            monetary_adjustment = rfm['monetary_score']

            rfm_loyalty = (
                0.4 * recency_adjustment +
                0.4 * frequency_adjustment +
                0.2 * monetary_adjustment
            )

            churn_reduction = 0.5 * rfm_loyalty
            churn_prob = churn_prob * (1 - churn_reduction)

            if status == 'at_risk' and rfm_loyalty > 0.7:

                status = 'active'
            elif status == 'churned' and rfm_loyalty > 0.6:

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

        prior = 0.5

        likelihood_sample = min(1.0, n_orders / 15)

        likelihood_consistency = consistency

        if cycle_median and cycle_median > 0:
            recency_factor = min(1.0, cycle_median / max(1, days_since_last))
        else:
            recency_factor = 0.5

        if rfm:

            rfm_consistency_factor = rfm['consistency_score']

            rfm_frequency_factor = rfm['frequency_score']

            rfm_recency_factor = rfm['recency_score']

            rfm_factor = (
                0.4 * rfm_consistency_factor +
                0.3 * rfm_frequency_factor +
                0.3 * rfm_recency_factor
            )

            posterior = (
                0.1 * prior +
                0.25 * likelihood_sample +
                0.20 * likelihood_consistency +
                0.10 * recency_factor +
                0.35 * rfm_factor
            )
        else:

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
        if len(orders) == 0:
            return {
                'recency_score': 0.0,
                'frequency_score': 0.0,
                'monetary_score': 0.0,
                'consistency_score': 0.0,
                'segment': 'new'
            }

        if cycle_median and cycle_median > 0:

            recency_ratio = cycle_median / max(1, days_since_last)
            recency_score = min(1.0, max(0.0, recency_ratio))
        else:

            if days_since_last < 30:
                recency_score = 1.0
            elif days_since_last < 90:
                recency_score = 0.7
            elif days_since_last < 180:
                recency_score = 0.4
            else:
                recency_score = 0.1

        first_order = orders[0]['date']
        last_order = orders[-1]['date']
        total_days = (datetime.fromisoformat(as_of_date) - first_order).days

        if total_days > 0:
            orders_per_year = len(orders) / (total_days / 365.25)
        else:
            orders_per_year = 0

        if orders_per_year < 1:
            frequency_score = 0.3 * orders_per_year
        elif orders_per_year < 4:
            frequency_score = 0.3 + 0.4 * (orders_per_year - 1) / 3
        else:
            frequency_score = 0.7 + 0.3 * min(1.0, (orders_per_year - 4) / 8)

        frequency_score = round(min(1.0, max(0.0, frequency_score)), 3)

        avg_quantity = np.mean([o['quantity'] for o in orders])

        if avg_quantity < 5:
            monetary_score = 0.3 + 0.4 * (avg_quantity / 5)
        elif avg_quantity < 20:
            monetary_score = 0.7 + 0.2 * ((avg_quantity - 5) / 15)
        else:
            monetary_score = 0.9 + 0.1 * min(1.0, (avg_quantity - 20) / 30)

        monetary_score = round(min(1.0, max(0.0, monetary_score)), 3)

        if cycle_cv > 0:

            if cycle_cv < 0.5:
                consistency_score = 0.7 + 0.3 * (1 - cycle_cv / 0.5)
            elif cycle_cv < 1.5:
                consistency_score = 0.3 + 0.4 * (1 - (cycle_cv - 0.5) / 1.0)
            else:
                consistency_score = 0.3 * max(0.0, 1 - (cycle_cv - 1.5) / 2.0)
        else:
            consistency_score = 1.0

        consistency_score = round(min(1.0, max(0.0, consistency_score)), 3)

        r_high = recency_score >= 0.6
        r_med = 0.3 <= recency_score < 0.6
        f_high = frequency_score >= 0.6
        f_med = 0.3 <= frequency_score < 0.6
        m_high = monetary_score >= 0.6

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
        if len(orders) < 3:
            return {
                'velocity': 0.0,
                'acceleration': 0.0,
                'trend': 'stable'
            }

        intervals = []
        for i in range(1, len(orders)):
            days = (orders[i]['date'] - orders[i-1]['date']).days
            if days > 0:
                intervals.append(days)

        if len(intervals) < 2:
            return {
                'velocity': 0.0,
                'acceleration': 0.0,
                'trend': 'stable'
            }

        if len(intervals) >= 4:

            split_point = max(1, len(intervals) * 3 // 4)
            historical_intervals = intervals[:split_point]
            recent_intervals = intervals[split_point:]

            historical_median = np.median(historical_intervals)
            recent_median = np.median(recent_intervals)

            if historical_median > 0:
                velocity = (recent_median - historical_median) / historical_median
            else:
                velocity = 0.0
        else:

            mid = len(intervals) // 2
            first_half_median = np.median(intervals[:mid+1])
            second_half_median = np.median(intervals[mid:])

            if first_half_median > 0:
                velocity = (second_half_median - first_half_median) / first_half_median
            else:
                velocity = 0.0

        if len(intervals) >= 6:

            third = len(intervals) // 3
            early_intervals = intervals[:third]
            middle_intervals = intervals[third:2*third]
            late_intervals = intervals[2*third:]

            early_median = np.median(early_intervals)
            middle_median = np.median(middle_intervals)
            late_median = np.median(late_intervals)

            if early_median > 0:
                vel1 = (middle_median - early_median) / early_median
            else:
                vel1 = 0.0

            if middle_median > 0:
                vel2 = (late_median - middle_median) / middle_median
            else:
                vel2 = 0.0

            acceleration = vel2 - vel1
        else:
            acceleration = 0.0

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
