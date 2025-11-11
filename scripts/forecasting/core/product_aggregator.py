#!/usr/bin/env python3

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .customer_predictor import CustomerPrediction

logger = logging.getLogger(__name__)

@dataclass
class WeeklyForecast:
    week_start: datetime
    week_end: datetime

    predicted_quantity: float
    predicted_revenue: float
    predicted_orders: float

    quantity_confidence_lower: float
    quantity_confidence_upper: float

    expected_customers: List[Dict]

@dataclass
class ProductForecast:
    product_id: int
    forecast_period_weeks: int
    historical_weeks: int

    summary: Dict

    weekly_data: List[Dict]

    top_customers_by_volume: List[Dict]
    at_risk_customers: List[Dict]

    model_metadata: Dict

class ProductAggregator:

    def __init__(
        self,
        forecast_weeks: int = 12,
        historical_weeks: int = 3,
        unit_price: Optional[float] = None
    ):
        self.forecast_weeks = forecast_weeks
        self.historical_weeks = historical_weeks
        self.unit_price = unit_price
        logger.info(f"ProductAggregator initialized for {forecast_weeks} weeks + {historical_weeks} historical")

    def aggregate_forecast(
        self,
        product_id: int,
        predictions: List[CustomerPrediction],
        as_of_date: str,
        conn,
        product_name: Optional[str] = None,
        unit_price: Optional[float] = None
    ) -> ProductForecast:
        try:
            as_of_dt = datetime.fromisoformat(as_of_date)
            price = unit_price or self.unit_price or 35.0

            historical_weeks = self._fetch_historical_weeks(
                conn, product_id, as_of_dt, price
            )

            weekly_buckets = self._generate_weekly_buckets(as_of_dt)

            weekly_allocations = self._allocate_customers_to_weeks(
                predictions, weekly_buckets
            )

            weekly_forecasts = self._aggregate_weekly_forecasts(
                weekly_allocations, weekly_buckets, price
            )

            formatted_forecast_weeks = self._format_weekly_forecasts(weekly_forecasts)

            weekly_data = historical_weeks + formatted_forecast_weeks

            summary = self._calculate_summary(predictions, weekly_forecasts, historical_weeks)

            top_customers = self._identify_top_customers(predictions)

            at_risk = self._identify_at_risk_customers(predictions, as_of_dt)

            metadata = self._calculate_metadata(predictions)

            return ProductForecast(
                product_id=product_id,
                forecast_period_weeks=self.forecast_weeks,
                historical_weeks=self.historical_weeks,
                summary=summary,
                weekly_data=weekly_data,
                top_customers_by_volume=top_customers,
                at_risk_customers=at_risk,
                model_metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error aggregating forecast for product {product_id}: {e}")
            raise

    def _generate_weekly_buckets(self, start_date: datetime) -> List[Tuple[datetime, datetime]]:
        buckets = []
        for week_idx in range(self.forecast_weeks):
            week_start = start_date + timedelta(days=week_idx * 7)
            week_end = week_start + timedelta(days=7)
            buckets.append((week_start, week_end))
        return buckets

    def _fetch_historical_weeks(
        self,
        conn,
        product_id: int,
        as_of_date: datetime,
        unit_price: float
    ) -> List[Dict]:
        historical_data = []

        try:
            cursor = conn.cursor()

            start_date = as_of_date - timedelta(days=self.historical_weeks * 7)

            query = """
            SELECT
                DATEADD(day, DATEDIFF(day, 0, o.Created) / 7 * 7, 0) as week_start,
                SUM(oi.Qty) as total_quantity,
                COUNT(DISTINCT o.ID) as total_orders
            FROM dbo.OrderItem oi WITH (NOLOCK)
            INNER JOIN dbo.[Order] o WITH (NOLOCK) ON oi.OrderID = o.ID
            WHERE oi.ProductID = %s
              AND o.Created >= %s
              AND o.Created < %s
            GROUP BY DATEADD(day, DATEDIFF(day, 0, o.Created) / 7 * 7, 0)
            ORDER BY week_start
            """

            cursor.execute(query, (product_id, start_date, as_of_date))
            rows = cursor.fetchall()

            week_data_map = {}
            for row in rows:
                week_start_dt = row[0]
                quantity = float(row[1] or 0)
                orders = int(row[2] or 0)

                week_key = week_start_dt.strftime('%Y-%m-%d') if hasattr(week_start_dt, 'strftime') else str(week_start_dt)
                week_data_map[week_key] = {'quantity': quantity, 'orders': orders}

            def get_sql_week_start(date_obj):
                epoch = datetime(1900, 1, 1)
                days_since_epoch = (date_obj - epoch).days
                week_start_days = (days_since_epoch // 7) * 7
                return epoch + timedelta(days=week_start_days)

            for week_idx in range(self.historical_weeks):

                ideal_week_start = as_of_date - timedelta(days=(self.historical_weeks - week_idx) * 7)

                sql_week_start = get_sql_week_start(ideal_week_start)
                week_end = ideal_week_start + timedelta(days=7)

                week_key = sql_week_start.strftime('%Y-%m-%d')
                week_data = week_data_map.get(week_key, {'quantity': 0, 'orders': 0})

                revenue = week_data['quantity'] * unit_price
                historical_data.append({
                    'week_start': ideal_week_start.strftime('%Y-%m-%d'),
                    'week_end': week_end.strftime('%Y-%m-%d'),
                    'quantity': round(week_data['quantity'], 1),
                    'revenue': round(revenue, 2),
                    'orders': week_data['orders'],
                    'data_type': 'actual',
                    'confidence_lower': None,
                    'confidence_upper': None,
                    'expected_customers': []
                })

            cursor.close()
            logger.info(f"Fetched {len(historical_data)} historical weeks for product {product_id}")

        except Exception as e:
            logger.error(f"Error fetching historical data for product {product_id}: {e}")

        return historical_data

    def _allocate_customers_to_weeks(
        self,
        predictions: List[CustomerPrediction],
        weekly_buckets: List[Tuple[datetime, datetime]]
    ) -> Dict[int, List[Tuple[CustomerPrediction, float]]]:
        allocations = defaultdict(list)

        for pred in predictions:

            for week_idx, (week_start, week_end) in enumerate(weekly_buckets):

                week_prob = 0.0
                for prob_date, prob in pred.weekly_probabilities:
                    if week_start <= prob_date < week_end:
                        week_prob = prob
                        break

                if week_prob > 0.01:
                    allocations[week_idx].append((pred, week_prob))

        return allocations

    def _aggregate_weekly_forecasts(
        self,
        allocations: Dict[int, List[Tuple[CustomerPrediction, float]]],
        weekly_buckets: List[Tuple[datetime, datetime]],
        unit_price: float
    ) -> List[WeeklyForecast]:
        weekly_forecasts = []

        for week_idx, (week_start, week_end) in enumerate(weekly_buckets):
            customer_probs = allocations.get(week_idx, [])

            if not customer_probs:

                weekly_forecasts.append(WeeklyForecast(
                    week_start=week_start,
                    week_end=week_end,
                    predicted_quantity=0,
                    predicted_revenue=0,
                    predicted_orders=0,
                    quantity_confidence_lower=0,
                    quantity_confidence_upper=0,
                    expected_customers=[]
                ))
                continue

            total_quantity = 0.0
            total_variance = 0.0
            expected_orders = 0.0
            customer_details = []

            for pred, prob in customer_probs:

                expected_qty = pred.expected_quantity * prob
                total_quantity += expected_qty

                variance = (prob ** 2) * (pred.quantity_stddev ** 2)
                total_variance += variance

                expected_orders += prob

                if prob >= 0.15:
                    customer_details.append({
                        'customer_id': pred.customer_id,
                        'probability': round(prob, 3),
                        'expected_quantity': round(expected_qty, 1),
                        'expected_date': pred.expected_order_date.isoformat(),
                        'days_since_last_order': pred.days_since_last_order,
                        'avg_reorder_cycle': round(pred.reorder_cycle_days, 1)
                    })

            customer_details.sort(key=lambda x: x['probability'], reverse=True)

            total_stddev = np.sqrt(total_variance)
            ci_multiplier = 1.96
            qty_lower = max(0, total_quantity - ci_multiplier * total_stddev)
            qty_upper = total_quantity + ci_multiplier * total_stddev

            predicted_revenue = total_quantity * unit_price

            weekly_forecasts.append(WeeklyForecast(
                week_start=week_start,
                week_end=week_end,
                predicted_quantity=round(total_quantity, 1),
                predicted_revenue=round(predicted_revenue, 2),
                predicted_orders=round(expected_orders, 1),
                quantity_confidence_lower=round(qty_lower, 1),
                quantity_confidence_upper=round(qty_upper, 1),
                expected_customers=customer_details
            ))

        return weekly_forecasts

    def _calculate_summary(
        self,
        predictions: List[CustomerPrediction],
        weekly_forecasts: List[WeeklyForecast],
        historical_weeks: List[Dict]
    ) -> Dict:
        total_quantity = sum(w.predicted_quantity for w in weekly_forecasts)
        total_revenue = sum(w.predicted_revenue for w in weekly_forecasts)
        total_orders = sum(w.predicted_orders for w in weekly_forecasts)

        active_customers = sum(1 for p in predictions if p.status == 'active')
        at_risk_customers = sum(1 for p in predictions if p.status == 'at_risk')

        avg_weekly_quantity = total_quantity / self.forecast_weeks if self.forecast_weeks > 0 else 0

        historical_avg = 0.0
        if historical_weeks:
            historical_qty = sum(w.get('quantity', 0) for w in historical_weeks)
            historical_avg = historical_qty / len(historical_weeks) if len(historical_weeks) > 0 else 0

        return {
            'total_predicted_quantity': round(total_quantity, 1),
            'total_predicted_revenue': round(total_revenue, 2),
            'total_predicted_orders': round(total_orders, 1),
            'average_weekly_quantity': round(avg_weekly_quantity, 1),
            'historical_average': round(historical_avg, 1),
            'active_customers': active_customers,
            'at_risk_customers': at_risk_customers
        }

    def _identify_top_customers(
        self,
        predictions: List[CustomerPrediction],
        top_n: int = 10
    ) -> List[Dict]:

        customer_volumes = []
        total_volume = sum(
            p.expected_quantity * p.probability_orders_this_period
            for p in predictions
        )

        for pred in predictions:
            predicted_qty = pred.expected_quantity * pred.probability_orders_this_period
            contribution_pct = (predicted_qty / total_volume * 100) if total_volume > 0 else 0

            customer_volumes.append({
                'customer_id': pred.customer_id,
                'predicted_quantity': round(predicted_qty, 1),
                'contribution_pct': round(contribution_pct, 1)
            })

        customer_volumes.sort(key=lambda x: x['predicted_quantity'], reverse=True)
        return customer_volumes[:top_n]

    def _identify_at_risk_customers(
        self,
        predictions: List[CustomerPrediction],
        as_of_date: datetime
    ) -> List[Dict]:
        at_risk = []

        for pred in predictions:
            if pred.status == 'at_risk' or pred.churn_probability > 0.3:

                expected_reorder = pred.expected_order_date
                days_overdue = (as_of_date - expected_reorder).days

                if pred.churn_probability > 0.7:
                    action = "urgent_outreach_required"
                elif pred.churn_probability > 0.4:
                    action = "proactive_outreach_recommended"
                else:
                    action = "monitor_closely"

                at_risk.append({
                    'customer_id': pred.customer_id,
                    'last_order': (
                        as_of_date - timedelta(days=pred.days_since_last_order)
                    ).strftime('%Y-%m-%d'),
                    'expected_reorder': expected_reorder.strftime('%Y-%m-%d'),
                    'days_overdue': max(0, days_overdue),
                    'churn_probability': round(pred.churn_probability, 3),
                    'action': action
                })

        at_risk.sort(key=lambda x: x['churn_probability'], reverse=True)
        return at_risk

    def _calculate_metadata(self, predictions: List[CustomerPrediction]) -> Dict:
        if not predictions:
            return {
                'model_type': 'customer_based_aggregate',
                'training_customers': 0,
                'forecast_accuracy_estimate': 0.0,
                'seasonality_detected': False
            }

        avg_confidence = np.mean([p.prediction_confidence for p in predictions])

        seasonality_detected = any(
            len(p.weekly_probabilities) > 0 for p in predictions
        )

        return {
            'model_type': 'customer_based_aggregate',
            'training_customers': len(predictions),
            'forecast_accuracy_estimate': round(avg_confidence, 3),
            'seasonality_detected': seasonality_detected,
            'model_version': '1.0.0',
            'statistical_methods': [
                'bayesian_inference',
                'mann_kendall_trend',
                'fft_seasonality',
                'survival_analysis',
                'rfm_analysis',
                'velocity_tracking',
                'robust_statistics'
            ]
        }

    def _format_weekly_forecasts(self, weekly_forecasts: List[WeeklyForecast]) -> List[Dict]:
        formatted = []

        for week in weekly_forecasts:
            formatted.append({
                'week_start': week.week_start.strftime('%Y-%m-%d'),
                'week_end': week.week_end.strftime('%Y-%m-%d'),
                'quantity': week.predicted_quantity,  # Unified field name (not "predicted_quantity")
                'revenue': week.predicted_revenue,    # Unified field name
                'orders': week.predicted_orders,      # Unified field name
                'data_type': 'predicted',             # Chart type indicator
                'confidence_lower': week.quantity_confidence_lower,
                'confidence_upper': week.quantity_confidence_upper,
                'expected_customers': week.expected_customers
            })

        return formatted
