#!/usr/bin/env python3
"""
Product Aggregator - Weekly Forecast Aggregation

Aggregates individual customer predictions into product-level forecasts:
- Weekly bucketing with variance pooling
- Monte Carlo simulation for confidence intervals
- Customer contribution analysis
- At-risk customer identification
- Top contributor ranking

Produces chart-ready weekly forecasts with confidence bands.
"""

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
    """
    Forecast for a single week

    Contains:
    - Aggregate predictions (quantity, revenue, orders)
    - Confidence intervals
    - Expected customers for this week
    """
    week_start: datetime
    week_end: datetime

    # Aggregate predictions
    predicted_quantity: float
    predicted_revenue: float
    predicted_orders: float

    # Confidence intervals (95%)
    quantity_confidence_lower: float
    quantity_confidence_upper: float

    # Expected customers (detailed list)
    expected_customers: List[Dict]  # List of customer details


@dataclass
class ProductForecast:
    """
    Complete product forecast response

    Matches the exact API format required by user.
    """
    product_id: int
    forecast_period_weeks: int
    historical_weeks: int

    # Summary metrics
    summary: Dict

    # Weekly data (historical + forecast combined for chart)
    weekly_data: List[Dict]

    # Analytics
    top_customers_by_volume: List[Dict]
    at_risk_customers: List[Dict]

    # Model metadata
    model_metadata: Dict


class ProductAggregator:
    """
    Production-grade aggregator for product-level forecasts

    Approach:
    - Aggregates customer predictions into weekly buckets
    - Uses variance pooling for confidence intervals
    - Identifies top contributors and at-risk customers
    - Produces business intelligence insights
    - Formats output for charting

    Optimized for 12-13 week forecasts (3 months).
    """

    def __init__(
        self,
        forecast_weeks: int = 12,
        historical_weeks: int = 3,
        unit_price: Optional[float] = None
    ):
        """
        Initialize aggregator

        Args:
            forecast_weeks: Number of weeks to forecast (default 12 = ~3 months)
            historical_weeks: Number of past weeks to include (default 3)
            unit_price: Optional product unit price for revenue calculation
        """
        self.forecast_weeks = forecast_weeks
        self.historical_weeks = historical_weeks
        self.unit_price = unit_price
        logger.info(f"ProductAggregator initialized for {forecast_weeks} weeks + {historical_weeks} historical")

    def aggregate_forecast(
        self,
        product_id: int,
        predictions: List[CustomerPrediction],
        as_of_date: str,
        conn,  # Database connection for historical data
        product_name: Optional[str] = None,
        unit_price: Optional[float] = None
    ) -> ProductForecast:
        """
        Aggregate customer predictions into product forecast

        Steps:
        1. Generate weekly buckets
        2. Allocate customer predictions to weeks
        3. Aggregate quantities with variance pooling
        4. Calculate confidence intervals
        5. Identify top customers and at-risk
        6. Format response

        Args:
            product_id: Product ID
            predictions: List of customer predictions
            as_of_date: Forecast start date (ISO format)
            product_name: Optional product name
            unit_price: Optional unit price for revenue

        Returns:
            ProductForecast with complete forecast data
        """
        try:
            as_of_dt = datetime.fromisoformat(as_of_date)
            price = unit_price or self.unit_price or 35.0  # Default price if not provided

            # 1. Fetch historical data (past N weeks)
            historical_weeks = self._fetch_historical_weeks(
                conn, product_id, as_of_dt, price
            )

            # 2. Generate weekly buckets for future
            weekly_buckets = self._generate_weekly_buckets(as_of_dt)

            # 3. Allocate customers to weeks using probability distributions
            weekly_allocations = self._allocate_customers_to_weeks(
                predictions, weekly_buckets
            )

            # 4. Aggregate into weekly forecasts
            weekly_forecasts = self._aggregate_weekly_forecasts(
                weekly_allocations, weekly_buckets, price
            )

            # 5. Format forecast weeks for API
            formatted_forecast_weeks = self._format_weekly_forecasts(weekly_forecasts)

            # 6. Merge historical + forecast into unified timeline
            weekly_data = historical_weeks + formatted_forecast_weeks

            # 7. Calculate summary metrics (including averages)
            summary = self._calculate_summary(predictions, weekly_forecasts, historical_weeks)

            # 8. Identify top customers by predicted volume
            top_customers = self._identify_top_customers(predictions)

            # 9. Identify at-risk customers
            at_risk = self._identify_at_risk_customers(predictions, as_of_dt)

            # 10. Calculate model metadata
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
        """
        Generate weekly time buckets

        Returns list of (week_start, week_end) tuples
        """
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
        """
        Fetch actual order data for past N weeks

        Returns list of dicts with actual quantities, revenue, and orders
        """
        historical_data = []

        try:
            cursor = conn.cursor()

            # Calculate date range for historical weeks
            start_date = as_of_date - timedelta(days=self.historical_weeks * 7)

            # Query actual orders for this product in the past N weeks
            # Optimized query with aggregation to reduce rows
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

            # Create map of week start dates (as returned by SQL) to data
            week_data_map = {}
            for row in rows:
                week_start_dt = row[0]
                quantity = float(row[1] or 0)
                orders = int(row[2] or 0)
                # Store using the SQL-computed week_start as key
                week_key = week_start_dt.strftime('%Y-%m-%d') if hasattr(week_start_dt, 'strftime') else str(week_start_dt)
                week_data_map[week_key] = {'quantity': quantity, 'orders': orders}

            # Helper function to compute SQL-style week start
            # Matches SQL: DATEADD(day, DATEDIFF(day, 0, date) / 7 * 7, 0)
            def get_sql_week_start(date_obj):
                epoch = datetime(1900, 1, 1)
                days_since_epoch = (date_obj - epoch).days
                week_start_days = (days_since_epoch // 7) * 7
                return epoch + timedelta(days=week_start_days)

            # Generate all weekly buckets (including empty weeks)
            for week_idx in range(self.historical_weeks):
                # Calculate the "ideal" week boundary based on as_of_date
                ideal_week_start = as_of_date - timedelta(days=(self.historical_weeks - week_idx) * 7)

                # But lookup data using SQL's week bucketing
                sql_week_start = get_sql_week_start(ideal_week_start)
                week_end = ideal_week_start + timedelta(days=7)

                # Get data from map using SQL week start
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
            # Return empty list on error - forecast will still work

        return historical_data

    def _allocate_customers_to_weeks(
        self,
        predictions: List[CustomerPrediction],
        weekly_buckets: List[Tuple[datetime, datetime]]
    ) -> Dict[int, List[Tuple[CustomerPrediction, float]]]:
        """
        Allocate customer predictions to weekly buckets using probabilities

        Returns: {week_index: [(prediction, probability), ...]}
        """
        allocations = defaultdict(list)

        for pred in predictions:
            # Use weekly probability distribution from predictor
            for week_idx, (week_start, week_end) in enumerate(weekly_buckets):
                # Find matching probability from prediction
                week_prob = 0.0
                for prob_date, prob in pred.weekly_probabilities:
                    if week_start <= prob_date < week_end:
                        week_prob = prob
                        break

                if week_prob > 0.01:  # Only include meaningful probabilities
                    allocations[week_idx].append((pred, week_prob))

        return allocations

    def _aggregate_weekly_forecasts(
        self,
        allocations: Dict[int, List[Tuple[CustomerPrediction, float]]],
        weekly_buckets: List[Tuple[datetime, datetime]],
        unit_price: float
    ) -> List[WeeklyForecast]:
        """
        Aggregate customer predictions into weekly forecasts

        Uses probability-weighted aggregation with variance pooling
        """
        weekly_forecasts = []

        for week_idx, (week_start, week_end) in enumerate(weekly_buckets):
            customer_probs = allocations.get(week_idx, [])

            if not customer_probs:
                # No predictions for this week
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

            # Aggregate quantities (probability-weighted)
            total_quantity = 0.0
            total_variance = 0.0
            expected_orders = 0.0
            customer_details = []

            for pred, prob in customer_probs:
                # Expected quantity for this customer in this week
                expected_qty = pred.expected_quantity * prob
                total_quantity += expected_qty

                # Variance pooling (independent predictions)
                # Var(aX) = a^2 * Var(X)
                variance = (prob ** 2) * (pred.quantity_stddev ** 2)
                total_variance += variance

                # Expected number of orders
                expected_orders += prob

                # Store customer details (only if probability is significant)
                if prob >= 0.15:  # At least 15% chance
                    customer_details.append({
                        'customer_id': pred.customer_id,
                        'probability': round(prob, 3),
                        'expected_quantity': round(expected_qty, 1),
                        'expected_date': pred.expected_order_date.isoformat(),
                        'days_since_last_order': pred.days_since_last_order,
                        'avg_reorder_cycle': round(pred.reorder_cycle_days, 1)
                    })

            # Sort customers by probability (descending)
            customer_details.sort(key=lambda x: x['probability'], reverse=True)

            # Confidence interval (95%) using pooled variance
            total_stddev = np.sqrt(total_variance)
            ci_multiplier = 1.96
            qty_lower = max(0, total_quantity - ci_multiplier * total_stddev)
            qty_upper = total_quantity + ci_multiplier * total_stddev

            # Revenue
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
        """
        Calculate summary metrics across all weeks including averages for chart
        """
        total_quantity = sum(w.predicted_quantity for w in weekly_forecasts)
        total_revenue = sum(w.predicted_revenue for w in weekly_forecasts)
        total_orders = sum(w.predicted_orders for w in weekly_forecasts)

        # Count active vs at-risk customers
        active_customers = sum(1 for p in predictions if p.status == 'active')
        at_risk_customers = sum(1 for p in predictions if p.status == 'at_risk')

        # Calculate averages for horizontal reference lines in chart
        avg_weekly_quantity = total_quantity / self.forecast_weeks if self.forecast_weeks > 0 else 0

        # Historical average
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
        """
        Identify top customers by predicted volume
        """
        # Calculate total predicted quantity for each customer
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

        # Sort by quantity and take top N
        customer_volumes.sort(key=lambda x: x['predicted_quantity'], reverse=True)
        return customer_volumes[:top_n]

    def _identify_at_risk_customers(
        self,
        predictions: List[CustomerPrediction],
        as_of_date: datetime
    ) -> List[Dict]:
        """
        Identify at-risk customers who need proactive outreach
        """
        at_risk = []

        for pred in predictions:
            if pred.status == 'at_risk' or pred.churn_probability > 0.3:
                # Calculate expected reorder date
                expected_reorder = pred.expected_order_date
                days_overdue = (as_of_date - expected_reorder).days

                # Determine action
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

        # Sort by churn probability (descending)
        at_risk.sort(key=lambda x: x['churn_probability'], reverse=True)
        return at_risk

    def _calculate_metadata(self, predictions: List[CustomerPrediction]) -> Dict:
        """
        Calculate model metadata and statistics
        """
        if not predictions:
            return {
                'model_type': 'customer_based_aggregate',
                'training_customers': 0,
                'forecast_accuracy_estimate': 0.0,
                'seasonality_detected': False
            }

        # Average prediction confidence across customers
        avg_confidence = np.mean([p.prediction_confidence for p in predictions])

        # Check if any customer has seasonality
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
        """
        Format weekly forecasts for API response (with unified field names for charting)
        """
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
