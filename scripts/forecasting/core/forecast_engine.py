#!/usr/bin/env python3
"""
Forecast Engine - Main Orchestrator

Coordinates the complete forecasting pipeline:
1. Identify customers who ordered the product
2. Analyze patterns for each customer-product pair
3. Predict next orders for each customer
4. Aggregate into product-level forecast
5. Enrich with product/customer metadata

Production-grade orchestration with error handling and caching.
"""

import logging
import sys
from typing import List, Optional, Dict
from datetime import datetime
import json

# Add parent directory to path for datetime_utils
sys.path.insert(0, '/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts')
from datetime_utils import parse_as_of_date, format_date_iso

from .pattern_analyzer import PatternAnalyzer, CustomerProductPattern
from .customer_predictor import CustomerPredictor, CustomerPrediction
from .product_aggregator import ProductAggregator, ProductForecast

logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Production-grade forecast orchestrator

    Coordinates multi-layer forecasting pipeline:
    - Pattern Analysis Layer (robust statistics)
    - Prediction Layer (Bayesian inference)
    - Aggregation Layer (weekly bucketing)
    - Intelligence Layer (business insights)

    Uses connection pooling and caching for performance.
    """

    def __init__(self, conn, forecast_weeks: int = 12):
        """
        Initialize forecast engine

        Args:
            conn: pymssql database connection (from pool)
            forecast_weeks: Number of weeks to forecast (default 12 = ~3 months)
        """
        self.conn = conn
        self.forecast_weeks = forecast_weeks

        # Initialize components
        self.pattern_analyzer = PatternAnalyzer(conn)
        self.predictor = CustomerPredictor(forecast_horizon_days=forecast_weeks * 7)
        self.aggregator = ProductAggregator(forecast_weeks=forecast_weeks)

        logger.info(f"ForecastEngine initialized for {forecast_weeks} weeks")

    def generate_forecast(
        self,
        product_id: int,
        as_of_date: Optional[str] = None,
        min_orders: int = 2,
        min_confidence: float = 0.3
    ) -> Optional[ProductForecast]:
        """
        Generate complete product forecast

        Pipeline:
        1. Get customers who ordered this product
        2. Analyze pattern for each customer
        3. Predict next order for each customer
        4. Aggregate into product forecast
        5. Enrich with metadata

        Args:
            product_id: Product ID to forecast
            as_of_date: Reference date (ISO format, default: today)
            min_orders: Minimum orders required for pattern (default 2)
            min_confidence: Minimum prediction confidence (default 0.3)

        Returns:
            ProductForecast or None if no predictable customers
        """
        try:
            # Parse and validate as_of_date with timezone awareness
            as_of_dt = parse_as_of_date(as_of_date)
            as_of_date = format_date_iso(as_of_dt)

            logger.info(f"Generating forecast for product {product_id} as of {as_of_date}")

            # Step 1: Get customers who ordered this product
            customers = self._get_product_customers(product_id, as_of_date)

            if not customers:
                logger.warning(f"No customers found for product {product_id}")
                return None

            logger.info(f"Found {len(customers)} customers for product {product_id}")

            # Step 2: Analyze patterns and predict for each customer
            predictions = []
            patterns_analyzed = 0
            predictions_made = 0

            for customer_id in customers:
                # Analyze pattern
                pattern = self.pattern_analyzer.analyze_customer_product(
                    customer_id=customer_id,
                    product_id=product_id,
                    as_of_date=as_of_date
                )

                if pattern is None:
                    continue

                patterns_analyzed += 1

                # Filter by minimum orders
                if pattern.total_orders < min_orders:
                    logger.debug(
                        f"Customer {customer_id} has only {pattern.total_orders} orders, "
                        f"minimum is {min_orders}"
                    )
                    continue

                # Predict next order
                prediction = self.predictor.predict_next_order(
                    pattern=pattern,
                    as_of_date=as_of_date
                )

                if prediction is None:
                    continue

                # Filter by minimum confidence
                if prediction.prediction_confidence < min_confidence:
                    logger.debug(
                        f"Customer {customer_id} prediction confidence "
                        f"{prediction.prediction_confidence} below minimum {min_confidence}"
                    )
                    continue

                predictions.append(prediction)
                predictions_made += 1

            logger.info(
                f"Pattern analysis: {patterns_analyzed}/{len(customers)} customers, "
                f"Predictions: {predictions_made} (confidence >= {min_confidence})"
            )

            if not predictions:
                logger.warning(f"No predictable customers for product {product_id}")
                return None

            # Step 3: Get product metadata
            product_info = self._get_product_info(product_id)

            # Step 4: Aggregate into product-level forecast
            forecast = self.aggregator.aggregate_forecast(
                product_id=product_id,
                predictions=predictions,
                as_of_date=as_of_date,
                conn=self.conn,  # Pass database connection for historical data
                product_name=product_info.get('product_name'),
                unit_price=product_info.get('unit_price')
            )

            # Step 5: Enrich with customer names
            forecast = self._enrich_with_customer_names(forecast)

            logger.info(
                f"Forecast complete for product {product_id}: "
                f"{forecast.summary['total_predicted_quantity']} units, "
                f"{forecast.summary['total_predicted_orders']} orders"
            )

            return forecast

        except Exception as e:
            logger.error(f"Error generating forecast for product {product_id}: {e}")
            raise

    def _get_product_customers(
        self,
        product_id: int,
        as_of_date: str
    ) -> List[int]:
        """
        Get all customers who have ordered this product

        Returns list of customer IDs with historical orders
        """
        query = """
        SELECT DISTINCT ca.ClientID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE oi.ProductID = %s
              AND o.Created < %s
        ORDER BY ca.ClientID
        """

        cursor = self.conn.cursor()
        cursor.execute(query, (product_id, as_of_date))

        customers = [row[0] for row in cursor.fetchall()]
        cursor.close()

        return customers

    def _get_product_info(self, product_id: int) -> Dict:
        """
        Get product metadata (name, price, etc.)

        Returns dict with product_name and unit_price
        """
        query = """
        SELECT TOP 1
            p.Name as product_name,
            oi.PricePerItem as unit_price
        FROM dbo.Product p
        LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID
        WHERE p.ID = %s
              AND oi.PricePerItem IS NOT NULL
        ORDER BY oi.Created DESC
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query, (product_id,))

        row = cursor.fetchone()
        cursor.close()

        if row:
            return {
                'product_name': row['product_name'],
                'unit_price': float(row['unit_price']) if row['unit_price'] else 35.0
            }
        else:
            return {
                'product_name': f"Product {product_id}",
                'unit_price': 35.0  # Default price
            }

    def _get_customer_names(self, customer_ids: List[int]) -> Dict[int, str]:
        """
        Get customer names for list of IDs

        Returns: {customer_id: customer_name}
        """
        if not customer_ids:
            return {}

        # Build parameterized query
        placeholders = ','.join(['%s'] * len(customer_ids))
        query = f"""
        SELECT
            ID as customer_id,
            Name as customer_name
        FROM dbo.Client
        WHERE ID IN ({placeholders})
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query, customer_ids)

        customer_names = {
            row['customer_id']: row['customer_name']
            for row in cursor.fetchall()
        }
        cursor.close()

        return customer_names

    def _enrich_with_customer_names(self, forecast: ProductForecast) -> ProductForecast:
        """
        Enrich forecast with customer names

        Adds customer_name field to:
        - weekly_data.expected_customers
        - top_customers_by_volume
        - at_risk_customers
        """
        # Collect all customer IDs
        customer_ids = set()

        # From weekly data
        for week in forecast.weekly_data:
            for cust in week.get('expected_customers', []):
                customer_ids.add(cust['customer_id'])

        # From top customers
        for cust in forecast.top_customers_by_volume:
            customer_ids.add(cust['customer_id'])

        # From at-risk customers
        for cust in forecast.at_risk_customers:
            customer_ids.add(cust['customer_id'])

        if not customer_ids:
            return forecast

        # Fetch names
        customer_names = self._get_customer_names(list(customer_ids))

        # Enrich weekly data
        for week in forecast.weekly_data:
            for cust in week.get('expected_customers', []):
                cust_id = cust['customer_id']
                cust['customer_name'] = customer_names.get(
                    cust_id,
                    f"Customer {cust_id}"
                )

        # Enrich top customers
        for cust in forecast.top_customers_by_volume:
            cust_id = cust['customer_id']
            cust['customer_name'] = customer_names.get(
                cust_id,
                f"Customer {cust_id}"
            )

        # Enrich at-risk customers
        for cust in forecast.at_risk_customers:
            cust_id = cust['customer_id']
            cust['customer_name'] = customer_names.get(
                cust_id,
                f"Customer {cust_id}"
            )

        return forecast

    def generate_forecast_cached(
        self,
        product_id: int,
        redis_client,
        as_of_date: Optional[str] = None,
        cache_ttl: int = 3600
    ) -> Optional[ProductForecast]:
        """
        Generate forecast with Redis caching

        Cache key: forecast:product:{product_id}:{as_of_date}
        TTL: Default 1 hour (for ad-hoc requests)

        Args:
            product_id: Product ID
            redis_client: Redis client instance
            as_of_date: Reference date (default: today)
            cache_ttl: Cache TTL in seconds (default 3600 = 1 hour)

        Returns:
            ProductForecast from cache or fresh generation
        """
        # Parse and validate as_of_date with timezone awareness
        as_of_dt = parse_as_of_date(as_of_date)
        as_of_date = format_date_iso(as_of_dt)

        cache_key = f"forecast:product:{product_id}:{as_of_date}"

        try:
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Forecast cache HIT for product {product_id}")
                forecast_dict = json.loads(cached)
                # Reconstruct ProductForecast from dict
                return self._dict_to_forecast(forecast_dict)

        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        # Cache miss - generate forecast
        logger.info(f"Forecast cache MISS for product {product_id}")
        forecast = self.generate_forecast(
            product_id=product_id,
            as_of_date=as_of_date
        )

        if forecast:
            try:
                # Store in cache
                forecast_dict = self._forecast_to_dict(forecast)
                redis_client.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps(forecast_dict)
                )
                logger.info(f"Forecast cached for product {product_id} (TTL: {cache_ttl}s)")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

        return forecast

    def _forecast_to_dict(self, forecast: ProductForecast) -> Dict:
        """Convert ProductForecast to JSON-serializable dict"""
        return {
            'product_id': forecast.product_id,
            'forecast_period_weeks': forecast.forecast_period_weeks,
            'summary': forecast.summary,
            'weekly_forecasts': forecast.weekly_forecasts,
            'top_customers_by_volume': forecast.top_customers_by_volume,
            'at_risk_customers': forecast.at_risk_customers,
            'model_metadata': forecast.model_metadata
        }

    def _dict_to_forecast(self, data: Dict) -> ProductForecast:
        """Convert dict back to ProductForecast"""
        return ProductForecast(
            product_id=data['product_id'],
            forecast_period_weeks=data['forecast_period_weeks'],
            summary=data['summary'],
            weekly_forecasts=data['weekly_forecasts'],
            top_customers_by_volume=data['top_customers_by_volume'],
            at_risk_customers=data['at_risk_customers'],
            model_metadata=data['model_metadata']
        )
