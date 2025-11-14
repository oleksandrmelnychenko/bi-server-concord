#!/usr/bin/env python3

import logging
import sys
from typing import List, Optional, Dict
from datetime import datetime
import json

sys.path.insert(0, '/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts')
from datetime_utils import parse_as_of_date, format_date_iso

from .pattern_analyzer import PatternAnalyzer, CustomerProductPattern
from .customer_predictor import CustomerPredictor, CustomerPrediction
from .product_aggregator import ProductAggregator, ProductForecast

logger = logging.getLogger(__name__)

class ForecastEngine:

    def __init__(self, conn, forecast_weeks: int = 12):
        self.conn = conn
        self.forecast_weeks = forecast_weeks

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
        try:

            as_of_dt = parse_as_of_date(as_of_date)
            as_of_date = format_date_iso(as_of_dt)

            logger.info(f"Generating forecast for product {product_id} as of {as_of_date}")

            customers = self._get_product_customers(product_id, as_of_date)

            if not customers:
                logger.warning(f"No customers found for product {product_id}")
                return None

            logger.info(f"Found {len(customers)} customers for product {product_id}")

            predictions = []
            patterns_analyzed = 0
            predictions_made = 0

            for customer_id in customers:

                pattern = self.pattern_analyzer.analyze_customer_product(
                    customer_id=customer_id,
                    product_id=product_id,
                    as_of_date=as_of_date
                )

                if pattern is None:
                    continue

                patterns_analyzed += 1

                if pattern.total_orders < min_orders:
                    logger.debug(
                        f"Customer {customer_id} has only {pattern.total_orders} orders, "
                        f"minimum is {min_orders}"
                    )
                    continue

                prediction = self.predictor.predict_next_order(
                    pattern=pattern,
                    as_of_date=as_of_date
                )

                if prediction is None:
                    continue

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

            product_info = self._get_product_info(product_id)

            forecast = self.aggregator.aggregate_forecast(
                product_id=product_id,
                predictions=predictions,
                as_of_date=as_of_date,
                conn=self.conn,
                product_name=product_info.get('product_name'),
                unit_price=product_info.get('unit_price')
            )

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
        if not customer_ids:
            return {}

        placeholders = ','.join(['%s'] * len(customer_ids))
        query = f"""
        SELECT
            ID as customer_id,
            Name as customer_name
        FROM dbo.Client
        WHERE ID IN ({placeholders})
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query, tuple(customer_ids))

        customer_names = {
            row['customer_id']: row['customer_name']
            for row in cursor.fetchall()
        }
        cursor.close()

        return customer_names

    def _enrich_with_customer_names(self, forecast: ProductForecast) -> ProductForecast:

        customer_ids = set()

        for week in forecast.weekly_data:
            for cust in week.get('expected_customers', []):
                customer_ids.add(cust['customer_id'])

        for cust in forecast.top_customers_by_volume:
            customer_ids.add(cust['customer_id'])

        for cust in forecast.at_risk_customers:
            customer_ids.add(cust['customer_id'])

        if not customer_ids:
            return forecast

        customer_names = self._get_customer_names(list(customer_ids))

        for week in forecast.weekly_data:
            for cust in week.get('expected_customers', []):
                cust_id = cust['customer_id']
                cust['customer_name'] = customer_names.get(
                    cust_id,
                    f"Customer {cust_id}"
                )

        for cust in forecast.top_customers_by_volume:
            cust_id = cust['customer_id']
            cust['customer_name'] = customer_names.get(
                cust_id,
                f"Customer {cust_id}"
            )

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

        as_of_dt = parse_as_of_date(as_of_date)
        as_of_date = format_date_iso(as_of_dt)

        cache_key = f"forecast:product:{product_id}:{as_of_date}"

        try:

            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Forecast cache HIT for product {product_id}")
                forecast_dict = json.loads(cached)

                return self._dict_to_forecast(forecast_dict)

        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        logger.info(f"Forecast cache MISS for product {product_id}")
        forecast = self.generate_forecast(
            product_id=product_id,
            as_of_date=as_of_date
        )

        if forecast:
            try:

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
        return ProductForecast(
            product_id=data['product_id'],
            forecast_period_weeks=data['forecast_period_weeks'],
            summary=data['summary'],
            weekly_forecasts=data['weekly_forecasts'],
            top_customers_by_volume=data['top_customers_by_volume'],
            at_risk_customers=data['at_risk_customers'],
            model_metadata=data['model_metadata']
        )
