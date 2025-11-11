#!/usr/bin/env python3

import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
from typing import List, Dict
import redis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from api.db_pool import get_connection, close_pool
from scripts.forecasting import ForecastEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TEST_PRODUCTS = 500
AS_OF_DATE = os.getenv('AS_OF_DATE', datetime.now().strftime('%Y-%m-%d'))
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class RFMAccuracyTester:

    def __init__(self):
        self.as_of_date = AS_OF_DATE

        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"✓ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get_test_products(self, limit: int = TEST_PRODUCTS) -> List[int]:
        logger.info(f"Fetching {limit} test products...")

        conn = get_connection()
        cursor = conn.cursor()

        query = """
        SELECT
            oi.ProductID,
            COUNT(DISTINCT o.ID) as order_count,
            COUNT(DISTINCT ca.ClientID) as customer_count
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        INNER JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
        WHERE o.Created >= '2019-01-01'
          AND o.Created < %s
          AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        HAVING COUNT(DISTINCT ca.ClientID) >= 2
           AND COUNT(DISTINCT o.ID) >= 3
        ORDER BY NEWID()  -- Random sample
        """

        cursor.execute(query, (self.as_of_date,))
        products = [row[0] for row in cursor.fetchmany(limit)]

        cursor.close()
        conn.close()

        logger.info(f"Selected {len(products)} products for testing")
        return products

    def generate_forecasts(self, products: List[int]) -> Dict:
        logger.info(f"Generating forecasts for {len(products)} products...")

        results = {}
        conn = get_connection()

        for idx, product_id in enumerate(products):
            try:
                if (idx + 1) % 100 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(products)} ({(idx + 1)/len(products)*100:.1f}%)")

                engine = ForecastEngine(conn=conn, forecast_weeks=12)
                forecast = engine.generate_forecast_cached(
                    product_id=product_id,
                    redis_client=self.redis_client,
                    as_of_date=self.as_of_date,
                    cache_ttl=604800
                )

                if forecast:
                    results[product_id] = {
                        'total_quantity': forecast.summary['total_predicted_quantity'],
                        'total_revenue': forecast.summary['total_predicted_revenue'],
                        'active_customers': forecast.summary['active_customers'],
                        'at_risk_customers': forecast.summary['at_risk_customers'],
                        'confidence': forecast.model_metadata['forecast_accuracy_estimate'],
                        'status': 'success'
                    }
                else:
                    results[product_id] = {'status': 'no_forecast'}

            except Exception as e:
                logger.error(f"Error forecasting product {product_id}: {e}")
                results[product_id] = {'status': 'error', 'error': str(e)}

        conn.close()
        return results

    def calculate_metrics(self, results: Dict) -> Dict:
        successful = [r for r in results.values() if r['status'] == 'success']

        if not successful:
            return {
                'total_products': len(results),
                'successful': 0,
                'avg_confidence': 0.0,
                'median_confidence': 0.0,
                'std_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }

        confidences = [r['confidence'] for r in successful]

        return {
            'total_products': len(results),
            'successful': len(successful),
            'success_rate': len(successful) / len(results),
            'avg_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'avg_quantity': np.mean([r['total_quantity'] for r in successful]),
            'avg_customers': np.mean([r['active_customers'] for r in successful]),
            'avg_at_risk_pct': np.mean([
                r['at_risk_customers'] / max(1, r['active_customers'])
                for r in successful
            ])
        }

    def run(self) -> Dict:
        start_time = time.time()

        logger.info("="*80)
        logger.info("RFM ACCURACY TEST STARTING")
        logger.info("="*80)

        products = self.get_test_products(TEST_PRODUCTS)

        if not products:
            logger.warning("No products found for testing!")
            return {}

        logger.info("\nGenerating forecasts with RFM enhancements...")
        results = self.generate_forecasts(products)

        logger.info("\nCalculating metrics...")
        metrics = self.calculate_metrics(results)

        total_elapsed = time.time() - start_time

        logger.info("="*80)
        logger.info("RFM ACCURACY TEST COMPLETED")
        logger.info("="*80)
        logger.info(f"Total Products Tested: {metrics['total_products']}")
        logger.info(f"Successful Forecasts: {metrics['successful']} ({metrics.get('success_rate', 0)*100:.1f}%)")
        logger.info("")
        logger.info("CONFIDENCE METRICS (with RFM):")
        logger.info(f"  Average:  {metrics.get('avg_confidence', 0):.1%}")
        logger.info(f"  Median:   {metrics.get('median_confidence', 0):.1%}")
        logger.info(f"  Std Dev:  {metrics.get('std_confidence', 0):.1%}")
        logger.info(f"  Min:      {metrics.get('min_confidence', 0):.1%}")
        logger.info(f"  Max:      {metrics.get('max_confidence', 0):.1%}")
        logger.info("")
        logger.info("FORECAST CHARACTERISTICS:")
        logger.info(f"  Avg Quantity per Product: {metrics.get('avg_quantity', 0):.1f} units")
        logger.info(f"  Avg Active Customers: {metrics.get('avg_customers', 0):.1f}")
        logger.info(f"  Avg At-Risk %: {metrics.get('avg_at_risk_pct', 0):.1%}")
        logger.info("")
        logger.info(f"Elapsed Time: {total_elapsed/60:.1f} minutes")
        logger.info(f"Avg Time per Product: {total_elapsed/len(products):.2f}s")
        logger.info("="*80)

        metrics['elapsed_seconds'] = int(total_elapsed)
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['as_of_date'] = self.as_of_date

        return metrics

def main():
    try:
        tester = RFMAccuracyTester()
        result = tester.run()

        if result.get('successful', 0) == 0:
            logger.error("No successful forecasts generated!")
            sys.exit(1)
        else:
            logger.info(f"\n✓ Test completed: {result['successful']}/{result['total_products']} products forecasted")
            logger.info(f"  Average Confidence with RFM: {result['avg_confidence']:.1%}")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:

        try:
            close_pool()
            logger.info("Database connection pool closed")
        except:
            pass

if __name__ == "__main__":
    main()
