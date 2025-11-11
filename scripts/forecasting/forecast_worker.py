#!/usr/bin/env python3

import os
import sys
import time
import logging
from datetime import datetime
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple
import redis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from api.db_pool import get_connection, close_pool
from scripts.forecasting import ForecastEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AS_OF_DATE = os.getenv('AS_OF_DATE', datetime.now().strftime('%Y-%m-%d'))
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
CACHE_TTL = int(os.getenv('FORECAST_CACHE_TTL', 604800))  # 7 days default
NUM_WORKERS = int(os.getenv('FORECAST_WORKERS', 10))
MIN_CUSTOMERS = int(os.getenv('FORECAST_MIN_CUSTOMERS', 2))
MIN_ORDERS = int(os.getenv('FORECAST_MIN_ORDERS', 3))

def _process_product_worker(args: Tuple[int, int, int, str, int]) -> Dict:
    product_id, index, total_count, as_of_date, cache_ttl = args

    try:

        conn = get_connection()

        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5
        )

        try:

            engine = ForecastEngine(conn=conn, forecast_weeks=12)

            start_time = time.time()
            forecast = engine.generate_forecast_cached(
                product_id=product_id,
                redis_client=redis_client,
                as_of_date=as_of_date,
                cache_ttl=cache_ttl
            )
            elapsed = time.time() - start_time

            if forecast:

                if (index + 1) % 100 == 0:
                    logger.info(
                        f"Progress: {index + 1}/{total_count} "
                        f"({(index + 1) / total_count * 100:.1f}%) - "
                        f"Product {product_id}: {forecast.summary['total_predicted_quantity']} units, "
                        f"{forecast.summary['active_customers']} customers, "
                        f"{elapsed:.2f}s"
                    )

                return {
                    'product_id': product_id,
                    'status': 'success',
                    'quantity': forecast.summary['total_predicted_quantity'],
                    'customers': forecast.summary['active_customers'],
                    'confidence': forecast.model_metadata['forecast_accuracy_estimate'],
                    'elapsed': elapsed
                }
            else:
                logger.debug(f"Product {product_id}: No forecast generated (insufficient data)")
                return {
                    'product_id': product_id,
                    'status': 'no_forecast',
                    'elapsed': elapsed
                }

        finally:

            conn.close()
            redis_client.close()

    except Exception as e:
        logger.error(f"Error processing product {product_id}: {e}")
        return {
            'product_id': product_id,
            'status': 'error',
            'error': str(e)
        }

class ForecastWorker:

    def __init__(self):
        self.as_of_date = AS_OF_DATE
        self.cache_ttl = CACHE_TTL
        self.num_workers = NUM_WORKERS
        self.min_customers = MIN_CUSTOMERS
        self.min_orders = MIN_ORDERS

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

        logger.info(f"ForecastWorker initialized:")
        logger.info(f"  AS_OF_DATE: {self.as_of_date}")
        logger.info(f"  Workers: {self.num_workers}")
        logger.info(f"  Cache TTL: {self.cache_ttl}s ({self.cache_ttl // 86400} days)")

    def get_forecastable_products(self) -> List[int]:
        logger.info("Fetching forecastable products from database...")

        conn = get_connection()
        cursor = conn.cursor()

        query = f"""
        SELECT oi.ProductID
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        INNER JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
        WHERE o.Created >= '2019-01-01'
          AND o.Created < %s
          AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        HAVING COUNT(DISTINCT ca.ClientID) >= %s
           AND COUNT(DISTINCT o.ID) >= %s
        ORDER BY COUNT(DISTINCT o.ID) DESC
        """

        cursor.execute(query, (self.as_of_date, self.min_customers, self.min_orders))
        products = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        logger.info(f"Found {len(products):,} forecastable products")
        return products

    def run(self) -> Dict:
        start_time = time.time()

        logger.info("="*80)
        logger.info("FORECAST WORKER STARTING")
        logger.info("="*80)

        products = self.get_forecastable_products()

        if not products:
            logger.warning("No forecastable products found!")
            return {
                'status': 'completed',
                'total_products': 0,
                'successful': 0,
                'failed': 0,
                'no_forecast': 0,
                'elapsed_seconds': 0
            }

        args_list = [
            (pid, idx, len(products), self.as_of_date, self.cache_ttl)
            for idx, pid in enumerate(products)
        ]

        logger.info(f"Starting parallel processing with {self.num_workers} workers...")
        logger.info(f"Processing {len(products):,} products...")

        results = []
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_process_product_worker, args_list)

        total_elapsed = time.time() - start_time

        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        no_forecast = sum(1 for r in results if r['status'] == 'no_forecast')

        total_quantity = sum(r.get('quantity', 0) for r in results if r['status'] == 'success')
        avg_customers = sum(r.get('customers', 0) for r in results if r['status'] == 'success') / max(1, successful)
        avg_confidence = sum(r.get('confidence', 0) for r in results if r['status'] == 'success') / max(1, successful)

        logger.info("="*80)
        logger.info("FORECAST WORKER COMPLETED")
        logger.info("="*80)
        logger.info(f"Total Products: {len(products):,}")
        logger.info(f"Successful: {successful:,} ({successful/len(products)*100:.1f}%)")
        logger.info(f"No Forecast: {no_forecast:,} ({no_forecast/len(products)*100:.1f}%)")
        logger.info(f"Failed: {failed:,} ({failed/len(products)*100:.1f}%)")
        logger.info(f"")
        logger.info(f"Total Predicted Quantity: {total_quantity:,.1f} units")
        logger.info(f"Avg Customers per Product: {avg_customers:.1f}")
        logger.info(f"Avg Forecast Confidence: {avg_confidence:.1%}")
        logger.info(f"")
        logger.info(f"Elapsed Time: {total_elapsed/60:.1f} minutes")
        logger.info(f"Avg Time per Product: {total_elapsed/len(products):.2f}s")
        logger.info("="*80)

        metadata = {
            'last_run': datetime.now().isoformat(),
            'as_of_date': self.as_of_date,
            'total_products': len(products),
            'successful': successful,
            'failed': failed,
            'no_forecast': no_forecast,
            'elapsed_seconds': int(total_elapsed)
        }

        try:
            import json
            self.redis_client.setex(
                'forecast:metadata:last_run',
                self.cache_ttl,
                json.dumps(metadata)
            )
            logger.info("Metadata stored in Redis")
        except Exception as e:
            logger.warning(f"Failed to store metadata in Redis: {e}")

        return metadata

def main():
    try:
        worker = ForecastWorker()
        result = worker.run()

        if result['failed'] > 0:
            logger.warning(f"{result['failed']} products failed to process")
            sys.exit(1)
        else:
            logger.info("All products processed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Worker failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:

        try:
            close_pool()
            logger.info("Database connection pool closed")
        except:
            pass

if __name__ == "__main__":
    main()
