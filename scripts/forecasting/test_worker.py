#!/usr/bin/env python3

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.forecasting.forecast_worker import ForecastWorker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_worker(limit: int = 50):
    logger.info("="*80)
    logger.info(f"FORECAST WORKER TEST - {limit} PRODUCTS")
    logger.info("="*80)

    try:

        worker = ForecastWorker()

        all_products = worker.get_forecastable_products()

        if not all_products:
            logger.error("No forecastable products found!")
            sys.exit(1)

        logger.info(f"Total forecastable products: {len(all_products):,}")

        test_products = all_products[:limit]
        logger.info(f"Testing with {len(test_products)} products")

        original_get_products = worker.get_forecastable_products
        worker.get_forecastable_products = lambda: test_products

        start_time = time.time()
        result = worker.run()
        elapsed = time.time() - start_time

        logger.info("="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Products Tested: {len(test_products)}")
        logger.info(f"Successful: {result['successful']} ({result['successful']/len(test_products)*100:.1f}%)")
        logger.info(f"No Forecast: {result['no_forecast']} ({result['no_forecast']/len(test_products)*100:.1f}%)")
        logger.info(f"Failed: {result['failed']} ({result['failed']/len(test_products)*100:.1f}%)")
        logger.info(f"Elapsed Time: {elapsed:.1f}s")
        logger.info(f"Avg Time per Product: {elapsed/len(test_products):.2f}s")
        logger.info("")
        logger.info(f"Projected time for all {len(all_products):,} products: {elapsed/len(test_products)*len(all_products)/60:.1f} minutes")
        logger.info("="*80)

        if result['failed'] > 0:
            logger.warning("Some products failed to process!")
            sys.exit(1)
        else:
            logger.info("✓ All test products processed successfully")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)

def main():
    limit = 50

    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid limit: {sys.argv[1]}")
            sys.exit(1)

    test_worker(limit)

if __name__ == "__main__":
    main()
