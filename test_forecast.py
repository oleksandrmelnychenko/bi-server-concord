#!/usr/bin/env python3
"""
Quick test script for forecasting system

Tests the complete forecast pipeline with real data.
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from api.db_pool import get_connection, close_pool
from scripts.forecasting import ForecastEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_forecast(product_id: int, as_of_date: str = "2024-07-01"):
    """Test forecast generation for a product"""

    logger.info("="*60)
    logger.info(f"Testing forecast for product {product_id}")
    logger.info(f"As of date: {as_of_date}")
    logger.info("="*60)

    try:
        # Get connection from pool
        conn = get_connection()

        try:
            # Initialize forecast engine
            logger.info("Initializing ForecastEngine...")
            engine = ForecastEngine(conn=conn, forecast_weeks=12)

            # Generate forecast
            logger.info("Generating forecast...")
            start_time = datetime.now()

            forecast = engine.generate_forecast(
                product_id=product_id,
                as_of_date=as_of_date,
                min_orders=2,
                min_confidence=0.3
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Forecast generated in {elapsed:.2f}s")

            if forecast is None:
                logger.warning(f"No forecast available for product {product_id}")
                return None

            # Print summary
            logger.info("")
            logger.info("="*60)
            logger.info("FORECAST SUMMARY")
            logger.info("="*60)
            logger.info(f"Product ID: {forecast.product_id}")
            logger.info(f"Forecast Period: {forecast.forecast_period_weeks} weeks")
            logger.info("")
            logger.info(f"Total Predicted Quantity: {forecast.summary['total_predicted_quantity']}")
            logger.info(f"Total Predicted Revenue: ${forecast.summary['total_predicted_revenue']:,.2f}")
            logger.info(f"Total Predicted Orders: {forecast.summary['total_predicted_orders']}")
            logger.info(f"Active Customers: {forecast.summary['active_customers']}")
            logger.info(f"At-Risk Customers: {forecast.summary['at_risk_customers']}")
            logger.info("")

            # Print model metadata
            logger.info("MODEL METADATA:")
            logger.info(f"  Training Customers: {forecast.model_metadata['training_customers']}")
            logger.info(f"  Forecast Accuracy: {forecast.model_metadata['forecast_accuracy_estimate']:.1%}")
            logger.info(f"  Seasonality Detected: {forecast.model_metadata['seasonality_detected']}")
            logger.info("")

            # Print first 3 weeks
            logger.info("WEEKLY FORECASTS (first 3 weeks):")
            for i, week in enumerate(forecast.weekly_forecasts[:3]):
                logger.info(f"\n  Week {i+1}: {week['week_start']} to {week['week_end']}")
                logger.info(f"    Quantity: {week['predicted_quantity']} (CI: {week['confidence_lower']}-{week['confidence_upper']})")
                logger.info(f"    Revenue: ${week['predicted_revenue']:,.2f}")
                logger.info(f"    Orders: {week['predicted_orders']}")
                logger.info(f"    Expected Customers: {len(week['expected_customers'])}")

                # Print top customers for this week
                if week['expected_customers']:
                    logger.info(f"    Top customers:")
                    for cust in week['expected_customers'][:3]:
                        logger.info(
                            f"      - {cust.get('customer_name', f'Customer {cust['customer_id']}')} "
                            f"(prob: {cust['probability']:.1%}, qty: {cust['expected_quantity']})"
                        )

            # Print top customers
            if forecast.top_customers_by_volume:
                logger.info("\nTOP CUSTOMERS BY VOLUME:")
                for i, cust in enumerate(forecast.top_customers_by_volume[:5]):
                    logger.info(
                        f"  {i+1}. {cust.get('customer_name', f'Customer {cust['customer_id']}')} "
                        f"- {cust['predicted_quantity']} units ({cust['contribution_pct']:.1f}%)"
                    )

            # Print at-risk customers
            if forecast.at_risk_customers:
                logger.info("\nAT-RISK CUSTOMERS:")
                for i, cust in enumerate(forecast.at_risk_customers[:3]):
                    logger.info(
                        f"  {i+1}. {cust.get('customer_name', f'Customer {cust['customer_id']}')} "
                        f"- {cust['days_overdue']} days overdue, "
                        f"churn prob: {cust['churn_probability']:.1%}, "
                        f"action: {cust['action']}"
                    )

            logger.info("")
            logger.info("="*60)
            logger.info("TEST SUCCESSFUL!")
            logger.info("="*60)

            return forecast

        finally:
            # Return connection to pool
            conn.close()

    except Exception as e:
        logger.error(f"Error testing forecast: {e}", exc_info=True)
        return None
    finally:
        close_pool()


if __name__ == "__main__":
    # Test with a product ID
    # Using AS_OF_DATE from .env (2024-07-01)

    # You can change this to any product ID
    product_id = 25367399

    if len(sys.argv) > 1:
        product_id = int(sys.argv[1])

    result = test_forecast(product_id)

    if result:
        print("\nForecast test completed successfully!")
        sys.exit(0)
    else:
        print("\nForecast test failed!")
        sys.exit(1)
