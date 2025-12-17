#!/usr/bin/env python3
"""
Backtest point-estimate accuracy for the forecasting engine.

This script:
- Samples forecastable products from MSSQL.
- Trains as of AS_OF_DATE and compares forecasted totals to actual totals over the forecast horizon.
- Reports MAPE and hit-rate@±20% by segment (low/medium/high volume).

Environment:
  AS_OF_DATE (default: today)
  FORECAST_WEEKS (default: 12)
  SAMPLE_SIZE (default: 50)
  MIN_CUSTOMERS (default: 2)
  MIN_ORDERS (default: 3)

Usage:
  python scripts/forecasting/backtest_point_accuracy.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from api.db_pool import get_connection, close_pool
from scripts.forecasting import ForecastEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AS_OF_DATE = os.getenv('AS_OF_DATE') or datetime.now().strftime('%Y-%m-%d')
FORECAST_WEEKS = int(os.getenv('FORECAST_WEEKS', '12'))
SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', '50'))
MIN_CUSTOMERS = int(os.getenv('FORECAST_MIN_CUSTOMERS', '2'))
MIN_ORDERS = int(os.getenv('FORECAST_MIN_ORDERS', '3'))


def fetch_products(conn, as_of_date: str, limit: int) -> List[int]:
    cursor = conn.cursor()
    query = f"""
    SELECT TOP {limit}
        oi.ProductID,
        COUNT(DISTINCT ca.ClientID) AS customers,
        COUNT(DISTINCT o.ID) AS orders,
        SUM(oi.Qty) AS qty
    FROM dbo.OrderItem oi
    INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
    INNER JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
    WHERE o.Created >= '2019-01-01'
      AND o.Created < %s
      AND oi.ProductID IS NOT NULL
    GROUP BY oi.ProductID
    HAVING COUNT(DISTINCT ca.ClientID) >= %s AND COUNT(DISTINCT o.ID) >= %s
    ORDER BY qty DESC
    """
    cursor.execute(query, (as_of_date, MIN_CUSTOMERS, MIN_ORDERS))
    rows = cursor.fetchmany(limit)
    cursor.close()
    return [r[0] for r in rows]


def fetch_actuals(conn, product_id: int, start_date: str, end_date: str) -> float:
    cursor = conn.cursor()
    query = """
    SELECT SUM(oi.Qty)
    FROM dbo.OrderItem oi
    INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
    WHERE oi.ProductID = %s
      AND o.Deleted = 0
      AND oi.Deleted = 0
      AND o.Created >= %s
      AND o.Created < %s
    """
    cursor.execute(query, (product_id, start_date, end_date))
    row = cursor.fetchone()
    cursor.close()
    return float(row[0] or 0.0) if row else 0.0


def segment_from_metadata(forecast) -> str:
    return forecast.model_metadata.get('segment') or 'unknown'


def main():
    logger.info("Starting backtest...")
    conn = get_connection()

    try:
        products = fetch_products(conn, AS_OF_DATE, SAMPLE_SIZE)
        if not products:
            logger.error("No products found for backtest")
            return

        logger.info(f"Testing {len(products)} products; as_of_date={AS_OF_DATE}; horizon={FORECAST_WEEKS} weeks")

        engine = ForecastEngine(conn=conn, forecast_weeks=FORECAST_WEEKS)
        horizon_days = FORECAST_WEEKS * 7
        holdout_start = (datetime.fromisoformat(AS_OF_DATE)).strftime('%Y-%m-%d')
        holdout_end = (datetime.fromisoformat(AS_OF_DATE) + timedelta(days=horizon_days)).strftime('%Y-%m-%d')

        results: List[Dict] = []
        for pid in products:
            try:
                fc = engine.generate_forecast(product_id=pid, as_of_date=AS_OF_DATE)
                if not fc:
                    continue
                predicted = fc.summary.get('total_predicted_quantity', 0.0)
                actual = fetch_actuals(conn, pid, holdout_start, holdout_end)
                segment = segment_from_metadata(fc)
                ape = abs(predicted - actual) / actual if actual > 0 else None
                hit = 1 if (ape is not None and ape <= 0.2) else 0
                results.append({
                    'product_id': pid,
                    'predicted': predicted,
                    'actual': actual,
                    'ape': ape,
                    'hit': hit,
                    'segment': segment
                })
            except Exception as e:
                logger.warning(f"Product {pid} failed: {e}")
                continue

        if not results:
            logger.error("No successful forecasts for backtest")
            return

        def agg(filter_fn):
            subset = [r for r in results if filter_fn(r)]
            if not subset:
                return {'count': 0, 'mape': None, 'hit_rate': None}
            apes = [r['ape'] for r in subset if r['ape'] is not None]
            mape = sum(apes) / len(apes) if apes else None
            hit_rate = sum(r['hit'] for r in subset) / len(subset)
            return {'count': len(subset), 'mape': mape, 'hit_rate': hit_rate}

        overall = agg(lambda r: True)
        segs = {
            seg: agg(lambda r, s=seg: r['segment'] == s)
            for seg in set(r['segment'] for r in results)
        }

        logger.info("Backtest summary (hit_rate = pct within ±20%):")
        logger.info(f"Overall: count={overall['count']}, MAPE={overall['mape']}, hit_rate={overall['hit_rate']}")
        for seg, vals in segs.items():
            logger.info(f"  Segment {seg}: count={vals['count']}, MAPE={vals['mape']}, hit_rate={vals['hit_rate']}")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        close_pool()


if __name__ == "__main__":
    main()
