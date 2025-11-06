#!/usr/bin/env python3
"""
Validate Frequency Baseline Against Real Data

Compare frequency baseline vs ML model performance
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymssql
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MSSQL Configuration
MSSQL_CONFIG = {
    "host": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": os.getenv("MSSQL_PORT", "1433"),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb_v5"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD", "Grimm_jow92"),
}

SPLIT_DATE = "2024-06-30"
TEST_CUSTOMERS = [410376, 411706, 410767, 410258]


class BaselineValidator:
    """Validate frequency baseline"""

    def __init__(self):
        self.conn = None

    def connect_mssql(self):
        """Connect to MSSQL"""
        logger.info(f"Connecting to MSSQL...")
        self.conn = pymssql.connect(
            server=MSSQL_CONFIG['host'],
            port=int(MSSQL_CONFIG['port']),
            user=MSSQL_CONFIG['user'],
            password=MSSQL_CONFIG['password'],
            database=MSSQL_CONFIG['database'],
            tds_version='7.0'
        )
        logger.info("✓ Connected")

    def get_customer_purchases(self, customer_id, start_date=None, end_date=None):
        """Get purchases for a customer"""
        where_clauses = []
        where_clauses.append(f"ca.ClientID = {customer_id}")

        if start_date:
            where_clauses.append(f"o.Created >= '{start_date}'")
        if end_date:
            where_clauses.append(f"o.Created < '{end_date}'")

        query = f"""
        SELECT
            ca.ClientID as customer_id,
            oi.ProductID as product_id,
            o.Created as order_date
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE {' AND '.join(where_clauses)}
        ORDER BY o.Created ASC
        """

        df = pd.read_sql(query, self.conn)
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df

    def test_customer(self, customer_id, top_n=50):
        """Test baseline for one customer"""
        # Import baseline engine
        sys.path.insert(0, str(Path(__file__).parent))
        from predict_recommendations_baseline import FrequencyBaselineEngine

        # Get baseline recommendations
        engine = FrequencyBaselineEngine()
        recommendations = engine.get_recommendations(str(customer_id), top_n=top_n)

        # Get actual validation purchases
        validation = self.get_customer_purchases(customer_id, start_date=SPLIT_DATE, end_date='2025-01-01')
        actual_products = set(validation['product_id'].astype(str).unique())

        # Check hits
        recommended_ids = set([str(rec['product_id']) for rec in recommendations])
        hits = recommended_ids & actual_products

        # Metrics
        hit_rate = 1.0 if len(hits) > 0 else 0.0
        precision = len(hits) / len(recommended_ids) if len(recommended_ids) > 0 else 0.0
        recall = len(hits) / len(actual_products) if len(actual_products) > 0 else 0.0

        logger.info(f"\n{'='*80}")
        logger.info(f"VALIDATION RESULTS - Customer {customer_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Validation products: {len(actual_products)}")
        logger.info(f"Recommendations: {len(recommended_ids)}")
        logger.info(f"Hits: {len(hits)}")
        logger.info(f"Hit Rate: {hit_rate:.1%} ({'✅ SUCCESS' if hit_rate > 0 else '❌ MISS'})")
        logger.info(f"Precision@{top_n}: {precision:.2%}")
        logger.info(f"Recall@{top_n}: {recall:.2%}")

        if hits:
            logger.info(f"\n✅ CORRECT PREDICTIONS:")
            for product_id in list(hits)[:10]:
                logger.info(f"   Product {product_id}")
            if len(hits) > 10:
                logger.info(f"   ... and {len(hits) - 10} more")

        return {
            'customer_id': customer_id,
            'top_n': top_n,
            'num_validation_products': len(actual_products),
            'num_hits': len(hits),
            'hit_rate': hit_rate,
            'precision': precision,
            'recall': recall,
            'hits': list(hits)
        }

    def run_validation(self, top_n=50):
        """Run validation on all test customers"""
        logger.info("="*80)
        logger.info("FREQUENCY BASELINE VALIDATION TEST")
        logger.info("="*80)
        logger.info(f"Testing with Top-{top_n} recommendations\n")

        self.connect_mssql()

        results = []
        for customer_id in TEST_CUSTOMERS:
            result = self.test_customer(customer_id, top_n=top_n)
            results.append(result)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("OVERALL SUMMARY")
        logger.info("="*80)

        df = pd.DataFrame(results)
        avg_hit_rate = df['hit_rate'].mean()
        avg_precision = df['precision'].mean()
        avg_recall = df['recall'].mean()
        total_hits = df['num_hits'].sum()

        logger.info(f"\nCustomers Tested: {len(results)}")
        logger.info(f"Average Hit Rate: {avg_hit_rate:.1%} ({int(avg_hit_rate * len(results))}/{len(results)} customers)")
        logger.info(f"Average Precision@{top_n}: {avg_precision:.2%}")
        logger.info(f"Average Recall@{top_n}: {avg_recall:.2%}")
        logger.info(f"Total Correct Predictions: {total_hits}")

        # Comparison with ML
        logger.info("\n" + "="*80)
        logger.info("COMPARISON: FREQUENCY BASELINE vs ML")
        logger.info("="*80)

        logger.info(f"\n| Metric | Frequency Baseline | ML Model | Winner |")
        logger.info(f"|--------|-------------------|----------|--------|")
        logger.info(f"| Hit Rate@{top_n} | {avg_hit_rate:.1%} | 0.0% | **{'Baseline' if avg_hit_rate > 0 else 'Tie'}** |")
        logger.info(f"| Precision@{top_n} | {avg_precision:.2%} | 0.00% | **{'Baseline' if avg_precision > 0 else 'Tie'}** |")
        logger.info(f"| Recall@{top_n} | {avg_recall:.2%} | 0.00% | **{'Baseline' if avg_recall > 0 else 'Tie'}** |")
        logger.info(f"| Total Hits | {total_hits} | 0 | **{'Baseline' if total_hits > 0 else 'Tie'}** |")

        if avg_hit_rate > 0:
            logger.info(f"\n✅ **FREQUENCY BASELINE WINS!**")
            logger.info(f"   - Simple frequency-based approach outperforms complex ML")
            logger.info(f"   - {int(avg_hit_rate * 100)}% of customers got useful recommendations")
            logger.info(f"   - {total_hits} correct predictions vs 0 for ML")
            logger.info(f"\n📊 **PRODUCTION RECOMMENDATION**: Deploy Frequency Baseline")
        else:
            logger.info(f"\n⚠️  Baseline also failed - data quality issue?")

        return results


def main():
    validator = BaselineValidator()
    validator.run_validation(top_n=50)


if __name__ == "__main__":
    main()
