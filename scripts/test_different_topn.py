#!/usr/bin/env python3
"""
Test Different Top-N Values

Test validation with 50, 100, 200 recommendations to find optimal count for B2B
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


class TopNTester:
    """Test different top-N values"""

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
            o.Created as order_date,
            o.ID as order_id
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE {' AND '.join(where_clauses)}
        ORDER BY o.Created ASC
        """

        df = pd.read_sql(query, self.conn)
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df

    def test_customer_with_topn(self, customer_id, top_n):
        """Test one customer with specific top-N"""
        # Import recommendation engine
        sys.path.insert(0, str(Path(__file__).parent))
        from predict_recommendations import RecommendationEngine

        # Get ML recommendations
        engine = RecommendationEngine()
        engine.load_models()
        recommendations = engine.get_recommendations(str(customer_id), top_n=top_n)

        # Get actual purchases
        validation = self.get_customer_purchases(customer_id, start_date=SPLIT_DATE, end_date='2025-01-01')
        actual_products = set(validation['product_id'].astype(str).unique())

        # Check overlap
        recommended_ids = set([str(rec['product_id']) for rec in recommendations])
        hits = recommended_ids & actual_products

        # Calculate metrics
        hit_rate = 1.0 if len(hits) > 0 else 0.0
        precision = len(hits) / len(recommended_ids) if len(recommended_ids) > 0 else 0.0
        recall = len(hits) / len(actual_products) if len(actual_products) > 0 else 0.0

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

    def run_topn_experiment(self):
        """Test different top-N values"""
        logger.info("="*80)
        logger.info("TESTING DIFFERENT TOP-N VALUES")
        logger.info("="*80)

        self.connect_mssql()

        # Test different top-N values
        topn_values = [20, 50, 100, 200]

        results = []
        for customer_id in TEST_CUSTOMERS:
            logger.info(f"\n{'='*80}")
            logger.info(f"Customer {customer_id}")
            logger.info(f"{'='*80}")

            for top_n in topn_values:
                logger.info(f"\nTesting Top-{top_n}...")
                result = self.test_customer_with_topn(customer_id, top_n)
                results.append(result)

                logger.info(f"  Hits: {result['num_hits']}/{top_n} recommendations")
                logger.info(f"  Hit Rate: {result['hit_rate']:.1%}")
                logger.info(f"  Precision: {result['precision']:.1%}")
                logger.info(f"  Recall: {result['recall']:.1%}")

                if result['hits']:
                    logger.info(f"  ✅ HIT! Products: {result['hits']}")

        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY BY TOP-N")
        logger.info("="*80)

        df = pd.DataFrame(results)

        for top_n in topn_values:
            topn_results = df[df['top_n'] == top_n]
            avg_hit_rate = topn_results['hit_rate'].mean()
            avg_precision = topn_results['precision'].mean()
            avg_recall = topn_results['recall'].mean()
            total_hits = topn_results['num_hits'].sum()

            logger.info(f"\n📊 Top-{top_n} Results:")
            logger.info(f"   Hit Rate:  {avg_hit_rate:.1%} ({int(avg_hit_rate * len(TEST_CUSTOMERS))}/{len(TEST_CUSTOMERS)} customers)")
            logger.info(f"   Precision: {avg_precision:.2%}")
            logger.info(f"   Recall:    {avg_recall:.2%}")
            logger.info(f"   Total Hits: {total_hits}")

        # Recommendation
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATION")
        logger.info("="*80)

        # Find best top-N (best recall with acceptable precision)
        best_topn = None
        best_recall = 0

        for top_n in topn_values:
            topn_results = df[df['top_n'] == top_n]
            avg_recall = topn_results['recall'].mean()
            avg_precision = topn_results['precision'].mean()

            if avg_recall > best_recall and avg_precision >= 0.01:  # At least 1% precision
                best_recall = avg_recall
                best_topn = top_n

        if best_topn:
            logger.info(f"\n✅ Recommended Top-N: {best_topn}")
            logger.info(f"   This gives {best_recall:.1%} recall on average")
            logger.info(f"\n   For B2B context:")
            logger.info(f"   - Sales reps can review {best_topn} products quickly")
            logger.info(f"   - Covers more of customer's actual needs")
            logger.info(f"   - Better than browsing {df['num_validation_products'].mean():.0f} avg products purchased")
        else:
            logger.info(f"\n⚠️  No Top-N value achieved >0% hit rate")
            logger.info(f"   Model needs fundamental fixes before choosing Top-N")

        return results


def main():
    tester = TopNTester()
    tester.run_topn_experiment()


if __name__ == "__main__":
    main()
