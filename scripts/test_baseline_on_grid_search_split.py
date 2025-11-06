#!/usr/bin/env python3
"""
Test Phase C Baseline on Grid Search Validation Split

Tests the Phase C baseline weights (84.2% on original test set)
on the SAME validation customers used by grid search.

This provides a fair comparison baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
import pymssql
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.hybrid_recommender import HybridRecommender

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def connect_mssql():
    """Connect to MSSQL database"""
    return pymssql.connect(
        server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
        port=int(os.environ.get('MSSQL_PORT', '1433')),
        database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
        user=os.environ.get('MSSQL_USER', 'ef_migrator'),
        password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
    )


def get_heavy_customers(conn):
    """Get all heavy user customers (500+ unique products)"""
    query = """
    SELECT
        ca.ClientID as customer_id,
        COUNT(DISTINCT oi.ProductID) as unique_products
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE o.Created < '2024-07-01'
        AND o.Created IS NOT NULL
        AND oi.ProductID IS NOT NULL
    GROUP BY ca.ClientID
    HAVING COUNT(DISTINCT oi.ProductID) >= 500
    ORDER BY unique_products DESC
    """

    df = pd.read_sql(query, conn)
    return df['customer_id'].tolist()


def split_customers(customers, seed=42):
    """Split customers into train/validation/test sets (60/20/20)"""
    np.random.seed(seed)
    shuffled = np.random.permutation(customers)

    n = len(shuffled)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train = shuffled[:train_end].tolist()
    validation = shuffled[train_end:val_end].tolist()
    test = shuffled[val_end:].tolist()

    return train, validation, test


def test_baseline_on_validation():
    """Test Phase C baseline on grid search validation split"""
    logger.info("="*80)
    logger.info("TESTING PHASE C BASELINE ON GRID SEARCH VALIDATION SPLIT")
    logger.info("="*80)

    conn = connect_mssql()

    try:
        # Get same customer split as grid search
        heavy_customers = get_heavy_customers(conn)
        train, validation, test = split_customers(heavy_customers, seed=42)

        logger.info(f"\nCustomer splits (same as grid search):")
        logger.info(f"  Train: {len(train)} customers")
        logger.info(f"  Validation: {len(validation)} customers")
        logger.info(f"  Test: {len(test)} customers")

        logger.info(f"\nValidation customers: {validation[:5]}... (showing first 5)")

        # Test Phase C baseline on validation set
        logger.info(f"\nTesting Phase C baseline weights on {len(validation)} validation customers...")
        logger.info("Phase C weights: freq=0.70, rec=0.15, maint=0.02, seas=0.05, mon=0.05, compat=0.03")

        recommender = HybridRecommender()

        precisions = []

        for i, customer_id in enumerate(validation, 1):
            logger.info(f"  Testing customer {i}/{len(validation)}: {customer_id}...")

            # Get recommendations for H1 2024
            recs = recommender.get_recommendations(
                customer_id=customer_id,
                top_n=50,
                as_of_date=datetime(2024, 7, 1)
            )

            if not recs:
                logger.info(f"    No recommendations generated")
                continue

            rec_ids = set(int(r['product_id']) for r in recs)

            # Get H2 2024 actual purchases
            test_query = f"""
            SELECT DISTINCT CAST(oi.ProductID AS VARCHAR(50)) as product_id
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE ca.ClientID = {customer_id}
                AND o.Created >= '2024-07-01'
                AND o.Created IS NOT NULL
                AND oi.ProductID IS NOT NULL
            """

            test_products = set(
                int(p) for p in pd.read_sql(test_query, conn)['product_id'].tolist()
            )

            if len(test_products) == 0:
                logger.info(f"    No test purchases")
                continue

            hits = len(rec_ids & test_products)
            precision = hits / min(50, len(test_products))
            precisions.append(precision)

            logger.info(f"    Precision: {precision:.3f} ({hits}/{len(test_products)} products)")

        # Calculate average precision
        avg_precision = np.mean(precisions) if precisions else 0.0

        logger.info("\n" + "="*80)
        logger.info("RESULTS")
        logger.info("="*80)
        logger.info(f"\nPhase C Baseline on Grid Search Validation Split:")
        logger.info(f"  Average Precision: {avg_precision:.3f}")
        logger.info(f"  Customers evaluated: {len(precisions)}/{len(validation)}")

        logger.info(f"\nComparison:")
        logger.info(f"  Phase C baseline on original test set: 0.842")
        logger.info(f"  Phase C baseline on grid search val:   {avg_precision:.3f}")
        logger.info(f"  Grid search 'best' on val:              0.635")

        if avg_precision > 0.635:
            improvement = avg_precision - 0.635
            logger.info(f"\n✅ Phase C baseline BEATS grid search best by {improvement:.3f} ({improvement*100:.1f}pp)")
            logger.info(f"   Conclusion: Grid search failed to find better weights")
        elif avg_precision >= 0.635 - 0.02:
            logger.info(f"\n⚠️  Phase C baseline similar to grid search best")
            logger.info(f"   Conclusion: Grid search didn't improve over baseline")
        else:
            logger.info(f"\n❌ Grid search 'best' outperforms Phase C baseline")
            logger.info(f"   Conclusion: Grid search found better config")

        logger.info(f"\n⚠️  NOTE: Validation customers appear harder than Phase C test customers")
        logger.info(f"   (84.2% → {avg_precision*100:.1f}% on different customer set)")

    finally:
        conn.close()


if __name__ == '__main__':
    test_baseline_on_validation()
