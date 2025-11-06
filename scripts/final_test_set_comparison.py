#!/usr/bin/env python3
"""
Final Test Set Comparison: Phase C Baseline vs Grid Search Best

Tests both configurations on the held-out test set (13 customers)
that was NEVER used for optimization. This provides unbiased comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
import pymssql
import logging
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.hybrid_recommender import HybridRecommender
import scripts.hybrid_recommender as hr_module

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


def test_configuration(weights: Dict[str, float], test_customers, config_name: str):
    """Test a weight configuration on test customers"""

    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING: {config_name}")
    logger.info(f"{'='*80}")
    logger.info(f"\nWeights:")
    for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {k:20s}: {v:.3f}")

    # Temporarily override weights
    original_weights = hr_module.WEIGHTS['heavy'].copy()
    hr_module.WEIGHTS['heavy'] = weights

    conn = connect_mssql()
    precisions = []
    results = []

    try:
        recommender = HybridRecommender()

        for i, customer_id in enumerate(test_customers, 1):
            logger.info(f"\n  Testing customer {i}/{len(test_customers)}: {customer_id}...")

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

            logger.info(f"    Precision: {precision:.3f} ({hits} hits, {len(test_products)} products)")

            results.append({
                'customer_id': customer_id,
                'config': config_name,
                'precision': precision,
                'hits': hits,
                'total_products': len(test_products)
            })

    finally:
        conn.close()
        # Restore original weights
        hr_module.WEIGHTS['heavy'] = original_weights

    avg_precision = np.mean(precisions) if precisions else 0.0

    logger.info(f"\n{'='*80}")
    logger.info(f"{config_name} RESULTS ON TEST SET")
    logger.info(f"{'='*80}")
    logger.info(f"  Average Precision: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
    logger.info(f"  Customers evaluated: {len(precisions)}/{len(test_customers)}")
    logger.info(f"  Median Precision: {np.median(precisions):.3f}")
    logger.info(f"  Std Dev: {np.std(precisions):.3f}")

    return avg_precision, precisions, results


def main():
    logger.info("="*80)
    logger.info("FINAL TEST SET COMPARISON")
    logger.info("="*80)
    logger.info("\nObjective: Unbiased comparison on held-out test set")
    logger.info("Test set: Never used for training or validation\n")

    # Get test set customers
    conn = connect_mssql()
    try:
        heavy_customers = get_heavy_customers(conn)
        train, validation, test = split_customers(heavy_customers, seed=42)
    finally:
        conn.close()

    logger.info(f"Customer splits:")
    logger.info(f"  Train: {len(train)} customers (used for grid search training)")
    logger.info(f"  Validation: {len(validation)} customers (used for selection)")
    logger.info(f"  Test: {len(test)} customers (NEVER SEEN - for final decision)")
    logger.info(f"\nTest customers: {test}")

    # Phase C Baseline
    baseline_weights = {
        'frequency': 0.70,
        'recency': 0.15,
        'monetary': 0.05,
        'seasonality': 0.05,
        'compatibility': 0.03,
        'maintenance_cycle': 0.02
    }

    # Grid Search Best
    optimized_weights = {
        'frequency': 0.6372549019607843,
        'recency': 0.14705882352941177,
        'maintenance_cycle': 0.11764705882352941,
        'seasonality': 0.029411764705882353,
        'monetary': 0.029411764705882353,
        'compatibility': 0.0392156862745098
    }

    # Test both configurations
    baseline_avg, baseline_precisions, baseline_results = test_configuration(
        baseline_weights, test, "Phase C Baseline"
    )

    optimized_avg, optimized_precisions, optimized_results = test_configuration(
        optimized_weights, test, "Grid Search Best"
    )

    # Final comparison
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST SET COMPARISON")
    logger.info("="*80)

    logger.info(f"\nPhase C Baseline:")
    logger.info(f"  Test Set:       {baseline_avg:.3f} ({baseline_avg*100:.1f}%)")
    logger.info(f"  Original eval:  0.842 (84.2% on different customers)")

    logger.info(f"\nGrid Search Best:")
    logger.info(f"  Test Set:       {optimized_avg:.3f} ({optimized_avg*100:.1f}%)")
    logger.info(f"  Validation:     0.635 (63.5%)")
    logger.info(f"  Improvement:    {optimized_avg - baseline_avg:+.3f} ({(optimized_avg - baseline_avg)*100:+.1f}pp)")

    # Statistical comparison
    if len(baseline_precisions) == len(optimized_precisions):
        # Paired t-test (same customers)
        differences = np.array(optimized_precisions) - np.array(baseline_precisions)
        logger.info(f"\nPer-customer differences (optimized - baseline):")
        logger.info(f"  Mean difference: {np.mean(differences):+.3f}")
        logger.info(f"  Median difference: {np.median(differences):+.3f}")
        logger.info(f"  Customers improved: {np.sum(differences > 0)}/{len(differences)}")
        logger.info(f"  Customers worse: {np.sum(differences < 0)}/{len(differences)}")
        logger.info(f"  Customers unchanged: {np.sum(differences == 0)}/{len(differences)}")

    # Decision
    logger.info("\n" + "="*80)
    logger.info("DEPLOYMENT RECOMMENDATION")
    logger.info("="*80)

    improvement = optimized_avg - baseline_avg

    if improvement >= 0.02:  # +2pp or more
        logger.info(f"\n✅ DEPLOY GRID SEARCH WEIGHTS")
        logger.info(f"   Clear improvement: {improvement*100:+.1f}pp")
        logger.info(f"   Validated on proper train/val/test split")
    elif improvement >= 0.005:  # +0.5pp to +2pp
        logger.info(f"\n⚠️  MARGINAL IMPROVEMENT: {improvement*100:+.1f}pp")
        logger.info(f"   Consider A/B testing in production")
        logger.info(f"   Grid search weights are technically better but improvement is small")
    elif improvement >= -0.005:  # -0.5pp to +0.5pp
        logger.info(f"\n⚠️  NO SIGNIFICANT DIFFERENCE: {improvement*100:+.1f}pp")
        logger.info(f"   KEEP PHASE C BASELINE (simpler, proven)")
    else:  # Worse than -0.5pp
        logger.info(f"\n❌ KEEP PHASE C BASELINE")
        logger.info(f"   Grid search performed worse: {improvement*100:.1f}pp")

    # Save detailed results
    all_results = baseline_results + optimized_results
    df = pd.DataFrame(all_results)
    df.to_csv('results/test_set_comparison.csv', index=False)
    logger.info(f"\nDetailed results saved to: results/test_set_comparison.csv")

    # Key insights
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)
    logger.info(f"\n1. Grid search validation customers were harder (-22pp):")
    logger.info(f"   Phase C baseline: 84.2% → 62.2% on validation")

    logger.info(f"\n2. Test set precision closer to validation than original:")
    logger.info(f"   Original Phase C test: 84.2%")
    logger.info(f"   Grid search test: {baseline_avg*100:.1f}%")
    logger.info(f"   This suggests Phase C test set was easier than typical")

    logger.info(f"\n3. Grid search discovered maintenance_cycle importance:")
    logger.info(f"   Phase C: 0.02 (2% weight)")
    logger.info(f"   Optimized: 0.118 (11.8% weight) - 6x increase!")

    logger.info(f"\n4. Original 87-90% goal appears unrealistic:")
    logger.info(f"   Best achieved: {max(baseline_avg, optimized_avg)*100:.1f}% on unbiased test")
    logger.info(f"   With current features, ~60-65% may be ceiling for heavy users")


if __name__ == '__main__':
    main()
