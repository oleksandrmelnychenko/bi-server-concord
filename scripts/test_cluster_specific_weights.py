#!/usr/bin/env python3
"""
Phase G: Test Cluster-Specific Weight Optimization

Quick validation of targeted weight adjustments based on Phase F behavioral insights.

Hypothesis:
- Cluster 2 (Routine): Increase maintenance_cycle (0.118 → 0.20)
- Cluster 1 (Exploratory): Decrease maintenance_cycle (0.118 → 0.05), increase recency

Expected: 56.8% → 59-62% (+2-5pp)
Time: 2-4 hours
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.hybrid_recommender import HybridRecommender
import scripts.hybrid_recommender as hr_module

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Phase E test set customers (for final validation)
PHASE_E_TEST_CUSTOMERS = [
    410237, 410232, 411146, 412255, 410376, 410275, 411390,
    410756, 411317, 416107, 410321, 410521, 411809
]

# Cluster-specific weights (targeted adjustments from Phase F insights)
CLUSTER_WEIGHTS = {
    'cluster_2_routine': {  # Emphasize patterns & cycles
        'frequency': 0.60,
        'recency': 0.12,
        'maintenance_cycle': 0.20,  # ← MAJOR increase from 0.118
        'compatibility': 0.03,
        'seasonality': 0.03,
        'monetary': 0.02
    },
    'cluster_1_exploratory': {  # Reduce patterns, increase freshness
        'frequency': 0.65,
        'recency': 0.18,  # ← Increase from 0.147
        'maintenance_cycle': 0.05,  # ← MAJOR decrease from 0.118
        'compatibility': 0.05,
        'seasonality': 0.04,
        'monetary': 0.03
    },
    'phase_e_baseline': {  # Current production weights
        'frequency': 0.637,
        'recency': 0.147,
        'maintenance_cycle': 0.118,
        'compatibility': 0.039,
        'seasonality': 0.029,
        'monetary': 0.029
    }
}


def load_cluster_assignments():
    """Load customer cluster assignments from Phase F"""
    clusters = pd.read_csv('results/customer_clusters.csv')
    return dict(zip(clusters['customer_id'], clusters['cluster']))


def get_customer_cluster_config(customer_id: int, cluster_map: dict) -> str:
    """Determine which weight config to use for a customer"""
    cluster = cluster_map.get(customer_id)

    if cluster == 1:
        return 'cluster_1_exploratory'
    elif cluster == 2:
        return 'cluster_2_routine'
    else:
        # Clusters 0 and 3 are small outliers, use baseline
        return 'phase_e_baseline'


def test_weight_configuration(config_name: str, weights: dict, test_customers: list, cluster_map: dict):
    """Test a weight configuration on test customers"""

    logger.info(f"\n{'='*70}")
    logger.info(f"Testing Configuration: {config_name}")
    logger.info(f"{'='*70}")

    # Show weights
    logger.info("\nWeights:")
    for feature, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature:20s}: {weight:.3f}")
    logger.info(f"  {'Total':20s}: {sum(weights.values()):.3f}")

    # Temporarily override weights
    original_weights = hr_module.WEIGHTS['heavy'].copy()
    hr_module.WEIGHTS['heavy'] = weights

    results = []

    try:
        for customer_id in test_customers:
            recommender = None
            try:
                recommender = HybridRecommender()

                # Get recommendations as of July 1, 2024
                recs = recommender.get_recommendations(
                    customer_id=customer_id,
                    top_n=50,
                    as_of_date=datetime(2024, 7, 1)
                )

                # Get actual products purchased after July 1
                # FIXED: Match Phase E test period (open-ended, not 3-month window)
                ground_truth_query = f"""
                SELECT DISTINCT CAST(oi.ProductID AS VARCHAR(50)) as product_id
                FROM dbo.ClientAgreement ca
                INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
                INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
                WHERE ca.ClientID = {customer_id}
                    AND o.Created >= '2024-07-01'
                    AND oi.ProductID IS NOT NULL
                """

                ground_truth = pd.read_sql(ground_truth_query, recommender.conn)

                if len(ground_truth) == 0:
                    logger.warning(f"Customer {customer_id} has no test products")
                    continue

                # Calculate precision
                recommended_ids = set(str(r['product_id']) for r in recs)
                actual_ids = set(ground_truth['product_id'].values)
                hits = recommended_ids.intersection(actual_ids)

                precision = len(hits) / 50  # Precision@50

                cluster = cluster_map.get(customer_id, -1)
                cluster_name = 'Cluster 1 (Exploratory)' if cluster == 1 else \
                              'Cluster 2 (Routine)' if cluster == 2 else \
                              f'Cluster {cluster}'

                results.append({
                    'customer_id': customer_id,
                    'cluster': cluster,
                    'cluster_name': cluster_name,
                    'precision': precision,
                    'hits': len(hits),
                    'total_test_products': len(actual_ids)
                })

                logger.info(f"  Customer {customer_id} ({cluster_name}): "
                          f"{precision:.1%} ({len(hits)}/50 hits, {len(actual_ids)} test products)")

            finally:
                if recommender and hasattr(recommender, 'conn'):
                    recommender.conn.close()

    finally:
        # Restore original weights
        hr_module.WEIGHTS['heavy'] = original_weights

    # Calculate summary statistics
    if results:
        df = pd.DataFrame(results)

        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: {config_name}")
        logger.info(f"{'='*70}")
        logger.info(f"\nOverall Average Precision@50: {df['precision'].mean():.1%}")

        # By cluster
        logger.info(f"\nBy Cluster:")
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            cluster_name = cluster_data.iloc[0]['cluster_name']
            avg_prec = cluster_data['precision'].mean()
            count = len(cluster_data)
            logger.info(f"  {cluster_name}: {avg_prec:.1%} (n={count})")

        return df

    return pd.DataFrame()


def main():
    logger.info("="*70)
    logger.info("PHASE G: CLUSTER-SPECIFIC WEIGHT VALIDATION")
    logger.info("="*70)
    logger.info("\nGoal: Test targeted weight adjustments based on Phase F insights")
    logger.info("Expected: 56.8% → 59-62% (+2-5pp)\n")

    # Load cluster assignments
    logger.info("Loading cluster assignments from Phase F...")
    cluster_map = load_cluster_assignments()
    logger.info(f"Loaded {len(cluster_map)} customer cluster assignments\n")

    # Show test set composition
    logger.info("Phase E Test Set Composition:")
    test_clusters = {}
    for cust_id in PHASE_E_TEST_CUSTOMERS:
        cluster = cluster_map.get(cust_id, -1)
        test_clusters[cluster] = test_clusters.get(cluster, 0) + 1

    for cluster, count in sorted(test_clusters.items()):
        cluster_name = 'Exploratory' if cluster == 1 else 'Routine' if cluster == 2 else f'Cluster {cluster}'
        logger.info(f"  Cluster {cluster} ({cluster_name}): {count} customers")

    # Test 1: Baseline (Phase E weights)
    logger.info("\n" + "="*70)
    logger.info("TEST 1: BASELINE (Phase E Optimized Weights)")
    logger.info("="*70)
    baseline_results = test_weight_configuration(
        'Phase E Baseline',
        CLUSTER_WEIGHTS['phase_e_baseline'],
        PHASE_E_TEST_CUSTOMERS,
        cluster_map
    )

    # Test 2: Cluster-specific weights (adaptive)
    logger.info("\n" + "="*70)
    logger.info("TEST 2: CLUSTER-SPECIFIC WEIGHTS (Adaptive)")
    logger.info("="*70)
    logger.info("\nApproach: Use Cluster 2 weights for routine buyers, Cluster 1 weights for exploratory\n")

    # For cluster-specific approach, we need to test each customer with their own weights
    cluster_specific_results = []

    for customer_id in PHASE_E_TEST_CUSTOMERS:
        cluster = cluster_map.get(customer_id, -1)

        if cluster == 1:
            config = 'cluster_1_exploratory'
            weights = CLUSTER_WEIGHTS['cluster_1_exploratory']
        elif cluster == 2:
            config = 'cluster_2_routine'
            weights = CLUSTER_WEIGHTS['cluster_2_routine']
        else:
            config = 'phase_e_baseline'
            weights = CLUSTER_WEIGHTS['phase_e_baseline']

        # Test this one customer with their cluster-specific weights
        result_df = test_weight_configuration(
            f"{config} (Customer {customer_id})",
            weights,
            [customer_id],
            cluster_map
        )

        if len(result_df) > 0:
            cluster_specific_results.append(result_df)

    # Combine results
    if cluster_specific_results:
        cluster_specific_df = pd.concat(cluster_specific_results, ignore_index=True)

        logger.info(f"\n{'='*70}")
        logger.info("CLUSTER-SPECIFIC AGGREGATE RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"\nOverall Average Precision@50: {cluster_specific_df['precision'].mean():.1%}")

        logger.info(f"\nBy Cluster:")
        for cluster in sorted(cluster_specific_df['cluster'].unique()):
            cluster_data = cluster_specific_df[cluster_specific_df['cluster'] == cluster]
            cluster_name = cluster_data.iloc[0]['cluster_name']
            avg_prec = cluster_data['precision'].mean()
            count = len(cluster_data)
            logger.info(f"  {cluster_name}: {avg_prec:.1%} (n={count})")

    # Final comparison
    logger.info(f"\n{'='*70}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'='*70}")

    if len(baseline_results) > 0 and len(cluster_specific_df) > 0:
        baseline_avg = baseline_results['precision'].mean()
        cluster_avg = cluster_specific_df['precision'].mean()
        improvement = cluster_avg - baseline_avg

        logger.info(f"\nBaseline (Phase E): {baseline_avg:.1%}")
        logger.info(f"Cluster-Specific:   {cluster_avg:.1%}")
        logger.info(f"Improvement:        {improvement:+.1%} ({improvement*100:+.1f}pp)")

        if improvement >= 0.03:
            logger.info(f"\n✅ SUCCESS: Improvement >= 3pp - Ready for deployment!")
        elif improvement >= 0.01:
            logger.info(f"\n⚠️  MARGINAL: Improvement 1-3pp - Consider further tuning")
        else:
            logger.info(f"\n❌ NO IMPROVEMENT: Cluster-specific weights don't help")

        # Save results
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        comparison_df = pd.DataFrame({
            'customer_id': baseline_results['customer_id'],
            'cluster': baseline_results['cluster'],
            'baseline_precision': baseline_results['precision'],
            'cluster_specific_precision': cluster_specific_df['precision'],
            'improvement': cluster_specific_df['precision'] - baseline_results['precision']
        })

        comparison_df.to_csv(f'{results_dir}/phase_g_cluster_weights_comparison.csv', index=False)
        logger.info(f"\n💾 Results saved to: {results_dir}/phase_g_cluster_weights_comparison.csv")


if __name__ == '__main__':
    main()
