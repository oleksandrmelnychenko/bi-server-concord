#!/usr/bin/env python3
"""
Quick Feature Test: Validate Advanced Features Before Full Optimization

Tests if 6 new advanced features improve precision@50:
- sequence_frequency
- sequence_recency
- days_since_last
- purchase_overdue
- mean_cycle
- basket_frequency

Compares baseline (6 features) vs enhanced (12 features) on 5 test customers.

Decision: If improvement > +1pp, proceed with full optimization. Otherwise pivot.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.hybrid_recommender import HybridRecommender
from scripts.extract_advanced_features import AdvancedFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Test customers: mix of high/medium/low performers from Phase E
TEST_CUSTOMERS = [
    410376,  # High performer (94% baseline)
    410275,  # High performer (78% baseline)
    410232,  # Medium performer (50% baseline)
    411809,  # Low performer (26% baseline)
    411317,  # Low performer (10% baseline)
]


def get_baseline_recommendations(customer_id: int, as_of_date: datetime, top_n: int = 50):
    """Get recommendations using current baseline (6 features)"""

    recommender = HybridRecommender()
    try:
        recs = recommender.get_recommendations(
            customer_id=customer_id,
            top_n=top_n,
            as_of_date=as_of_date
        )
        return recs
    finally:
        recommender.conn.close()


def get_enhanced_recommendations(customer_id: int, as_of_date: datetime, top_n: int = 50):
    """
    Get recommendations with 6 new advanced features integrated

    New features weighted with initial guesses (will optimize later):
    - sequence_frequency: 0.05 (moderate signal)
    - sequence_recency: 0.02 (weak temporal signal)
    - days_since_last: 0.03 (per-product recency)
    - purchase_overdue: 0.02 (urgency flag)
    - mean_cycle: 0.02 (product-specific cycle)
    - basket_frequency: 0.01 (co-purchase signal)

    Total new: 0.15
    Reduce existing by 0.15 proportionally to keep sum = 1.0
    """

    recommender = HybridRecommender()
    extractor = AdvancedFeatureExtractor()

    try:
        # Get baseline recommendations with scores
        recs = recommender.get_recommendations(
            customer_id=customer_id,
            top_n=1000,  # Get more candidates to re-rank
            as_of_date=as_of_date
        )

        if len(recs) == 0:
            return []

        # Extract advanced features
        logger.info(f"  Extracting advanced features for customer {customer_id}...")
        advanced_features = extractor.extract_all_features(customer_id, as_of_date)

        # Re-score products with advanced features
        enhanced_scores = []

        for rec in recs:
            product_id = rec['product_id']
            baseline_score = rec['score']

            # Get advanced features for this product
            adv_feats = advanced_features.get(int(product_id), {})

            # Normalize and score new features
            # Normalize sequence_frequency (0-1000 range)
            seq_freq = min(adv_feats.get('sequence_frequency', 0) / 1000.0, 1.0)

            # Normalize sequence_recency (invert: lower is better, 0-500 range)
            seq_recency_days = adv_feats.get('sequence_recency', 999)
            seq_recency = max(0, 1 - (seq_recency_days / 500.0))

            # Normalize days_since_last (invert: lower is better, 0-400 range)
            days_since = adv_feats.get('days_since_last', 999)
            days_since_score = max(0, 1 - (days_since / 400.0))

            # Purchase overdue (binary)
            overdue = adv_feats.get('purchase_overdue', 0)

            # Normalize mean_cycle (typical B2B: 30-180 days, reward middle range)
            mean_cycle = adv_feats.get('mean_cycle', 999)
            if 30 <= mean_cycle <= 180:
                cycle_score = 1.0
            elif mean_cycle < 30:
                cycle_score = mean_cycle / 30.0
            elif mean_cycle < 999:
                cycle_score = max(0, 1 - ((mean_cycle - 180) / 300.0))
            else:
                cycle_score = 0

            # Normalize basket_frequency (0-10 range)
            basket_freq = min(adv_feats.get('basket_frequency', 0) / 10.0, 1.0)

            # Compute advanced feature contribution (15% of total score)
            advanced_score = (
                0.05 * seq_freq +
                0.02 * seq_recency +
                0.03 * days_since_score +
                0.02 * overdue +
                0.02 * cycle_score +
                0.01 * basket_freq
            )

            # Adjust baseline score down by 15% and add advanced features
            # This keeps total weight = 1.0
            enhanced_score = (baseline_score * 0.85) + advanced_score

            enhanced_scores.append({
                'product_id': product_id,
                'score': enhanced_score,
                'baseline_score': baseline_score,
                'advanced_contribution': advanced_score
            })

        # Sort by enhanced score and take top N
        enhanced_scores.sort(key=lambda x: x['score'], reverse=True)
        top_enhanced = enhanced_scores[:top_n]

        return top_enhanced

    finally:
        recommender.conn.close()
        extractor.conn.close()


def evaluate_recommendations(recs, customer_id: int, as_of_date: datetime):
    """Evaluate recommendations against ground truth"""

    from scripts.hybrid_recommender import HybridRecommender

    recommender = HybridRecommender()
    try:
        # Get ground truth: products purchased after as_of_date
        ground_truth_query = f"""
        SELECT DISTINCT CAST(oi.ProductID AS VARCHAR(50)) as product_id
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
            AND o.Created >= '{as_of_date.strftime('%Y-%m-%d')}'
            AND oi.ProductID IS NOT NULL
        """

        ground_truth = pd.read_sql(ground_truth_query, recommender.conn)

        if len(ground_truth) == 0:
            return None, 0, 0

        # Calculate precision@50
        recommended_ids = set(str(r['product_id']) for r in recs)
        actual_ids = set(ground_truth['product_id'].values)
        hits = recommended_ids.intersection(actual_ids)

        precision = len(hits) / len(recs)

        return precision, len(hits), len(actual_ids)

    finally:
        recommender.conn.close()


def main():
    logger.info("="*70)
    logger.info("QUICK FEATURE TEST: Advanced Features Validation")
    logger.info("="*70)
    logger.info("\nTesting 6 new features on 5 customers:")
    logger.info("- sequence_frequency, sequence_recency, days_since_last")
    logger.info("- purchase_overdue, mean_cycle, basket_frequency")
    logger.info("\nDecision: If improvement > +1pp, proceed with full optimization\n")

    as_of_date = datetime(2024, 7, 1)
    results = []

    for customer_id in TEST_CUSTOMERS:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing Customer {customer_id}")
        logger.info(f"{'='*70}")

        # Baseline recommendations
        logger.info("  Getting baseline recommendations (6 features)...")
        baseline_recs = get_baseline_recommendations(customer_id, as_of_date)

        baseline_prec, baseline_hits, total_test = evaluate_recommendations(
            baseline_recs, customer_id, as_of_date
        )

        if baseline_prec is None:
            logger.warning(f"  Customer {customer_id} has no test products, skipping")
            continue

        logger.info(f"  Baseline: {baseline_prec:.1%} ({baseline_hits}/50 hits, {total_test} test products)")

        # Enhanced recommendations
        logger.info("  Getting enhanced recommendations (12 features)...")
        enhanced_recs = get_enhanced_recommendations(customer_id, as_of_date)

        enhanced_prec, enhanced_hits, _ = evaluate_recommendations(
            enhanced_recs, customer_id, as_of_date
        )

        logger.info(f"  Enhanced: {enhanced_prec:.1%} ({enhanced_hits}/50 hits, {total_test} test products)")

        improvement = enhanced_prec - baseline_prec
        logger.info(f"  Improvement: {improvement:+.1%} ({improvement*100:+.1f}pp)")

        results.append({
            'customer_id': customer_id,
            'baseline_precision': baseline_prec,
            'enhanced_precision': enhanced_prec,
            'improvement': improvement,
            'baseline_hits': baseline_hits,
            'enhanced_hits': enhanced_hits,
            'total_test_products': total_test
        })

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY RESULTS")
    logger.info(f"{'='*70}")

    if len(results) > 0:
        df = pd.DataFrame(results)

        avg_baseline = df['baseline_precision'].mean()
        avg_enhanced = df['enhanced_precision'].mean()
        avg_improvement = df['improvement'].mean()

        logger.info(f"\nAverage Baseline:  {avg_baseline:.1%}")
        logger.info(f"Average Enhanced:  {avg_enhanced:.1%}")
        logger.info(f"Average Improvement: {avg_improvement:+.1%} ({avg_improvement*100:+.1f}pp)")

        improved_count = sum(1 for imp in df['improvement'] if imp > 0)
        logger.info(f"\nCustomers Improved: {improved_count}/{len(results)}")

        # Per-customer breakdown
        logger.info(f"\n{'Customer':<12} {'Baseline':<10} {'Enhanced':<10} {'Change':<10}")
        logger.info("-" * 45)
        for _, row in df.iterrows():
            logger.info(f"{row['customer_id']:<12} {row['baseline_precision']:>8.1%}  {row['enhanced_precision']:>8.1%}  {row['improvement']:>+8.1%}")

        # Decision
        logger.info(f"\n{'='*70}")
        logger.info("DECISION")
        logger.info(f"{'='*70}")

        if avg_improvement >= 0.01:
            logger.info(f"\n✅ SUCCESS: +{avg_improvement*100:.1f}pp improvement!")
            logger.info("→ RECOMMENDATION: Proceed with full Phase H optimization")
            logger.info("→ Expected final result: 56.8% → 59-62% (+2-5pp)")
            logger.info("→ Time investment: 80-90 hours")
        elif avg_improvement >= 0.005:
            logger.info(f"\n⚠️  MARGINAL: +{avg_improvement*100:.1f}pp improvement")
            logger.info("→ RECOMMENDATION: Advanced features show promise but need weight tuning")
            logger.info("→ Option 1: Run grid search on these 12 features (~40 hours)")
            logger.info("→ Option 2: Try collaborative filtering instead (~20 hours)")
        else:
            logger.info(f"\n❌ NO IMPROVEMENT: {avg_improvement*100:+.1f}pp change")
            logger.info("→ RECOMMENDATION: Advanced features don't help with default weights")
            logger.info("→ Next steps:")
            logger.info("  1. Try collaborative filtering approach")
            logger.info("  2. Accept 56.8% as performance ceiling")
            logger.info("  3. Pivot to diversity-focused recommendations")

        # Save results
        df.to_csv('results/phase_h_quick_feature_test.csv', index=False)
        logger.info(f"\n💾 Results saved to: results/phase_h_quick_feature_test.csv")

    else:
        logger.error("\nNo valid test results - all customers had no test products")


if __name__ == '__main__':
    main()
