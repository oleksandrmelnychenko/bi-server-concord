#!/usr/bin/env python3
"""
Validate Hybrid Recommender System

Tests segment-specific hybrid approach on 20 customers
Compares with previous results:
- Frequency Baseline: 42.5% precision
- ALS: 22.7% precision
- Neural Network: 2.6% precision

Target: 85-92% precision for predictable customers
"""

import os
import sys
import logging
import pandas as pd
import pymssql
from datetime import datetime
from typing import List, Dict

# Add scripts to path
sys.path.append('scripts')
from hybrid_recommender import HybridRecommender

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants
SPLIT_DATE = '2024-06-30'
VALIDATION_START = '2024-06-30'
VALIDATION_END = '2025-01-01'
TOP_N = 50


def connect_mssql():
    """Connect to MSSQL"""
    return pymssql.connect(
        server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
        port=int(os.environ.get('MSSQL_PORT', '1433')),
        database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
        user=os.environ.get('MSSQL_USER', 'ef_migrator'),
        password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
    )


def get_validation_purchases(customer_id: int, conn) -> set:
    """Get actual purchases in H2 2024"""
    query = f"""
    SELECT DISTINCT CAST(oi.ProductID AS VARCHAR(50)) as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND o.Created >= '{VALIDATION_START}'
        AND o.Created < '{VALIDATION_END}'
        AND o.Created IS NOT NULL
        AND oi.ProductID IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    return set(df['product_id'].astype(str).tolist())


def validate_hybrid_system(test_customers: List[int], top_n: int = 50):
    """Validate hybrid system on test customers"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING HYBRID RECOMMENDATION SYSTEM")
    logger.info("="*80)

    recommender = HybridRecommender()
    conn = connect_mssql()

    results_by_segment = {'heavy': [], 'regular': [], 'light': []}
    all_results = []

    for i, customer_id in enumerate(test_customers, 1):
        logger.info(f"\nTesting customer {i}/{len(test_customers)}: {customer_id}")

        # Get recommendations
        recommendations = recommender.get_recommendations(
            customer_id=customer_id,
            top_n=top_n,
            as_of_date=datetime(2024, 6, 30)
        )

        if not recommendations:
            logger.info("  ⚠️ No recommendations generated")
            continue

        segment = recommendations[0]['segment']

        # Get actual purchases
        actual_purchases = get_validation_purchases(customer_id, conn)

        # Calculate metrics
        recommended_products = set([r['product_id'] for r in recommendations])
        hits = recommended_products & actual_purchases

        precision = len(hits) / len(recommended_products) if recommended_products else 0
        recall = len(hits) / len(actual_purchases) if actual_purchases else 0
        hit_rate = 1.0 if len(hits) > 0 else 0.0

        logger.info(f"  Segment: {segment.upper()}")
        logger.info(f"  Precision: {precision:.1%} ({len(hits)}/{len(recommended_products)})")
        logger.info(f"  Actual purchases in H2 2024: {len(actual_purchases)}")

        result = {
            'customer_id': customer_id,
            'segment': segment,
            'precision': precision,
            'recall': recall,
            'hit_rate': hit_rate,
            'num_hits': len(hits),
            'num_recommended': len(recommended_products),
            'num_actual': len(actual_purchases)
        }

        all_results.append(result)
        results_by_segment[segment].append(result)

    conn.close()

    # Generate comprehensive report
    if not all_results:
        logger.error("❌ NO RESULTS")
        return None

    df_all = pd.DataFrame(all_results)

    logger.info("\n" + "="*80)
    logger.info("HYBRID SYSTEM RESULTS")
    logger.info("="*80)

    # Overall metrics
    avg_precision = df_all['precision'].mean()
    avg_recall = df_all['recall'].mean()
    hit_rate = df_all['hit_rate'].mean()

    logger.info(f"\n📊 OVERALL PERFORMANCE")
    logger.info(f"  Customers tested: {len(df_all)}")
    logger.info(f"  Hit rate: {hit_rate:.1%} ({int(hit_rate * len(df_all))}/{len(df_all)} customers)")
    logger.info(f"  Average Precision@{top_n}: {avg_precision:.1%}")
    logger.info(f"  Average Recall@{top_n}: {avg_recall:.1%}")

    # Segment-specific performance
    logger.info(f"\n📊 PERFORMANCE BY SEGMENT")
    for segment in ['heavy', 'regular', 'light']:
        if results_by_segment[segment]:
            df_segment = pd.DataFrame(results_by_segment[segment])
            seg_precision = df_segment['precision'].mean()
            seg_hit_rate = df_segment['hit_rate'].mean()
            seg_count = len(df_segment)

            status = "✅" if seg_precision >= 0.85 else "⚠️" if seg_precision >= 0.60 else "❌"

            logger.info(f"\n  {segment.upper()} Users ({seg_count} customers):")
            logger.info(f"    Precision: {seg_precision:.1%} {status}")
            logger.info(f"    Hit Rate: {seg_hit_rate:.1%}")

    # Top performers
    logger.info(f"\n🏆 TOP 5 PERFORMERS")
    top_5 = df_all.nlargest(5, 'precision')[['customer_id', 'segment', 'precision', 'num_hits']]
    for idx, row in top_5.iterrows():
        logger.info(f"  Customer {int(row['customer_id'])} ({row['segment']}): {row['precision']:.1%} ({int(row['num_hits'])}/{top_n})")

    # Bottom 5
    logger.info(f"\n⚠️  BOTTOM 5 PERFORMERS")
    bottom_5 = df_all.nsmallest(5, 'precision')[['customer_id', 'segment', 'precision', 'num_hits']]
    for idx, row in bottom_5.iterrows():
        logger.info(f"  Customer {int(row['customer_id'])} ({row['segment']}): {row['precision']:.1%} ({int(row['num_hits'])}/{top_n})")

    # Comparison with previous systems
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL COMPARISON - ALL SYSTEMS")
    logger.info("="*80)

    logger.info("\n| Model | Precision@50 | Hit Rate | Best Customer | Status |")
    logger.info("|-------|-------------|----------|---------------|--------|")

    # Previous results
    freq_precision = 0.425
    freq_hit_rate = 1.00
    als_precision = 0.227
    als_hit_rate = 0.95
    neural_precision = 0.026
    neural_hit_rate = 0.40

    # Sort systems by precision
    systems = [
        ("Hybrid System", avg_precision, hit_rate),
        ("Frequency Baseline", freq_precision, freq_hit_rate),
        ("ALS Collaborative", als_precision, als_hit_rate),
        ("Neural Network", neural_precision, neural_hit_rate)
    ]
    systems.sort(key=lambda x: x[1], reverse=True)

    for i, (name, prec, hr) in enumerate(systems):
        winner = "🏆" if i == 0 else "✅" if prec >= 0.60 else "⚠️" if prec >= 0.40 else "❌"
        logger.info(f"| {name:20s} | {prec:7.1%} | {hr:7.1%} | - | {winner} |")

    # Determine if we met our goal
    logger.info("\n" + "="*80)
    logger.info("🎯 GOAL ASSESSMENT: 90% PRECISION TARGET")
    logger.info("="*80)

    if avg_precision >= 0.90:
        logger.info(f"\n✅ SUCCESS! Achieved {avg_precision:.1%} precision (target: 90%)")
    elif avg_precision >= 0.85:
        logger.info(f"\n⚠️  CLOSE! Achieved {avg_precision:.1%} precision (target: 90%)")
        logger.info(f"   Gap: {(0.90 - avg_precision):.1%}")
        logger.info(f"   Recommendation: Add Phase 3 advanced ML techniques")
    elif avg_precision >= 0.70:
        logger.info(f"\n⚠️  MODERATE! Achieved {avg_precision:.1%} precision (target: 90%)")
        logger.info(f"   Gap: {(0.90 - avg_precision):.1%}")
        logger.info(f"   Recommendation: Needs Phase 3 OR redefine success criteria")
    else:
        logger.info(f"\n❌ BELOW TARGET! Achieved {avg_precision:.1%} precision (target: 90%)")
        logger.info(f"   Gap: {(0.90 - avg_precision):.1%}")
        logger.info(f"   Recommendation: Consider redefining success criteria")

    # Check segment-specific achievement
    if results_by_segment['heavy']:
        heavy_precision = pd.DataFrame(results_by_segment['heavy'])['precision'].mean()
        logger.info(f"\n📊 HEAVY USER SEGMENT:")
        logger.info(f"   Precision: {heavy_precision:.1%}")
        if heavy_precision >= 0.90:
            logger.info(f"   ✅ Heavy users meet 90% target!")

    if results_by_segment['regular']:
        regular_precision = pd.DataFrame(results_by_segment['regular'])['precision'].mean()
        logger.info(f"\n📊 REGULAR USER SEGMENT:")
        logger.info(f"   Precision: {regular_precision:.1%}")
        if regular_precision >= 0.80:
            logger.info(f"   ✅ Regular users at good level (80%+)")

    return df_all


def main():
    """Main validation pipeline"""
    print("="*80)
    print("HYBRID RECOMMENDATION SYSTEM - VALIDATION")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print(f"Validation period: {VALIDATION_START} to {VALIDATION_END}")
    print()

    # Same 20 customers as previous tests
    test_customers = [
        410376, 410756, 410849, 411539, 410282,  # Regular/Mixed
        410280, 411211, 410827, 411726, 410187,  # Mixed/Heavy
        411390, 410380, 411706, 411600, 410916,  # Heavy/Light
        410338, 410235, 411551, 410744, 410381   # Light
    ]

    logger.info(f"Test set: {len(test_customers)} customers")

    results = validate_hybrid_system(test_customers, top_n=TOP_N)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
