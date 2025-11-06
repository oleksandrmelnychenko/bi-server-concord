#!/usr/bin/env python3
"""
Validate Improved Hybrid Recommender V2

Tests improved model on same 50 customers from Phase 1.3
to measure improvement over baseline (28.7%)

Target: 40%+ overall precision
"""

import os
import json
import pymssql
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict
import sys

# Add scripts directory to path
sys.path.append(os.path.dirname(__file__))
from improved_hybrid_recommender import ImprovedHybridRecommender

# Configuration
DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
    'as_dict': True
}

AS_OF_DATE = '2024-07-01'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_validation_products(conn, customer_id: int, as_of_date: str) -> List[int]:
    """Get products purchased after as_of_date"""
    query = f"""
    SELECT DISTINCT oi.ProductID
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
          AND o.Created >= '{as_of_date}'
          AND oi.ProductID IS NOT NULL
    """

    cursor = conn.cursor()
    cursor.execute(query)
    products = [row['ProductID'] for row in cursor]
    cursor.close()

    return products


def calculate_precision(recommendations: List[Dict], validation_products: List[int]) -> Tuple[float, int]:
    """Calculate precision@50"""
    validation_set = set(validation_products)
    hits = sum(1 for rec in recommendations[:50] if rec['product_id'] in validation_set)
    precision = hits / 50 if len(recommendations) >= 50 else hits / len(recommendations)
    return precision, hits


def load_baseline_customers() -> List[int]:
    """Load customers from 50-customer baseline test"""
    try:
        with open('validation_50_customers_results.json', 'r') as f:
            data = json.load(f)
        return [r['customer_id'] for r in data['results']]
    except FileNotFoundError:
        logger.error("validation_50_customers_results.json not found")
        return []


def main():
    logger.info("=" * 80)
    logger.info("IMPROVED HYBRID RECOMMENDER V2 - VALIDATION")
    logger.info("=" * 80)
    logger.info(f"As-of date: {AS_OF_DATE}")
    logger.info(f"Baseline: 28.7% (frequency-only)")
    logger.info(f"Target: 40%+")
    logger.info("")

    # Load customers from baseline
    customers = load_baseline_customers()
    if not customers:
        logger.error("Failed to load customers")
        return

    logger.info(f"Testing on {len(customers)} customers (same as baseline)")
    logger.info("")

    # Initialize recommender
    recommender = ImprovedHybridRecommender()
    conn = pymssql.connect(**DB_CONFIG)

    try:
        results = []
        segment_results = defaultdict(list)
        subsegment_results = defaultdict(list)

        for idx, customer_id in enumerate(customers, 1):
            segment, subsegment = recommender.classify_customer(customer_id, AS_OF_DATE)
            segment_label = f"{segment}_{subsegment}" if subsegment else segment

            logger.info(f"[{idx}/{len(customers)}] Testing customer {customer_id} ({segment_label})...")

            # Get recommendations
            recs = recommender.get_recommendations(customer_id, AS_OF_DATE, top_n=50)

            # Get validation products
            validation_products = get_validation_products(conn, customer_id, AS_OF_DATE)

            # Calculate precision
            precision, hits = calculate_precision(recs, validation_products)

            result = {
                'customer_id': customer_id,
                'segment': segment,
                'subsegment': subsegment,
                'precision': precision,
                'hits': hits,
                'validation_products': len(validation_products)
            }

            results.append(result)
            segment_results[segment].append(precision)
            if subsegment:
                subsegment_results[f"{segment}_{subsegment}"].append(precision)

            logger.info(f"  Precision: {precision:.1%} ({hits}/50 hits)")

        logger.info("")
        logger.info("=" * 80)
        logger.info("IMPROVED MODEL RESULTS")
        logger.info("=" * 80)

        # Overall statistics
        overall_precision = sum(r['precision'] for r in results) / len(results)
        logger.info(f"Overall Precision: {overall_precision:.1%}")
        logger.info("")

        # Segment breakdown
        logger.info("SEGMENT BREAKDOWN:")
        for segment in ['HEAVY', 'REGULAR', 'LIGHT']:
            if segment in segment_results:
                precisions = segment_results[segment]
                avg_precision = sum(precisions) / len(precisions)
                n = len(precisions)
                logger.info(f"  {segment}: {avg_precision:.1%} (n={n})")

        # Sub-segment breakdown for REGULAR
        if 'REGULAR_CONSISTENT' in subsegment_results or 'REGULAR_EXPLORATORY' in subsegment_results:
            logger.info("")
            logger.info("REGULAR USER SUB-SEGMENTS:")
            for subseg in ['REGULAR_CONSISTENT', 'REGULAR_EXPLORATORY']:
                if subseg in subsegment_results:
                    precisions = subsegment_results[subseg]
                    avg_precision = sum(precisions) / len(precisions)
                    n = len(precisions)
                    logger.info(f"  {subseg}: {avg_precision:.1%} (n={n})")

        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPARISON TO BASELINE")
        logger.info("=" * 80)

        # Load baseline results
        try:
            with open('validation_50_customers_results.json', 'r') as f:
                baseline_data = json.load(f)
            baseline_overall = baseline_data['summary']['overall_frequency_only']
            baseline_heavy = baseline_data['segment_summary']['HEAVY']['frequency_only']
            baseline_regular = baseline_data['segment_summary']['REGULAR']['frequency_only']
            baseline_light = baseline_data['segment_summary']['LIGHT']['frequency_only']

            logger.info(f"Overall:    Baseline={baseline_overall:.1%} | Improved={overall_precision:.1%} | Δ={overall_precision - baseline_overall:+.1%}")
            logger.info(f"Heavy:      Baseline={baseline_heavy:.1%} | Improved={sum(segment_results['HEAVY']) / len(segment_results['HEAVY']):.1%} | Δ={(sum(segment_results['HEAVY']) / len(segment_results['HEAVY'])) - baseline_heavy:+.1%}")
            logger.info(f"Regular:    Baseline={baseline_regular:.1%} | Improved={sum(segment_results['REGULAR']) / len(segment_results['REGULAR']):.1%} | Δ={(sum(segment_results['REGULAR']) / len(segment_results['REGULAR'])) - baseline_regular:+.1%}")
            logger.info(f"Light:      Baseline={baseline_light:.1%} | Improved={sum(segment_results['LIGHT']) / len(segment_results['LIGHT']):.1%} | Δ={(sum(segment_results['LIGHT']) / len(segment_results['LIGHT'])) - baseline_light:+.1%}")

            logger.info("")

            # Check if we hit target
            if overall_precision >= 0.40:
                logger.info("🎯 TARGET ACHIEVED: {:.1%} >= 40%".format(overall_precision))
            elif overall_precision >= 0.35:
                logger.info("⚠️  CLOSE TO TARGET: {:.1%} (need 40%)".format(overall_precision))
            else:
                logger.info("❌ BELOW TARGET: {:.1%} (need 40%)".format(overall_precision))
                logger.info(f"   Gap: {0.40 - overall_precision:.1%} ({(0.40 - overall_precision) / 0.40 * 100:.1f}% away)")

        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")

        logger.info("")

        # Save results
        output = {
            'meta': {
                'test_date': datetime.now().isoformat(),
                'as_of_date': AS_OF_DATE,
                'customers_tested': len(customers),
                'model_version': 'improved_v2',
                'strategy': 'segment_specific_weights'
            },
            'results': results,
            'summary': {
                'overall_precision': overall_precision,
                'segment_breakdown': {
                    segment: sum(segment_results[segment]) / len(segment_results[segment])
                    for segment in segment_results.keys()
                },
                'subsegment_breakdown': {
                    subseg: sum(subsegment_results[subseg]) / len(subsegment_results[subseg])
                    for subseg in subsegment_results.keys()
                }
            }
        }

        output_file = 'improved_model_validation_results.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"✓ Results saved to {output_file}")
        logger.info("")

    finally:
        recommender.close()
        conn.close()


if __name__ == '__main__':
    main()
