#!/usr/bin/env python3
"""
Comprehensive Verification Testing Script

Tests ensemble performance on 15-20 customers to verify 74% baseline
Includes component analysis and statistical confidence metrics
"""

import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import torch

# Import recommendation systems
from hybrid_recommender import HybridRecommender
from validate_gnn_recommender import (
    load_trained_model,
    get_recommendations_gnn,
    get_test_data,
    get_customer_segment
)
from build_gnn_recommender import load_graph_data
from ensemble_recommender import ensemble_recommend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test customers (same as validation scripts)
TEST_CUSTOMERS = {
    'heavy': [411706, 410376, 410849, 410187, 411539, 411330, 410280, 411726, 410827, 411457],
    'regular': [410280, 411726, 410827, 411457, 410964],
    'light': [411330, 410964, 411457, 410827, 411726]
}


def get_test_customers(n_customers: int = 20) -> List[Tuple[int, str]]:
    """
    Select diverse test customers across all segments

    Returns:
        List of (customer_id, segment) tuples
    """
    selected = []

    # Get unique customers from each segment
    all_heavy = TEST_CUSTOMERS['heavy']
    all_regular = [c for c in TEST_CUSTOMERS['regular'] if c not in all_heavy]
    all_light = [c for c in TEST_CUSTOMERS['light'] if c not in all_heavy and c not in all_regular]

    # Add from each segment
    selected.extend([(c, 'heavy') for c in all_heavy[:10]])
    selected.extend([(c, 'regular') for c in all_regular[:7]])
    selected.extend([(c, 'light') for c in all_light[:3]])

    return selected[:n_customers]


def test_hybrid_only(customer_id: int, train_products: set, test_products: set, top_n: int = 50) -> Dict:
    """Test Hybrid recommender alone"""
    recommender = HybridRecommender()
    recommendations = recommender.get_recommendations(
        customer_id=customer_id,
        top_n=top_n,
        as_of_date=datetime(2024, 6, 30)
    )

    # Convert to product IDs
    rec_ids = set(int(r['product_id']) for r in recommendations)
    test_ids = set(int(p) for p in test_products)

    hits = len(rec_ids & test_ids)
    precision = hits / len(rec_ids) if len(rec_ids) > 0 else 0

    return {
        'customer_id': customer_id,
        'system': 'hybrid',
        'hits': hits,
        'total_recs': len(rec_ids),
        'precision': precision
    }


def test_gnn_only(customer_id: int, train_products: set, test_products: set,
                  model, edge_index_dict, metadata, device, top_n: int = 50) -> Dict:
    """Test GNN recommender alone"""
    recommendations = get_recommendations_gnn(
        model, edge_index_dict, customer_id, metadata,
        train_products, top_n=top_n, device=device
    )

    rec_ids = set(int(r) for r in recommendations)
    test_ids = set(int(p) for p in test_products)

    hits = len(rec_ids & test_ids)
    precision = hits / len(rec_ids) if len(rec_ids) > 0 else 0

    return {
        'customer_id': customer_id,
        'system': 'gnn',
        'hits': hits,
        'total_recs': len(rec_ids),
        'precision': precision
    }


def test_ensemble(customer_id: int, train_products: set, test_products: set, top_n: int = 50) -> Dict:
    """Test Ensemble (70/30 Hybrid-GNN)"""
    recommendations = ensemble_recommend(
        customer_id=customer_id,
        hybrid_weight=0.7,
        gnn_weight=0.3,
        top_n=top_n
    )

    rec_ids = set(int(r) for r in recommendations)
    test_ids = set(int(p) for p in test_products)

    hits = len(rec_ids & test_ids)
    precision = hits / len(rec_ids) if len(rec_ids) > 0 else 0

    return {
        'customer_id': customer_id,
        'system': 'ensemble',
        'hits': hits,
        'total_recs': len(rec_ids),
        'precision': precision
    }


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate statistical metrics"""
    precisions = [r['precision'] for r in results]

    return {
        'mean': np.mean(precisions),
        'std': np.std(precisions),
        'min': np.min(precisions),
        'max': np.max(precisions),
        'median': np.median(precisions),
        'q25': np.percentile(precisions, 25),
        'q75': np.percentile(precisions, 75),
        'count': len(precisions)
    }


def main():
    logger.info("="*80)
    logger.info("COMPREHENSIVE VERIFICATION TESTING")
    logger.info("="*80)

    # Get test customers
    logger.info("\nStep 1: Selecting test customers...")
    test_customers = get_test_customers(n_customers=20)
    logger.info(f"Selected {len(test_customers)} customers:")

    segment_counts = {}
    for cid, segment in test_customers:
        segment_counts[segment] = segment_counts.get(segment, 0) + 1
    for segment, count in segment_counts.items():
        logger.info(f"  {segment}: {count} customers")

    # Load GNN model once
    logger.info("\nStep 2: Loading GNN model...")
    device = torch.device('cpu')
    model, metadata = load_trained_model(device=device)
    edge_index_dict, _ = load_graph_data()

    # Run tests
    logger.info("\nStep 3: Running comprehensive tests...")
    all_results = []

    for i, (customer_id, segment) in enumerate(test_customers, 1):
        logger.info(f"\n[{i}/{len(test_customers)}] Testing customer {customer_id} ({segment})...")

        # Get test data
        train_products, test_products = get_test_data(customer_id)

        if len(test_products) < 10:
            logger.warning(f"  Skipping - insufficient test data ({len(test_products)} products)")
            continue

        logger.info(f"  Train: {len(train_products)} products | Test: {len(test_products)} products")

        # Test each system
        try:
            # Hybrid
            result_hybrid = test_hybrid_only(customer_id, train_products, test_products)
            result_hybrid['segment'] = segment
            all_results.append(result_hybrid)
            logger.info(f"  Hybrid:   {result_hybrid['precision']:.1%} ({result_hybrid['hits']}/{result_hybrid['total_recs']})")

            # GNN
            result_gnn = test_gnn_only(customer_id, train_products, test_products,
                                       model, edge_index_dict, metadata, device)
            result_gnn['segment'] = segment
            all_results.append(result_gnn)
            logger.info(f"  GNN:      {result_gnn['precision']:.1%} ({result_gnn['hits']}/{result_gnn['total_recs']})")

            # Ensemble
            result_ensemble = test_ensemble(customer_id, train_products, test_products)
            result_ensemble['segment'] = segment
            all_results.append(result_ensemble)
            logger.info(f"  Ensemble: {result_ensemble['precision']:.1%} ({result_ensemble['hits']}/{result_ensemble['total_recs']})")

        except Exception as e:
            logger.error(f"  Error testing customer {customer_id}: {e}")
            continue

    # Save detailed results
    df_results = pd.DataFrame(all_results)
    results_path = 'results/verification_detailed_results.csv'
    df_results.to_csv(results_path, index=False)
    logger.info(f"\n✅ Detailed results saved to {results_path}")

    # Calculate statistics per system
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION RESULTS - STATISTICAL ANALYSIS")
    logger.info("="*80)

    for system in ['hybrid', 'gnn', 'ensemble']:
        system_results = [r for r in all_results if r['system'] == system]
        stats = calculate_statistics(system_results)

        logger.info(f"\n{system.upper()} Performance:")
        logger.info(f"  Mean Precision:   {stats['mean']:.1%} ± {stats['std']:.1%}")
        logger.info(f"  Median Precision: {stats['median']:.1%}")
        logger.info(f"  Range:            {stats['min']:.1%} - {stats['max']:.1%}")
        logger.info(f"  IQR:              {stats['q25']:.1%} - {stats['q75']:.1%}")
        logger.info(f"  Sample Size:      {stats['count']} customers")

    # Per-segment analysis for ensemble
    logger.info("\n" + "-"*80)
    logger.info("ENSEMBLE Performance by Segment:")
    logger.info("-"*80)

    ensemble_results = [r for r in all_results if r['system'] == 'ensemble']
    for segment in ['heavy', 'regular', 'light']:
        segment_results = [r for r in ensemble_results if r['segment'] == segment]
        if segment_results:
            stats = calculate_statistics(segment_results)
            logger.info(f"\n{segment.upper()}:")
            logger.info(f"  Mean:   {stats['mean']:.1%} ± {stats['std']:.1%}")
            logger.info(f"  Median: {stats['median']:.1%}")
            logger.info(f"  Range:  {stats['min']:.1%} - {stats['max']:.1%}")
            logger.info(f"  n =     {stats['count']}")

    # Comparison with Phase A
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH PHASE A")
    logger.info("="*80)

    phase_a_precision = 0.744  # From Phase A: 74.4%
    ensemble_stats = calculate_statistics([r for r in all_results if r['system'] == 'ensemble'])

    logger.info(f"\nPhase A (5 customers):     {phase_a_precision:.1%}")
    logger.info(f"Verification ({ensemble_stats['count']} customers): {ensemble_stats['mean']:.1%} ± {ensemble_stats['std']:.1%}")

    diff = ensemble_stats['mean'] - phase_a_precision
    diff_pct = (diff / phase_a_precision) * 100

    logger.info(f"Difference:                {diff:+.1%} ({diff_pct:+.1f}%)")

    # Decision point
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION DECISION")
    logger.info("="*80)

    if abs(diff) <= 0.05:  # Within 5 percentage points
        logger.info("\n✅ VERIFICATION PASSED")
        logger.info(f"Ensemble performance confirmed at {ensemble_stats['mean']:.1%}")
        logger.info("Ready to proceed to Phase C (GNN v2 validation)")
    else:
        logger.info("\n⚠️  VERIFICATION REVIEW NEEDED")
        logger.info(f"Performance differs by {abs(diff):.1%} from Phase A")
        logger.info("Recommend investigation before proceeding")

    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
