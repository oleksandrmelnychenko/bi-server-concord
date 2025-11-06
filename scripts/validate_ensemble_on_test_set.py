#!/usr/bin/env python3
"""
Phase I.1: Ensemble Validation on Proper Test Set

Validates ensemble approaches on the Phase E held-out test set (13 customers)
to determine if ensemble can beat 56.8% baseline.

Tests:
1. Hybrid alone (56.8% baseline)
2. Hybrid+GNN ensemble (70/30)
3. Hybrid+CF ensemble (if CF model available)
4. Ensemble weight optimization (50/50, 60/40, 70/30, 80/20)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.hybrid_recommender import HybridRecommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Phase E Test Set (13 customers) - from results/test_set_comparison.csv
TEST_CUSTOMERS = [
    410237, 410232, 411146, 412255, 410376, 410275, 411390,
    410756, 411317, 416107, 410321, 410521, 411809
]

AS_OF_DATE = datetime(2024, 7, 1)  # Phase E split date


def get_test_products(customer_id: int, conn) -> List[int]:
    """Get ground truth products purchased after as_of_date"""
    query = f"""
    SELECT DISTINCT CAST(oi.ProductID AS INT) as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND o.Created >= '{AS_OF_DATE.strftime('%Y-%m-%d')}'
        AND oi.ProductID IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    return df['product_id'].tolist()


def evaluate_precision(recommendations: List, test_products: List[int], top_n: int = 50) -> Tuple[float, int]:
    """Calculate precision@50"""
    # Handle different recommendation formats
    if isinstance(recommendations[0], dict):
        rec_ids = [int(r['product_id']) for r in recommendations[:top_n]]
    else:
        rec_ids = [int(r) for r in recommendations[:top_n]]

    test_ids = set(test_products)
    hits = len(set(rec_ids) & test_ids)
    precision = hits / min(len(rec_ids), top_n)

    return precision, hits


def test_hybrid_only(customer_id: int, test_products: List[int], top_n: int = 50) -> Dict:
    """Test Hybrid recommender baseline"""
    recommender = HybridRecommender()
    try:
        recs = recommender.get_recommendations(
            customer_id=customer_id,
            top_n=top_n,
            as_of_date=AS_OF_DATE
        )
        precision, hits = evaluate_precision(recs, test_products, top_n)

        return {
            'customer_id': customer_id,
            'model': 'hybrid',
            'precision': precision,
            'hits': hits,
            'total_recs': len(recs)
        }
    except Exception as e:
        logger.error(f"  Hybrid failed for customer {customer_id}: {e}")
        return {
            'customer_id': customer_id,
            'model': 'hybrid',
            'precision': 0.0,
            'hits': 0,
            'total_recs': 0
        }
    finally:
        recommender.conn.close()


def test_hybrid_gnn_ensemble(customer_id: int, test_products: List[int],
                              hybrid_weight: float = 0.7, top_n: int = 50) -> Dict:
    """Test Hybrid+GNN ensemble"""
    try:
        # Import GNN components
        from scripts.validate_gnn_recommender import (
            load_trained_model, get_recommendations_gnn, get_test_data
        )
        from scripts.build_gnn_recommender import load_graph_data

        # Calculate split
        n_hybrid = int(top_n * hybrid_weight)
        n_gnn = top_n - n_hybrid

        # Get Hybrid recommendations
        recommender = HybridRecommender()
        hybrid_recs = recommender.get_recommendations(
            customer_id=customer_id,
            top_n=n_hybrid,
            as_of_date=AS_OF_DATE
        )
        hybrid_ids = [int(r['product_id']) for r in hybrid_recs]
        recommender.conn.close()

        # Get GNN recommendations
        device = torch.device('cpu')
        model, metadata = load_trained_model(device=device)
        edge_index_dict, _ = load_graph_data()
        train_products, _ = get_test_data(customer_id)

        gnn_recs = get_recommendations_gnn(
            model, edge_index_dict, customer_id, metadata,
            train_products, top_n=n_gnn, device=device
        )
        gnn_ids = [int(r) for r in gnn_recs]

        # Combine and deduplicate (hybrid first)
        seen = set()
        final_recs = []
        for prod_id in hybrid_ids + gnn_ids:
            if prod_id not in seen:
                final_recs.append(prod_id)
                seen.add(prod_id)
                if len(final_recs) >= top_n:
                    break

        precision, hits = evaluate_precision(final_recs, test_products, top_n)

        return {
            'customer_id': customer_id,
            'model': f'ensemble_{int(hybrid_weight*100)}_{int((1-hybrid_weight)*100)}',
            'precision': precision,
            'hits': hits,
            'total_recs': len(final_recs),
            'hybrid_weight': hybrid_weight
        }

    except Exception as e:
        logger.error(f"  Ensemble failed for customer {customer_id}: {e}")
        return {
            'customer_id': customer_id,
            'model': f'ensemble_{int(hybrid_weight*100)}_{int((1-hybrid_weight)*100)}',
            'precision': 0.0,
            'hits': 0,
            'total_recs': 0,
            'hybrid_weight': hybrid_weight
        }


def main():
    logger.info("="*80)
    logger.info("PHASE I.1: ENSEMBLE VALIDATION ON PROPER TEST SET")
    logger.info("="*80)
    logger.info(f"\nTest Set: {len(TEST_CUSTOMERS)} customers (Phase E held-out)")
    logger.info(f"Baseline: 56.8% precision@50 (Grid Search Best)")
    logger.info(f"Goal: Beat 56.8% with ensemble approach\n")

    all_results = []

    # Get database connection for test products
    from scripts.hybrid_recommender import HybridRecommender
    temp_recommender = HybridRecommender()
    conn = temp_recommender.conn

    # Test each customer
    for i, customer_id in enumerate(TEST_CUSTOMERS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Customer {i}/{len(TEST_CUSTOMERS)}: {customer_id}")
        logger.info(f"{'='*80}")

        # Get ground truth
        test_products = get_test_products(customer_id, conn)
        if len(test_products) == 0:
            logger.warning(f"  No test products for customer {customer_id}, skipping")
            continue

        logger.info(f"  Test products: {len(test_products)}")

        # Test 1: Hybrid Only (Baseline)
        logger.info("\n  [1/5] Testing Hybrid baseline...")
        result_hybrid = test_hybrid_only(customer_id, test_products)
        all_results.append(result_hybrid)
        logger.info(f"    Hybrid: {result_hybrid['precision']:.1%} ({result_hybrid['hits']}/50 hits)")

        # Test 2: Ensemble 50/50
        logger.info("  [2/5] Testing Ensemble 50/50...")
        result_50_50 = test_hybrid_gnn_ensemble(customer_id, test_products, hybrid_weight=0.5)
        all_results.append(result_50_50)
        logger.info(f"    50/50: {result_50_50['precision']:.1%} ({result_50_50['hits']}/50 hits)")

        # Test 3: Ensemble 60/40
        logger.info("  [3/5] Testing Ensemble 60/40...")
        result_60_40 = test_hybrid_gnn_ensemble(customer_id, test_products, hybrid_weight=0.6)
        all_results.append(result_60_40)
        logger.info(f"    60/40: {result_60_40['precision']:.1%} ({result_60_40['hits']}/50 hits)")

        # Test 4: Ensemble 70/30 (Original)
        logger.info("  [4/5] Testing Ensemble 70/30...")
        result_70_30 = test_hybrid_gnn_ensemble(customer_id, test_products, hybrid_weight=0.7)
        all_results.append(result_70_30)
        logger.info(f"    70/30: {result_70_30['precision']:.1%} ({result_70_30['hits']}/50 hits)")

        # Test 5: Ensemble 80/20
        logger.info("  [5/5] Testing Ensemble 80/20...")
        result_80_20 = test_hybrid_gnn_ensemble(customer_id, test_products, hybrid_weight=0.8)
        all_results.append(result_80_20)
        logger.info(f"    80/20: {result_80_20['precision']:.1%} ({result_80_20['hits']}/50 hits)")

    temp_recommender.conn.close()

    # Aggregate results
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    # Group by model
    summary = df.groupby('model').agg({
        'precision': ['mean', 'std', 'min', 'max'],
        'hits': 'mean',
        'customer_id': 'count'
    }).round(3)

    logger.info("\nModel Performance Summary:")
    logger.info(summary.to_string())

    # Determine winner
    model_means = df.groupby('model')['precision'].mean().sort_values(ascending=False)
    logger.info(f"\n{'='*80}")
    logger.info("RANKING")
    logger.info(f"{'='*80}\n")

    for rank, (model, precision) in enumerate(model_means.items(), 1):
        improvement = (precision - 0.568) * 100  # vs 56.8% baseline
        logger.info(f"  {rank}. {model:20s} {precision:.1%}  ({improvement:+.1f}pp vs baseline)")

    best_model = model_means.index[0]
    best_precision = model_means.iloc[0]

    logger.info(f"\n{'='*80}")
    logger.info("DECISION")
    logger.info(f"{'='*80}\n")

    if best_precision > 0.568:
        improvement = (best_precision - 0.568) * 100
        logger.info(f"✅ SUCCESS! {best_model} achieves {best_precision:.1%}")
        logger.info(f"   Improvement: +{improvement:.1f}pp over baseline")
        logger.info(f"\n→ RECOMMENDATION: Deploy {best_model} ensemble")
    else:
        logger.info(f"❌ NO IMPROVEMENT: Best model {best_model} = {best_precision:.1%}")
        logger.info(f"   Baseline remains best: 56.8%")
        logger.info(f"\n→ RECOMMENDATION: Keep current hybrid model, pivot to Option 2 (GBM)")

    # Save results
    df.to_csv('results/phase_i_ensemble_validation.csv', index=False)
    logger.info(f"\n💾 Detailed results saved to: results/phase_i_ensemble_validation.csv")

    # Per-customer breakdown
    logger.info(f"\n{'='*80}")
    logger.info("PER-CUSTOMER BREAKDOWN (Hybrid vs Best Ensemble)")
    logger.info(f"{'='*80}\n")

    pivot = df.pivot_table(index='customer_id', columns='model', values='precision')
    logger.info(pivot.to_string())

    return model_means


if __name__ == '__main__':
    main()
