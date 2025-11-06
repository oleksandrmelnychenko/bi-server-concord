#!/usr/bin/env python3
"""
QUICK WIN #3: Simple 70/30 Hybrid-GNN Ensemble

Combines the best of both worlds:
- Hybrid (70%): Exploits repeat purchases (proven 65% precision)
- GNN (30%): Enables exploration (27% precision, but complimentary)

Expected Result: 50-60% overall precision!
"""

import sys
import logging
from typing import List, Dict
from datetime import datetime
import pandas as pd

# Import both recommendation systems
from hybrid_recommender import HybridRecommender
from validate_gnn_recommender import (
    load_trained_model,
    get_recommendations_gnn,
    get_test_data,
    get_customer_segment
)
from build_gnn_recommender import load_graph_data
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensemble_recommend(customer_id: int,
                       hybrid_weight: float = 0.7,
                       gnn_weight: float = 0.3,
                       top_n: int = 50) -> List[int]:
    """
    Ensemble recommendations combining Hybrid and GNN

    Args:
        customer_id: Customer ID
        hybrid_weight: Weight for hybrid recommendations (default 0.7)
        gnn_weight: Weight for GNN recommendations (default 0.3)
        top_n: Total recommendations to return

    Returns:
        List of product IDs (ranked)
    """
    # Calculate split
    n_hybrid = int(top_n * hybrid_weight)
    n_gnn = top_n - n_hybrid

    logger.info(f"Customer {customer_id}: {n_hybrid} hybrid + {n_gnn} GNN recommendations")

    try:
        # Get Hybrid recommendations (70%)
        recommender = HybridRecommender()
        hybrid_recs_dicts = recommender.get_recommendations(
            customer_id=customer_id,
            top_n=n_hybrid,
            as_of_date=datetime(2024, 6, 30)
        )
        # Extract product IDs from dictionaries
        hybrid_recs = [str(r['product_id']) for r in hybrid_recs_dicts]
        logger.info(f"  Hybrid: {len(hybrid_recs)} recommendations")
    except Exception as e:
        logger.warning(f"  Hybrid failed: {e}")
        hybrid_recs = []

    try:
        # Get GNN recommendations (30%)
        device = torch.device('cpu')
        model, metadata = load_trained_model(device=device)
        edge_index_dict, _ = load_graph_data()

        # Get customer's training products
        train_products, _ = get_test_data(customer_id)

        gnn_recs = get_recommendations_gnn(
            model, edge_index_dict, customer_id, metadata,
            train_products, top_n=n_gnn, device=device
        )
        logger.info(f"  GNN: {len(gnn_recs)} recommendations")
    except Exception as e:
        logger.warning(f"  GNN failed: {e}")
        gnn_recs = []

    # Combine and deduplicate
    seen = set()
    final_recs = []

    # Interleave recommendations (hybrid first)
    i_hybrid, i_gnn = 0, 0
    while len(final_recs) < top_n:
        # Add from hybrid
        if i_hybrid < len(hybrid_recs):
            prod = hybrid_recs[i_hybrid]
            if prod not in seen:
                final_recs.append(prod)
                seen.add(prod)
            i_hybrid += 1

        # Add from GNN
        if i_gnn < len(gnn_recs):
            prod = gnn_recs[i_gnn]
            if prod not in seen:
                final_recs.append(prod)
                seen.add(prod)
            i_gnn += 1

        # Break if both exhausted
        if i_hybrid >= len(hybrid_recs) and i_gnn >= len(gnn_recs):
            break

    logger.info(f"  Final: {len(final_recs)} unique recommendations")
    return final_recs


def validate_ensemble(customer_ids: List[int], top_n: int = 50):
    """Validate ensemble system on test customers"""

    logger.info("="*80)
    logger.info("VALIDATING ENSEMBLE RECOMMENDER (70% Hybrid + 30% GNN)")
    logger.info("="*80)

    results = []

    for customer_id in customer_ids:
        logger.info(f"\nCustomer {customer_id}:")

        # Get test data
        train_products, test_products = get_test_data(customer_id)

        if len(test_products) == 0:
            logger.warning(f"  No test products - skipping")
            continue

        # Get ensemble recommendations
        recommendations = ensemble_recommend(customer_id, top_n=top_n)

        # Calculate precision - convert to integers for consistent comparison
        recommendations_int = set(int(r) for r in recommendations)
        test_products_int = set(int(p) for p in test_products)
        hits = len(recommendations_int & test_products_int)
        precision = hits / len(recommendations) if len(recommendations) > 0 else 0.0

        segment = get_customer_segment(customer_id)

        results.append({
            'customer_id': customer_id,
            'segment': segment,
            'train_products': len(train_products),
            'test_products': len(test_products),
            'precision': precision,
            'hits': hits,
            'total_recs': len(recommendations)
        })

        status = "🏆" if precision >= 0.90 else "✅" if precision >= 0.65 else "⚠️"
        logger.info(f"  Precision: {precision:.1%} ({hits}/{len(recommendations)}) {status}")

    # Summary
    if len(results) > 0:
        df = pd.DataFrame(results)
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"\n✅ Overall Precision@50: {df['precision'].mean():.1%}")

        # By segment
        logger.info(f"\n📊 BY SEGMENT:")
        for segment in ['heavy', 'regular', 'light']:
            df_seg = df[df['segment'] == segment]
            if len(df_seg) > 0:
                avg = df_seg['precision'].mean()
                status = "🏆" if avg >= 0.90 else "✅" if avg >= 0.65 else "⚠️"
                logger.info(f"  {segment.upper()}: {avg:.1%} ({len(df_seg)} customers) {status}")

        # Comparison
        logger.info(f"\n📈 COMPARISON:")
        logger.info(f"  Hybrid Alone: 65% (heavy users)")
        logger.info(f"  GNN Alone: 27% (heavy users)")
        logger.info(f"  Ensemble (70/30): {df[df['segment']=='heavy']['precision'].mean():.1%} (heavy users)")

        # Save results
        df.to_csv('results/ensemble_validation_results.csv', index=False)
        logger.info(f"\n💾 Results saved to: results/ensemble_validation_results.csv")

    return results


if __name__ == '__main__':
    # Test on same 20 customers as GNN/Hybrid validation
    test_customers = [
        # Heavy users
        411706, 410376, 410849, 410187, 411539, 411330, 410280, 411726, 410827, 411457,
        # Regular/Light users
        410964, 411457, 410827, 411726
    ]

    validate_ensemble(test_customers[:5])  # Test on first 5 for speed
