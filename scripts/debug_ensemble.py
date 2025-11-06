#!/usr/bin/env python3
"""
Debug ensemble to understand why performance is so bad
"""

import sys
import logging
from datetime import datetime
import torch

from hybrid_recommender import HybridRecommender
from validate_gnn_recommender import (
    load_trained_model,
    get_recommendations_gnn,
    get_test_data,
    get_customer_segment
)
from build_gnn_recommender import load_graph_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_customer_411706():
    """Debug customer 411706 - the best performing customer"""

    customer_id = 411706
    logger.info(f"\n{'='*80}")
    logger.info(f"DEBUGGING CUSTOMER {customer_id}")
    logger.info(f"{'='*80}")

    # Get test data
    train_products, test_products = get_test_data(customer_id)
    logger.info(f"\nTest products: {len(test_products)}")
    logger.info(f"First 5 test products: {list(test_products)[:5]}")
    logger.info(f"Test product types: {type(list(test_products)[0])}")

    # Get Hybrid recommendations
    logger.info(f"\n--- HYBRID RECOMMENDATIONS ---")
    recommender = HybridRecommender()
    hybrid_recs_dicts = recommender.get_recommendations(
        customer_id=customer_id,
        top_n=35,
        as_of_date=datetime(2024, 6, 30)
    )
    hybrid_recs = [str(r['product_id']) for r in hybrid_recs_dicts]
    logger.info(f"Hybrid returned: {len(hybrid_recs)} recommendations")
    logger.info(f"First 10 hybrid recs: {hybrid_recs[:10]}")
    logger.info(f"Hybrid rec types: {type(hybrid_recs[0])}")

    # Check hybrid hits
    hybrid_recs_set = set(hybrid_recs)
    test_products_str = set(str(p) for p in test_products)
    hybrid_hits = hybrid_recs_set & test_products_str
    logger.info(f"Hybrid hits: {len(hybrid_hits)}/35 = {len(hybrid_hits)/35:.1%}")

    # Get GNN recommendations
    logger.info(f"\n--- GNN RECOMMENDATIONS ---")
    device = torch.device('cpu')
    model, metadata = load_trained_model(device=device)
    edge_index_dict, _ = load_graph_data()

    gnn_recs = get_recommendations_gnn(
        model, edge_index_dict, customer_id, metadata,
        train_products, top_n=15, device=device
    )
    logger.info(f"GNN returned: {len(gnn_recs)} recommendations")
    logger.info(f"First 10 GNN recs: {gnn_recs[:10]}")
    logger.info(f"GNN rec types: {type(gnn_recs[0])}")

    # Check GNN hits
    gnn_recs_int = [int(r) for r in gnn_recs]
    test_products_int = set(int(p) for p in test_products)
    gnn_hits = set(gnn_recs_int) & test_products_int
    logger.info(f"GNN hits: {len(gnn_hits)}/15 = {len(gnn_hits)/15:.1%}")

    # Check overlap between hybrid and GNN
    hybrid_recs_int = [int(r) for r in hybrid_recs]
    overlap = set(hybrid_recs_int) & set(gnn_recs_int)
    logger.info(f"\nOverlap between Hybrid and GNN: {len(overlap)} products")

    # Simulate ensemble
    logger.info(f"\n--- ENSEMBLE SIMULATION ---")
    seen = set()
    final_recs = []

    # Interleave
    i_hybrid, i_gnn = 0, 0
    while len(final_recs) < 50:
        # Add from hybrid
        if i_hybrid < len(hybrid_recs):
            prod = hybrid_recs[i_hybrid]
            if prod not in seen:
                final_recs.append(prod)
                seen.add(prod)
            i_hybrid += 1

        # Add from GNN
        if i_gnn < len(gnn_recs):
            prod = str(gnn_recs[i_gnn])  # Convert to string
            if prod not in seen:
                final_recs.append(prod)
                seen.add(prod)
            i_gnn += 1

        # Break if both exhausted
        if i_hybrid >= len(hybrid_recs) and i_gnn >= len(gnn_recs):
            break

    logger.info(f"Final ensemble: {len(final_recs)} recommendations")
    logger.info(f"First 20: {final_recs[:20]}")

    # Check ensemble hits
    final_recs_int = [int(r) for r in final_recs]
    ensemble_hits = set(final_recs_int) & test_products_int
    logger.info(f"Ensemble hits: {len(ensemble_hits)}/50 = {len(ensemble_hits)/50:.1%}")

    logger.info(f"\n--- COMPARISON ---")
    logger.info(f"Hybrid alone (35):  {len(hybrid_hits)}/35 = {len(hybrid_hits)/35:.1%}")
    logger.info(f"GNN alone (15):     {len(gnn_hits)}/15 = {len(gnn_hits)/15:.1%}")
    logger.info(f"Ensemble (50):      {len(ensemble_hits)}/50 = {len(ensemble_hits)/50:.1%}")

    # Check what happened to the best recommendations
    logger.info(f"\n--- ANALYZING WHAT WENT WRONG ---")
    logger.info(f"Hybrid's first 10 recommendations:")
    for i, prod in enumerate(hybrid_recs[:10]):
        in_test = "✓ HIT" if int(prod) in test_products_int else "✗ MISS"
        logger.info(f"  {i+1}. {prod} {in_test}")

    logger.info(f"\nGNN's first 10 recommendations:")
    for i, prod in enumerate(gnn_recs[:10]):
        in_test = "✓ HIT" if int(prod) in test_products_int else "✗ MISS"
        logger.info(f"  {i+1}. {prod} {in_test}")

    logger.info(f"\nEnsemble's first 20 recommendations:")
    for i, prod in enumerate(final_recs[:20]):
        in_test = "✓ HIT" if int(prod) in test_products_int else "✗ MISS"
        source = "H" if i % 2 == 0 else "G"  # Rough estimate
        logger.info(f"  {i+1}. {prod} {in_test} [{source}]")

if __name__ == '__main__':
    debug_customer_411706()
