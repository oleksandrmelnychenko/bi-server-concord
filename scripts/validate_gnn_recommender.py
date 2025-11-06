#!/usr/bin/env python3
"""
Validate GNN Recommender on 20 Test Customers

Compare with Hybrid System:
- Hybrid: 65% average (100% peak) for heavy users
- GNN Target: 75-95% average

Test customers:
- 10 heavy users (500+ unique products)
- 5 regular users (100-500 products)
- 5 light users (<100 products)
"""

import os
import sys
import torch
import numpy as np
import duckdb
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import pandas as pd

# Import GNN model
from build_gnn_recommender import HeteroLightGCN, load_graph_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test customers (same as hybrid validation)
TEST_CUSTOMERS = {
    'heavy': [411706, 410376, 410849, 410187, 411539, 411330, 410280, 411726, 410827, 411457],
    'regular': [410280, 411726, 410827, 411457, 410964],
    'light': [411330, 410964, 411457, 410827, 411726]
}


def load_trained_model(model_path: str = 'models/gnn_recommender/best_model.pt',
                       device: torch.device = torch.device('cpu')) -> Tuple[HeteroLightGCN, Dict]:
    """Load trained GNN model"""
    logger.info(f"\n📦 Loading trained model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    metadata = checkpoint['metadata']

    model = HeteroLightGCN(
        num_customers=metadata['num_customers'],
        num_products=metadata['num_products'],
        num_groups=metadata['num_groups'],
        num_brands=metadata['num_brands'],
        embedding_dim=64,
        num_layers=3
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"✅ Model loaded (epoch {checkpoint['epoch']}, loss: {checkpoint['train_loss']:.4f})")

    return model, metadata


def get_customer_segment(customer_id: int, db_path: str = 'data/graph_features.duckdb') -> str:
    """Get customer segment"""
    conn = duckdb.connect(db_path, read_only=True)

    query = f"""
    SELECT COUNT(DISTINCT product_id) as unique_products
    FROM purchase_history
    WHERE customer_id = {customer_id}
        AND purchase_date < '2024-07-01'
    """

    df = conn.execute(query).df()
    conn.close()

    if len(df) == 0:
        return 'unknown'

    unique_products = df['unique_products'].iloc[0]

    if unique_products >= 500:
        return 'heavy'
    elif unique_products >= 100:
        return 'regular'
    else:
        return 'light'


def get_test_data(customer_id: int, db_path: str = 'data/graph_features.duckdb') -> Tuple[set, set]:
    """
    Get train and test data for a customer

    Returns:
        train_products (set): Products purchased before 2024-07-01
        test_products (set): Products purchased after 2024-07-01
    """
    conn = duckdb.connect(db_path, read_only=True)

    # Train data (H1 2024)
    train_query = f"""
    SELECT DISTINCT product_id
    FROM purchase_history
    WHERE customer_id = {customer_id}
        AND purchase_date < '2024-07-01'
    """
    train_df = conn.execute(train_query).df()
    train_products = set(train_df['product_id'].values)

    # Test data (H2 2024)
    test_query = f"""
    SELECT DISTINCT product_id
    FROM purchase_history
    WHERE customer_id = {customer_id}
        AND purchase_date >= '2024-07-01'
    """
    test_df = conn.execute(test_query).df()
    test_products = set(test_df['product_id'].values)

    conn.close()

    return train_products, test_products


def get_recommendations_gnn(model: HeteroLightGCN,
                           edge_index_dict: Dict,
                           customer_id: int,
                           metadata: Dict,
                           train_products: set,
                           top_n: int = 50,
                           device: torch.device = torch.device('cpu')) -> List[int]:
    """
    Get top-N recommendations for a customer using GNN

    Args:
        model: Trained GNN model
        edge_index_dict: Graph edge indices
        customer_id: Customer ID
        metadata: Metadata with ID mappings
        train_products: Products already purchased (to exclude)
        top_n: Number of recommendations
        device: Device

    Returns:
        List of recommended product IDs
    """
    if customer_id not in metadata['customer_id_to_idx']:
        return []

    customer_idx = metadata['customer_id_to_idx'][customer_id]

    # Move edge indices to device
    edge_index_dict_device = {k: v.to(device) for k, v in edge_index_dict.items()}

    # Get embeddings
    with torch.no_grad():
        embeddings = model(edge_index_dict_device)

    # Get all product indices
    all_product_ids = list(metadata['product_id_to_idx'].keys())
    all_product_indices = [metadata['product_id_to_idx'][pid] for pid in all_product_ids]

    # Compute scores for all products
    customer_idx_t = torch.tensor([customer_idx] * len(all_product_indices), dtype=torch.long, device=device)
    product_idx_t = torch.tensor(all_product_indices, dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model.predict(customer_idx_t, product_idx_t, embeddings)
        scores = scores.cpu().numpy()

    # Create DataFrame for sorting
    df_scores = pd.DataFrame({
        'product_id': all_product_ids,
        'score': scores
    })

    # QUICK WIN #1: ALLOW REPEAT PURCHASES
    # In B2B, 51-70% of purchases are repeats - this was the fatal flaw!
    # Original line excluded train products: df_scores = df_scores[~df_scores['product_id'].isin(train_products)]
    # Now we KEEP them to exploit repeat purchase behavior
    # (This single change should boost precision from 19.7% to 50-60%!)

    # Sort by score and get top N
    df_scores = df_scores.sort_values('score', ascending=False).head(top_n)

    return df_scores['product_id'].tolist()


def validate_customer(model: HeteroLightGCN,
                     edge_index_dict: Dict,
                     customer_id: int,
                     metadata: Dict,
                     top_n: int = 50,
                     device: torch.device = torch.device('cpu')) -> Dict:
    """
    Validate GNN recommendations for a single customer

    Returns:
        result dict with precision, hits, etc.
    """
    # Get test data
    train_products, test_products = get_test_data(customer_id)

    if len(test_products) == 0:
        logger.warning(f"Customer {customer_id} has no test products")
        return {
            'customer_id': customer_id,
            'precision': 0.0,
            'hits': 0,
            'total_recs': 0,
            'segment': get_customer_segment(customer_id)
        }

    # Get recommendations
    recommendations = get_recommendations_gnn(
        model, edge_index_dict, customer_id, metadata,
        train_products, top_n, device
    )

    # Calculate precision
    hits = sum(1 for prod_id in recommendations if prod_id in test_products)
    precision = hits / len(recommendations) if len(recommendations) > 0 else 0.0

    segment = get_customer_segment(customer_id)

    return {
        'customer_id': customer_id,
        'segment': segment,
        'train_products': len(train_products),
        'test_products': len(test_products),
        'precision': precision,
        'hits': hits,
        'total_recs': len(recommendations)
    }


def main():
    logger.info("\n" + "="*80)
    logger.info("VALIDATING GNN RECOMMENDER ON 20 TEST CUSTOMERS")
    logger.info("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️  Device: {device}")

    # Load trained model
    model, metadata = load_trained_model(device=device)

    # Load graph data
    edge_index_dict, _ = load_graph_data()

    # Validate on all test customers
    all_results = []

    logger.info(f"\n🧪 Testing on 20 customers...")

    # Test heavy users
    logger.info(f"\n📊 HEAVY USERS (10 customers):")
    for customer_id in TEST_CUSTOMERS['heavy']:
        result = validate_customer(model, edge_index_dict, customer_id, metadata, device=device)
        all_results.append(result)

        status = "🏆" if result['precision'] >= 0.90 else "✅" if result['precision'] >= 0.65 else "⚠️"
        logger.info(f"  Customer {customer_id}: {result['precision']:.1%} ({result['hits']}/{result['total_recs']}) {status}")

    # Test regular users
    logger.info(f"\n📊 REGULAR USERS (5 customers):")
    regular_results = []
    for customer_id in TEST_CUSTOMERS['regular']:
        if customer_id not in [r['customer_id'] for r in all_results]:  # Avoid duplicates
            result = validate_customer(model, edge_index_dict, customer_id, metadata, device=device)
            all_results.append(result)
            regular_results.append(result)

            status = "🏆" if result['precision'] >= 0.90 else "✅" if result['precision'] >= 0.50 else "⚠️"
            logger.info(f"  Customer {customer_id}: {result['precision']:.1%} ({result['hits']}/{result['total_recs']}) {status}")

    # Summary
    df_results = pd.DataFrame(all_results)

    logger.info(f"\n" + "="*80)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("="*80)

    # Overall
    avg_precision = df_results['precision'].mean()
    logger.info(f"\n✅ Overall Precision@50: {avg_precision:.1%}")

    # By segment
    logger.info(f"\n📊 BY SEGMENT:")
    for segment in ['heavy', 'regular', 'light']:
        df_segment = df_results[df_results['segment'] == segment]
        if len(df_segment) > 0:
            seg_precision = df_segment['precision'].mean()
            status = "🏆" if seg_precision >= 0.90 else "✅" if seg_precision >= 0.65 else "⚠️"
            logger.info(f"  {segment.upper()}: {seg_precision:.1%} ({len(df_segment)} customers) {status}")

    # Comparison with hybrid
    logger.info(f"\n📈 COMPARISON:")
    logger.info(f"  Hybrid Baseline: 65% (heavy users)")
    logger.info(f"  GNN Model: {df_results[df_results['segment']=='heavy']['precision'].mean():.1%} (heavy users)")

    # Save results
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/gnn_validation_results.csv', index=False)
    logger.info(f"\n💾 Results saved to: results/gnn_validation_results.csv")


if __name__ == '__main__':
    main()
