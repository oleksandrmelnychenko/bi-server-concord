#!/usr/bin/env python3
"""
Validate Neural Recommender on same 20 customers as ALS

Compares:
1. Neural Network (embeddings + deep learning)
2. ALS Collaborative Filtering (baseline ML)
3. Frequency Baseline (simple approach)

Expected: Neural Network > Frequency Baseline > ALS
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import pymssql
import pickle
import torch
from datetime import datetime
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import neural network model
sys.path.append('scripts')
from train_neural_recommender import NeuralRecommender

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
    SELECT DISTINCT CAST(oi.ProductID AS INT) as product_id
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


def load_neural_model():
    """Load trained neural network model"""
    logger.info("Loading neural network model...")

    # Load model checkpoint
    checkpoint = torch.load('models/neural_recommender/best_model.pt', map_location='cpu')

    # Load encoders and features
    with open('models/neural_recommender/encoders.pkl', 'rb') as f:
        data = pickle.load(f)
        customer_encoder = data['customer_encoder']
        product_encoder = data['product_encoder']
        customer_scaler = data['customer_scaler']
        product_scaler = data['product_scaler']
        customer_features = data['customer_features']
        product_features = data['product_features']

    # Reconstruct model
    n_customers = len(customer_encoder.classes_)
    n_products = len(product_encoder.classes_)
    n_customer_features = len([c for c in customer_features.columns if c != 'customer_id'])
    n_product_features = len([c for c in product_features.columns if c != 'product_id'])

    model = NeuralRecommender(
        n_customers=n_customers,
        n_products=n_products,
        n_customer_features=n_customer_features,
        n_product_features=n_product_features
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"✓ Model loaded (validation loss: {checkpoint['val_loss']:.4f})")

    return model, customer_encoder, product_encoder, customer_features, product_features


def get_neural_recommendations(customer_id: int, model, customer_encoder, product_encoder,
                               customer_features, product_features, top_n: int = 50) -> List[int]:
    """Get recommendations from neural network"""

    # Check if customer is in training set
    if customer_id not in customer_encoder.classes_:
        logger.warning(f"  Customer {customer_id} not in training set")
        return []

    # Encode customer
    customer_idx = customer_encoder.transform([customer_id])[0]
    customer_idx_tensor = torch.LongTensor([customer_idx])

    # Get customer features
    cust_feats = customer_features[customer_features['customer_id'] == customer_id].iloc[0]
    cust_feats = cust_feats[[c for c in customer_features.columns if c != 'customer_id']].values
    cust_feats_tensor = torch.FloatTensor([cust_feats])

    # Score all products
    all_products = product_encoder.classes_
    product_indices = torch.LongTensor(range(len(all_products)))

    # Get product features for all products
    prod_feats = product_features[[c for c in product_features.columns if c != 'product_id']].values
    prod_feats_tensor = torch.FloatTensor(prod_feats)

    # Expand customer features to match product count
    cust_feats_expanded = cust_feats_tensor.repeat(len(all_products), 1)
    customer_idx_expanded = customer_idx_tensor.repeat(len(all_products))

    # Get predictions
    with torch.no_grad():
        scores = model(customer_idx_expanded, product_indices, cust_feats_expanded, prod_feats_tensor)

    # Get top N
    top_indices = torch.argsort(scores.squeeze(), descending=True)[:top_n]
    top_product_ids = [int(all_products[idx]) for idx in top_indices]

    return top_product_ids


def validate_neural_network(test_customers: List[int], top_n: int = 50):
    """Validate neural network on test customers"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING NEURAL NETWORK")
    logger.info("="*80)

    # Load model
    model, customer_encoder, product_encoder, customer_features, product_features = load_neural_model()

    # Connect to database
    conn = connect_mssql()

    results = []

    for i, customer_id in enumerate(test_customers, 1):
        logger.info(f"\nTesting customer {i}/{len(test_customers)}: {customer_id}")

        # Get neural recommendations
        recommendations = get_neural_recommendations(
            customer_id, model, customer_encoder, product_encoder,
            customer_features, product_features, top_n=top_n
        )

        if not recommendations:
            logger.info("  ⚠️ No recommendations (customer not in training set)")
            continue

        # Get actual purchases
        actual_purchases = get_validation_purchases(customer_id, conn)

        # Calculate metrics
        recommended_set = set(map(str, recommendations))
        hits = recommended_set & actual_purchases

        precision = len(hits) / len(recommended_set) if recommended_set else 0
        recall = len(hits) / len(actual_purchases) if actual_purchases else 0
        hit_rate = 1.0 if len(hits) > 0 else 0.0

        logger.info(f"  Precision: {precision:.1%} ({len(hits)}/{len(recommended_set)})")
        logger.info(f"  Actual purchases in H2 2024: {len(actual_purchases)}")

        results.append({
            'customer_id': customer_id,
            'precision': precision,
            'recall': recall,
            'hit_rate': hit_rate,
            'num_hits': len(hits),
            'num_recommended': len(recommended_set),
            'num_actual': len(actual_purchases)
        })

    conn.close()

    # Generate report
    if not results:
        logger.error("❌ NO RESULTS - All customers not in training set")
        return None

    df = pd.DataFrame(results)

    logger.info("\n" + "="*80)
    logger.info("NEURAL NETWORK RESULTS")
    logger.info("="*80)

    avg_precision = df['precision'].mean()
    avg_recall = df['recall'].mean()
    hit_rate = df['hit_rate'].mean()

    logger.info(f"\nCustomers tested: {len(df)}")
    logger.info(f"Hit rate: {hit_rate:.1%} ({int(hit_rate * len(df))}/{len(df)} customers)")
    logger.info(f"Average Precision@{top_n}: {avg_precision:.1%}")
    logger.info(f"Average Recall@{top_n}: {avg_recall:.1%}")

    # Top performers
    logger.info(f"\nTop 5 performers:")
    top_5 = df.nlargest(5, 'precision')[['customer_id', 'precision', 'num_hits']]
    for _, row in top_5.iterrows():
        logger.info(f"  Customer {int(row['customer_id'])}: {row['precision']:.1%} ({int(row['num_hits'])}/{top_n})")

    return df


def compare_all_models():
    """Compare all three approaches"""
    logger.info("\n" + "="*80)
    logger.info("COMPARING ALL MODELS")
    logger.info("="*80)

    # Test customers (same 20 as ALS validation)
    test_customers = [
        410376, 410756, 410849, 411539, 410282,  # Regular
        410280, 411211, 410827, 411726, 410187,  # Heavy
        411390, 410380, 411706, 411600, 410916,  # Mixed
        410338, 410235, 411551, 410744, 410381   # Light
    ]

    logger.info(f"\nTest set: {len(test_customers)} customers")
    logger.info("Testing neural network...")

    # Validate neural network
    neural_results = validate_neural_network(test_customers, top_n=TOP_N)

    if neural_results is None:
        logger.error("Neural network validation failed")
        return

    # Load results from previous validations
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)

    # Neural network
    neural_precision = neural_results['precision'].mean()
    neural_hit_rate = neural_results['hit_rate'].mean()

    # These would come from previous runs
    als_precision = 0.227  # From ALS validation
    als_hit_rate = 0.95

    frequency_precision = 0.425  # From frequency baseline
    frequency_hit_rate = 1.00

    logger.info("\n| Model | Precision@50 | Hit Rate | Winner |")
    logger.info("|-------|-------------|----------|--------|")
    logger.info(f"| Neural Network | {neural_precision:.1%} | {neural_hit_rate:.1%} | {'✅' if neural_precision > max(als_precision, frequency_precision) else ''} |")
    logger.info(f"| Frequency Baseline | {frequency_precision:.1%} | {frequency_hit_rate:.1%} | {'✅' if frequency_precision > max(neural_precision, als_precision) else ''} |")
    logger.info(f"| ALS Collaborative | {als_precision:.1%} | {als_hit_rate:.1%} | {'✅' if als_precision > max(neural_precision, frequency_precision) else ''} |")

    # Determine winner
    winner = "Neural Network" if neural_precision > max(als_precision, frequency_precision) else \
             "Frequency Baseline" if frequency_precision > max(neural_precision, als_precision) else \
             "ALS"

    logger.info(f"\n🏆 WINNER: {winner}")
    logger.info(f"   Precision: {max(neural_precision, als_precision, frequency_precision):.1%}")

    if neural_precision > frequency_precision:
        improvement = ((neural_precision - frequency_precision) / frequency_precision) * 100
        logger.info(f"\n✅ Neural Network beats Frequency Baseline by {improvement:.1f}%!")
        logger.info(f"   {neural_precision:.1%} vs {frequency_precision:.1%}")
    elif neural_precision > als_precision:
        improvement = ((neural_precision - als_precision) / als_precision) * 100
        logger.info(f"\n✅ Neural Network beats ALS by {improvement:.1f}%!")
        logger.info(f"   {neural_precision:.1%} vs {als_precision:.1%}")
    else:
        logger.info(f"\n⚠️ Neural Network did not beat the baselines")
        logger.info(f"   Need to improve architecture or hyperparameters")


def main():
    """Main validation pipeline"""
    print("="*80)
    print("NEURAL NETWORK VALIDATION")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print(f"Validation period: {VALIDATION_START} to {VALIDATION_END}")
    print()

    compare_all_models()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
