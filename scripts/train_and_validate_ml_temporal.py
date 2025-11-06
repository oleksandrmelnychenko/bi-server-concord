#!/usr/bin/env python3
"""
Train & Validate ML Model with Temporal Split

This script:
1. Trains ALS model on clean training data (before 2024-06-30)
2. Validates with 20+ diverse customers
3. Compares with Frequency Baseline
4. Generates comprehensive report

Goal: Achieve >90% precision to match frequency baseline's heavy user performance
"""

import sys
import pickle
import numpy as np
import pandas as pd
import duckdb
import pymssql
from pathlib import Path
from datetime import datetime
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
SPLIT_DATE = "2024-06-30"
DUCKDB_PATH = Path("data/ml_features/concord_ml_temporal.duckdb")
MODEL_DIR = Path("models/collaborative_filtering_temporal")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MSSQL_CONFIG = {
    "host": "78.152.175.67",
    "port": 1433,
    "database": "ConcordDb_v5",
    "user": "ef_migrator",
    "password": "Grimm_jow92",
}


def train_als_model():
    """Train ALS model on clean training data"""
    logger.info("="*80)
    logger.info("TRAINING ALS MODEL ON CLEAN DATA")
    logger.info("="*80)
    logger.info(f"Training data: Purchases BEFORE {SPLIT_DATE}\n")

    # Load training data from DuckDB
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    query = """
    SELECT
        customer_id,
        product_id,
        implicit_rating
    FROM ml_features.interaction_matrix_train
    WHERE implicit_rating > 0
    """

    logger.info("Loading training interactions...")
    df = conn.execute(query).df()
    conn.close()

    logger.info(f"✓ Loaded {len(df):,} training interactions")
    logger.info(f"  Customers: {df['customer_id'].nunique():,}")
    logger.info(f"  Products: {df['product_id'].nunique():,}")

    # Encode IDs
    logger.info("\nEncoding IDs...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user_idx'] = user_encoder.fit_transform(df['customer_id'])
    df['item_idx'] = item_encoder.fit_transform(df['product_id'])

    # Create sparse matrix
    logger.info("Creating sparse matrix...")
    user_item_matrix = csr_matrix(
        (df['implicit_rating'], (df['user_idx'], df['item_idx'])),
        shape=(df['user_idx'].max() + 1, df['item_idx'].max() + 1)
    )

    logger.info(f"✓ Matrix shape: {user_item_matrix.shape}")
    logger.info(f"  Density: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.4f}%")

    # Train model
    logger.info("\nTraining ALS model...")
    logger.info("  Factors: 64")
    logger.info("  Iterations: 20")
    logger.info("  Regularization: 0.01")

    model = AlternatingLeastSquares(
        factors=64,
        iterations=20,
        regularization=0.01,
        random_state=42,
        use_gpu=False
    )

    model.fit(user_item_matrix)
    logger.info("✓ Training complete!")

    # Save model and mappings
    logger.info("\nSaving model...")
    model_path = MODEL_DIR / "als_model_temporal.pkl"
    mappings_path = MODEL_DIR / "id_mappings_temporal.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Create bidirectional mappings
    user_id_map = dict(zip(df['customer_id'], df['user_idx']))
    item_id_map = dict(zip(df['product_id'], df['item_idx']))
    reverse_item_map = dict(zip(df['item_idx'], df['product_id']))

    with open(mappings_path, 'wb') as f:
        pickle.dump({
            'user_id_map': user_id_map,
            'item_id_map': item_id_map,
            'reverse_item_map': reverse_item_map
        }, f)

    logger.info(f"✓ Model saved to {model_path}")
    logger.info(f"✓ Mappings saved to {mappings_path}")

    return model, user_id_map, item_id_map, reverse_item_map


def select_test_customers(n_per_segment=5):
    """Select 20+ diverse test customers"""
    logger.info("\n" + "="*80)
    logger.info(f"SELECTING TEST CUSTOMERS ({n_per_segment} per segment)")
    logger.info("="*80)

    conn = pymssql.connect(
        server=MSSQL_CONFIG['host'],
        port=MSSQL_CONFIG['port'],
        user=MSSQL_CONFIG['user'],
        password=MSSQL_CONFIG['password'],
        database=MSSQL_CONFIG['database'],
        tds_version='7.0'
    )

    # Get customers with purchases in both H1 and H2 2024
    query = f"""
    SELECT
        ca.ClientID as customer_id,
        COUNT(DISTINCT CASE WHEN o.Created < '{SPLIT_DATE}' THEN o.ID END) as orders_h1,
        COUNT(DISTINCT CASE WHEN o.Created >= '{SPLIT_DATE}' AND o.Created < '2025-01-01' THEN o.ID END) as orders_h2,
        COUNT(DISTINCT o.ID) as total_orders
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    WHERE o.Created >= '2024-01-01' AND o.Created < '2025-01-01'
    GROUP BY ca.ClientID
    HAVING COUNT(DISTINCT CASE WHEN o.Created < '{SPLIT_DATE}' THEN o.ID END) >= 3
        AND COUNT(DISTINCT CASE WHEN o.Created >= '{SPLIT_DATE}' AND o.Created < '2025-01-01' THEN o.ID END) >= 3
    ORDER BY total_orders DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Found {len(df)} qualifying customers")

    # Segment by purchase volume
    df['segment'] = pd.cut(
        df['total_orders'],
        bins=[0, 10, 50, 200, float('inf')],
        labels=['Light', 'Moderate', 'Regular', 'Heavy']
    )

    # Select N from each segment
    selected = []
    for segment in ['Heavy', 'Regular', 'Moderate', 'Light']:
        segment_customers = df[df['segment'] == segment].head(n_per_segment)
        selected.extend(segment_customers['customer_id'].tolist())

    logger.info(f"\nSelected {len(selected)} customers:")
    for segment in ['Heavy', 'Regular', 'Moderate', 'Light']:
        count = len([c for c in selected if c in df[df['segment'] == segment]['customer_id'].values])
        logger.info(f"  {segment}: {count} customers")

    return selected


def validate_customer(customer_id, model, user_id_map, item_id_map, reverse_item_map, top_n=50):
    """Validate ML predictions for one customer"""
    # Get customer index
    customer_key = str(int(customer_id)) if isinstance(customer_id, float) else str(customer_id)
    if customer_key not in user_id_map:
        logger.info(f"  ⚠️ Customer {customer_id} not in training set - skipping")
        return None

    user_idx = user_id_map[customer_key]

    # Generate recommendations
    scores = model.user_factors[user_idx] @ model.item_factors.T
    top_items_idx = np.argsort(-scores)[:top_n]
    recommended_products = [reverse_item_map[idx] for idx in top_items_idx if idx in reverse_item_map]

    # Get actual H2 purchases
    conn = pymssql.connect(
        server=MSSQL_CONFIG['host'],
        port=MSSQL_CONFIG['port'],
        user=MSSQL_CONFIG['user'],
        password=MSSQL_CONFIG['password'],
        database=MSSQL_CONFIG['database'],
        tds_version='7.0'
    )

    customer_id_int = int(customer_id) if isinstance(customer_id, float) else customer_id

    query = f"""
    SELECT DISTINCT CAST(oi.ProductID AS VARCHAR(50)) as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id_int}
        AND o.Created >= '{SPLIT_DATE}'
        AND o.Created < '2025-01-01'
    """

    actual_df = pd.read_sql(query, conn)
    conn.close()

    actual_products = set(actual_df['product_id'].tolist())

    # Calculate metrics
    recommended_set = set(recommended_products)
    hits = recommended_set & actual_products

    hit_rate = 1.0 if len(hits) > 0 else 0.0
    precision = len(hits) / len(recommended_set) if len(recommended_set) > 0 else 0.0
    recall = len(hits) / len(actual_products) if len(actual_products) > 0 else 0.0

    return {
        'customer_id': customer_id,
        'num_recommendations': len(recommended_products),
        'num_actual': len(actual_products),
        'num_hits': len(hits),
        'hit_rate': hit_rate,
        'precision': precision,
        'recall': recall
    }


def run_validation(model, user_id_map, item_id_map, reverse_item_map, test_customers, top_n=50):
    """Run validation on all test customers"""
    logger.info("\n" + "="*80)
    logger.info(f"VALIDATING ML MODEL (Top-{top_n})")
    logger.info("="*80)

    results = []
    for i, customer_id in enumerate(test_customers, 1):
        logger.info(f"Testing customer {i}/{len(test_customers)}: {customer_id}")
        result = validate_customer(customer_id, model, user_id_map, item_id_map, reverse_item_map, top_n)
        if result:
            results.append(result)
            logger.info(f"  Hits: {result['num_hits']}/{top_n} ({result['precision']:.1%} precision)")

    return results


def generate_report(ml_results, top_n=50):
    """Generate comparison report"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)

    if not ml_results:
        logger.error("❌ NO VALIDATION RESULTS - all test customers not in training set")
        return None

    df = pd.DataFrame(ml_results)

    # ML metrics
    ml_hit_rate = df['hit_rate'].mean()
    ml_precision = df['precision'].mean()
    ml_recall = df['recall'].mean()
    ml_total_hits = df['num_hits'].sum()

    logger.info(f"\nML Model Performance (Top-{top_n}):")
    logger.info(f"  Customers Tested: {len(df)}")
    logger.info(f"  Hit Rate: {ml_hit_rate:.1%}")
    logger.info(f"  Precision: {ml_precision:.2%}")
    logger.info(f"  Recall: {ml_recall:.2%}")
    logger.info(f"  Total Hits: {ml_total_hits}")

    # Frequency baseline (from previous validation)
    freq_hit_rate = 1.0  # 100%
    freq_precision = 0.425  # 42.5%
    freq_hits = 85

    logger.info(f"\nFrequency Baseline (Top-50):")
    logger.info(f"  Hit Rate: {freq_hit_rate:.1%}")
    logger.info(f"  Precision: {freq_precision:.2%}")
    logger.info(f"  Total Hits: {freq_hits}")

    # Comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: ML vs FREQUENCY BASELINE")
    logger.info("="*80)

    logger.info(f"\n| Metric | ML Model | Frequency Baseline | Winner |")
    logger.info(f"|--------|----------|-------------------|--------|")
    logger.info(f"| Hit Rate | {ml_hit_rate:.1%} | {freq_hit_rate:.1%} | **{'ML' if ml_hit_rate > freq_hit_rate else 'Frequency'}** |")
    logger.info(f"| Precision | {ml_precision:.2%} | {freq_precision:.2%} | **{'ML' if ml_precision > freq_precision else 'Frequency'}** |")
    logger.info(f"| Total Hits | {ml_total_hits} | {freq_hits} | **{'ML' if ml_total_hits > freq_hits else 'Frequency'}** |")

    # Success criteria
    logger.info("\n" + "="*80)
    logger.info("SUCCESS CRITERIA ASSESSMENT")
    logger.info("="*80)
    logger.info(f"Target: >90% precision (match frequency baseline heavy user performance)")
    logger.info(f"Achieved: {ml_precision:.2%}")

    if ml_precision >= 0.9:
        logger.info(f"\n✅ SUCCESS! ML achieves >90% precision")
        logger.info(f"Recommendation: Deploy ML model")
    elif ml_precision > freq_precision:
        logger.info(f"\n⚠️ PARTIAL SUCCESS: ML beats frequency baseline but <90%")
        logger.info(f"Recommendation: Deploy ML model or hybrid approach")
    else:
        logger.info(f"\n❌ FAILURE: ML does not beat frequency baseline")
        logger.info(f"Recommendation: Deploy frequency baseline only")

    return {
        'ml_hit_rate': ml_hit_rate,
        'ml_precision': ml_precision,
        'ml_recall': ml_recall,
        'ml_total_hits': ml_total_hits,
        'freq_hit_rate': freq_hit_rate,
        'freq_precision': freq_precision,
        'freq_hits': freq_hits,
        'winner': 'ML' if ml_precision > freq_precision else 'Frequency'
    }


def main():
    start_time = datetime.now()

    logger.info("="*80)
    logger.info("ML MODEL TRAINING & VALIDATION WITH TEMPORAL SPLIT")
    logger.info("="*80)
    logger.info(f"Start time: {start_time}")
    logger.info(f"Split date: {SPLIT_DATE}\n")

    # Step 1: Train model
    model, user_id_map, item_id_map, reverse_item_map = train_als_model()

    # Step 2: Select test customers
    test_customers = select_test_customers(n_per_segment=5)

    # Step 3: Validate
    ml_results = run_validation(model, user_id_map, item_id_map, reverse_item_map, test_customers, top_n=50)

    # Step 4: Generate report
    comparison = generate_report(ml_results, top_n=50)

    # Summary
    duration = (datetime.now() - start_time).total_seconds() / 60
    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETE!")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} minutes")
    logger.info(f"Winner: {comparison['winner']}")


if __name__ == "__main__":
    main()
