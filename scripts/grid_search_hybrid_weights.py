#!/usr/bin/env python3
"""
Phase E: Grid Search for Optimal Hybrid Feature Weights

Systematically searches for best feature weight combination using:
- Train set (60%): Optimize weights
- Validation set (20%): Select best config
- Test set (20%): Final evaluation on unseen data

Expected: Hybrid 84.2% → 87-90%+
"""

import os
import sys
import logging
import itertools
import numpy as np
import pandas as pd
import pymssql
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.hybrid_recommender import HybridRecommender

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Grid search space for heavy users (most customers)
WEIGHT_GRID_HEAVY = {
    'frequency': [0.60, 0.65, 0.70, 0.75],
    'recency': [0.10, 0.12, 0.15, 0.18],
    'maintenance_cycle': [0.05, 0.08, 0.10, 0.12],
    'seasonality': [0.03, 0.05, 0.07],
    'monetary': [0.03, 0.05, 0.07],
    'compatibility': [0.02, 0.03, 0.04]
}


def connect_mssql():
    """Connect to MSSQL database"""
    return pymssql.connect(
        server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
        port=int(os.environ.get('MSSQL_PORT', '1433')),
        database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
        user=os.environ.get('MSSQL_USER', 'ef_migrator'),
        password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
    )


def get_heavy_customers(conn) -> List[int]:
    """Get all heavy user customers (500+ unique products)"""
    query = """
    SELECT
        ca.ClientID as customer_id,
        COUNT(DISTINCT oi.ProductID) as unique_products
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE o.Created < '2024-07-01'
        AND o.Created IS NOT NULL
        AND oi.ProductID IS NOT NULL
    GROUP BY ca.ClientID
    HAVING COUNT(DISTINCT oi.ProductID) >= 500
    ORDER BY unique_products DESC
    """

    df = pd.read_sql(query, conn)
    logger.info(f"Found {len(df)} heavy customers (500+ products)")
    return df['customer_id'].tolist()


def split_customers(customers: List[int], seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Split customers into train/validation/test sets (60/20/20)"""
    np.random.seed(seed)
    shuffled = np.random.permutation(customers)

    n = len(shuffled)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train = shuffled[:train_end].tolist()
    validation = shuffled[train_end:val_end].tolist()
    test = shuffled[val_end:].tolist()

    logger.info(f"Customer splits: Train={len(train)}, Val={len(validation)}, Test={len(test)}")
    return train, validation, test


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1.0"""
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}


def generate_weight_combinations():
    """Generate all valid weight combinations from grid"""
    keys = list(WEIGHT_GRID_HEAVY.keys())
    values = [WEIGHT_GRID_HEAVY[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        weights = dict(zip(keys, combo))
        # Normalize to sum to 1.0
        normalized = normalize_weights(weights)
        combinations.append(normalized)

    logger.info(f"Generated {len(combinations)} weight combinations to test")
    return combinations


def evaluate_configuration(weights: Dict[str, float],
                          customers: List[int],
                          split_date: datetime = datetime(2024, 7, 1)) -> Dict:
    """
    Evaluate a weight configuration on a set of customers

    Returns dict with precision, hits, total

    NOTE: Creates a NEW HybridRecommender for each evaluation to avoid
    database connection timeouts during long-running grid search.
    """
    # Temporarily override weights in module
    import scripts.hybrid_recommender as hr_module

    # Save original
    original_weights = hr_module.WEIGHTS['heavy'].copy()

    # Update with new weights
    new_weights = original_weights.copy()
    for key, value in weights.items():
        if key in new_weights:
            new_weights[key] = value

    # Set new weights
    hr_module.WEIGHTS['heavy'] = new_weights

    total_hits = 0
    total_test_products = 0
    customer_results = []

    for customer_id in customers:
        # Create NEW recommender with fresh DB connection for each customer
        # This prevents connection timeouts during long grid search
        recommender = None
        try:
            recommender = HybridRecommender()

            # Get recommendations for H1 2024
            recs = recommender.get_recommendations(
                customer_id=customer_id,
                top_n=50,
                as_of_date=split_date
            )

            rec_ids = set(int(r['product_id']) for r in recs)

            # Get H2 2024 actual purchases
            test_query = f"""
            SELECT DISTINCT CAST(oi.ProductID AS VARCHAR(50)) as product_id
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE ca.ClientID = {customer_id}
                AND o.Created >= '2024-07-01'
                AND o.Created IS NOT NULL
                AND oi.ProductID IS NOT NULL
            """

            test_products = set(
                int(p) for p in pd.read_sql(test_query, recommender.conn)['product_id'].tolist()
            )

            if len(test_products) == 0:
                continue

            hits = len(rec_ids & test_products)
            total_hits += hits
            total_test_products += len(test_products)

            customer_results.append({
                'customer_id': customer_id,
                'hits': hits,
                'total': len(test_products),
                'precision': hits / min(50, len(test_products))
            })

        except Exception as e:
            logger.warning(f"Error evaluating customer {customer_id}: {e}")
            continue
        finally:
            # Close connection after each customer
            if recommender and hasattr(recommender, 'conn') and recommender.conn:
                try:
                    recommender.conn.close()
                except:
                    pass

    # Restore original weights
    hr_module.WEIGHTS['heavy'] = original_weights

    if total_test_products == 0:
        return {'precision': 0.0, 'hits': 0, 'total': 0, 'customers': []}

    precision = total_hits / total_test_products

    return {
        'precision': precision,
        'hits': total_hits,
        'total': total_test_products,
        'num_customers': len(customer_results),
        'customers': customer_results
    }


def grid_search(train_customers: List[int], val_customers: List[int]) -> Tuple[Dict, List]:
    """
    Run grid search on training set, validate on validation set

    Returns: (best_weights, all_results)
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE E: GRID SEARCH OPTIMIZATION")
    logger.info("="*80)

    # Generate weight combinations
    weight_combinations = generate_weight_combinations()

    logger.info(f"\nStep 1: Training on {len(train_customers)} customers")
    logger.info(f"Testing {len(weight_combinations)} configurations...")

    train_results = []

    for i, weights in enumerate(weight_combinations):
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{len(weight_combinations)}")

        result = evaluate_configuration(weights, train_customers)
        result['weights'] = weights
        train_results.append(result)

    # Sort by training precision
    train_results.sort(key=lambda x: x['precision'], reverse=True)

    logger.info("\nTop 10 configurations on training set:")
    for i, result in enumerate(train_results[:10]):
        logger.info(f"  {i+1}. Precision: {result['precision']:.3f} | Weights: {result['weights']}")

    # Validate top 20 on validation set
    logger.info(f"\nStep 2: Validating top 20 on {len(val_customers)} validation customers")

    val_results = []
    for i, train_result in enumerate(train_results[:20]):
        logger.info(f"Validating config {i+1}/20...")

        val_result = evaluate_configuration(
            train_result['weights'],
            val_customers
        )
        val_result['weights'] = train_result['weights']
        val_result['train_precision'] = train_result['precision']
        val_results.append(val_result)

    # Sort by validation precision
    val_results.sort(key=lambda x: x['precision'], reverse=True)

    logger.info("\nTop 10 configurations on validation set:")
    for i, result in enumerate(val_results[:10]):
        logger.info(f"  {i+1}. Val: {result['precision']:.3f} | Train: {result['train_precision']:.3f} | Weights: {result['weights']}")

    best_config = val_results[0]

    logger.info("\n" + "="*80)
    logger.info("BEST CONFIGURATION FOUND")
    logger.info("="*80)
    logger.info(f"Validation Precision: {best_config['precision']:.3f}")
    logger.info(f"Training Precision: {best_config['train_precision']:.3f}")
    logger.info(f"Weights:")
    for key, value in best_config['weights'].items():
        logger.info(f"  {key}: {value:.3f}")

    return best_config['weights'], val_results


def final_test(weights: Dict[str, float], test_customers: List[int]) -> Dict:
    """Final evaluation on held-out test set"""
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST ON HELD-OUT SET")
    logger.info("="*80)

    result = evaluate_configuration(weights, test_customers)

    logger.info(f"\nTest Set Results ({len(test_customers)} customers):")
    logger.info(f"  Precision: {result['precision']:.3f}")
    logger.info(f"  Hits: {result['hits']}/{result['total']}")
    logger.info(f"  Customers evaluated: {result['num_customers']}")

    return result


def main():
    logger.info("="*80)
    logger.info("PHASE E: HYBRID WEIGHT GRID SEARCH")
    logger.info("="*80)
    logger.info("\nObjective: Optimize Hybrid from 84.2% to 87-90%+")
    logger.info("Method: Proper train/val/test split with grid search\n")

    # Connect to database
    conn = connect_mssql()

    try:
        # Get heavy customers
        heavy_customers = get_heavy_customers(conn)

        # Split into train/val/test
        train, validation, test = split_customers(heavy_customers)

        # Run grid search
        best_weights, val_results = grid_search(train, validation)

        # Final test on held-out set
        test_results = final_test(best_weights, test)

        # Save results
        output = {
            'best_weights': best_weights,
            'validation_precision': val_results[0]['precision'],
            'test_precision': test_results['precision'],
            'baseline_precision': 0.842,
            'improvement': test_results['precision'] - 0.842,
            'all_val_results': val_results[:10]
        }

        os.makedirs('results', exist_ok=True)
        with open('results/grid_search_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH COMPLETE")
        logger.info("="*80)
        logger.info(f"Baseline (Phase C): 84.2%")
        logger.info(f"Optimized (Test Set): {test_results['precision']*100:.1f}%")
        logger.info(f"Improvement: {(test_results['precision'] - 0.842)*100:+.1f}pp")
        logger.info("\nResults saved to results/grid_search_results.json")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
