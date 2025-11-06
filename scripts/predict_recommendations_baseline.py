#!/usr/bin/env python3
"""
Frequency-Based Baseline Recommendation Engine

Simple, explainable, production-ready recommendations based on:
1. Purchase frequency (how many times customer bought this product)
2. Recency (when was it last purchased)
3. Monetary value (how much they spent on it)

This serves as:
- Production fallback when ML fails
- Baseline for ML to beat
- Proof that the task is solvable
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/ml_features")
DUCKDB_PATH = DATA_DIR / "concord_ml.duckdb"


class FrequencyBaselineEngine:
    """
    Simple frequency-based recommendation engine

    Strategy:
    1. Recommend products customer has purchased most frequently
    2. Use recency and monetary value as tie-breakers
    3. Filter out recently purchased products (< 30 days)
    4. Add popular products for discovery (if needed to fill slots)
    """

    def __init__(self):
        self.duckdb_path = DUCKDB_PATH

    def get_customer_purchase_history(self, customer_id):
        """
        Get customer's complete purchase history with frequency and recency

        Returns DataFrame with:
        - product_id
        - num_purchases (frequency)
        - total_spent (monetary value)
        - last_purchase_date (recency)
        - days_since_last_purchase
        """
        conn = duckdb.connect(str(self.duckdb_path), read_only=True)

        query = f"""
        SELECT
            product_id,
            num_purchases,
            total_spent,
            last_purchase_date,
            days_since_last_purchase
        FROM ml_features.interaction_matrix
        WHERE customer_id = '{customer_id}'
        ORDER BY num_purchases DESC, days_since_last_purchase ASC
        """

        try:
            df = conn.execute(query).df()
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error loading purchase history: {e}")
            conn.close()
            return pd.DataFrame()

    def get_popular_products(self, exclude_products, top_n=20):
        """
        Get globally popular products (most purchased across all customers)
        Used for discovery recommendations
        """
        conn = duckdb.connect(str(self.duckdb_path), read_only=True)

        # Convert exclude list to SQL format
        if len(exclude_products) > 0:
            exclude_list = ','.join([f"'{p}'" for p in exclude_products])
            exclude_clause = f"AND product_id NOT IN ({exclude_list})"
        else:
            exclude_clause = ""

        query = f"""
        SELECT
            product_id,
            SUM(num_purchases) as total_purchases,
            COUNT(DISTINCT customer_id) as num_customers,
            AVG(implicit_rating) as avg_rating
        FROM ml_features.interaction_matrix
        WHERE num_purchases > 0
            {exclude_clause}
        GROUP BY product_id
        ORDER BY total_purchases DESC, num_customers DESC
        LIMIT {top_n}
        """

        try:
            df = conn.execute(query).df()
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error loading popular products: {e}")
            conn.close()
            return pd.DataFrame()

    def get_recommendations(self, customer_id, top_n=50, repurchase_ratio=0.8):
        """
        Generate recommendations for a customer

        Args:
            customer_id: Customer ID
            top_n: Number of recommendations to return
            repurchase_ratio: Fraction of recommendations that should be repurchase (vs discovery)

        Returns:
            List of recommendation dictionaries
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"FREQUENCY BASELINE - Customer {customer_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Target: {top_n} total recommendations ({int(top_n * repurchase_ratio)} repurchase + {int(top_n * (1 - repurchase_ratio))} discovery)")

        # Get customer purchase history
        history = self.get_customer_purchase_history(customer_id)

        if history.empty:
            logger.warning(f"No purchase history found for customer {customer_id}")
            return []

        logger.info(f"Customer has purchased {len(history)} unique products")

        # Calculate target counts
        repurchase_target = int(top_n * repurchase_ratio)
        discovery_target = top_n - repurchase_target

        recommendations = []

        # ===================================================================
        # PART 1: REPURCHASE RECOMMENDATIONS (Frequency-based)
        # ===================================================================

        # Filter: Only products not purchased very recently (give them 30+ days)
        # This mimics "time to repurchase" logic
        repurchase_candidates = history[history['days_since_last_purchase'] >= 30].copy()

        if len(repurchase_candidates) == 0:
            # If everything was purchased recently, include everything
            repurchase_candidates = history.copy()

        # Score: Prioritize high-frequency + high-value + not-too-recent
        repurchase_candidates['score'] = (
            repurchase_candidates['num_purchases'] * 10 +  # Frequency is most important
            np.log1p(repurchase_candidates['total_spent']) * 1 +  # Value matters
            (365 - repurchase_candidates['days_since_last_purchase']) / 365 * 2  # Recent (but not TOO recent) is good
        )

        # Sort by score
        repurchase_candidates = repurchase_candidates.sort_values('score', ascending=False)

        # Take top N repurchase recommendations
        for _, row in repurchase_candidates.head(repurchase_target).iterrows():
            recommendations.append({
                'product_id': str(row['product_id']),
                'type': 'REPURCHASE',
                'score': float(row['score']),
                'num_purchases': int(row['num_purchases']),
                'days_since_last_purchase': int(row['days_since_last_purchase']),
                'total_spent': float(row['total_spent']),
                'reason': f"Frequently ordered ({int(row['num_purchases'])}x), last ordered {int(row['days_since_last_purchase'])} days ago"
            })

        logger.info(f"✓ Generated {len(recommendations)} repurchase recommendations (frequency-based)")

        # ===================================================================
        # PART 2: DISCOVERY RECOMMENDATIONS (Popular products)
        # ===================================================================

        if len(recommendations) < top_n:
            # Get products customer has NOT purchased yet
            purchased_products = set(history['product_id'].astype(str).tolist())
            popular = self.get_popular_products(
                exclude_products=purchased_products,
                top_n=discovery_target
            )

            for _, row in popular.iterrows():
                recommendations.append({
                    'product_id': str(row['product_id']),
                    'type': 'DISCOVERY',
                    'score': float(row['total_purchases']),
                    'num_purchases': 0,
                    'days_since_last_purchase': None,
                    'total_spent': 0,
                    'reason': f"Popular product ({int(row['total_purchases'])} total orders across {int(row['num_customers'])} customers)"
                })

            logger.info(f"✓ Added {len(recommendations) - repurchase_target} discovery recommendations (popular products)")

        # ===================================================================
        # FINAL: Return top N
        # ===================================================================

        recommendations = recommendations[:top_n]

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Generated {len(recommendations)} total recommendations")
        logger.info(f"{'='*80}")

        return recommendations

    def get_recommendations_with_details(self, customer_id, top_n=50):
        """
        Generate recommendations and print detailed output
        """
        recommendations = self.get_recommendations(customer_id, top_n=top_n)

        logger.info(f"\n📋 TOP {len(recommendations)} RECOMMENDATIONS:\n")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i:3d}. Product: {rec['product_id']:<12} | "
                       f"Type: {rec['type']:<10} | "
                       f"Score: {rec['score']:>8.1f} | "
                       f"Purchases: {rec['num_purchases']:>3} | "
                       f"Reason: {rec['reason']}")

        return recommendations


def main():
    """Test the frequency baseline engine"""
    engine = FrequencyBaselineEngine()

    # Test with a sample customer
    test_customer = "410187"  # Customer from our tests

    logger.info("="*80)
    logger.info("FREQUENCY BASELINE RECOMMENDATION ENGINE - TEST")
    logger.info("="*80)

    recommendations = engine.get_recommendations_with_details(test_customer, top_n=50)

    logger.info(f"\n✅ Successfully generated {len(recommendations)} recommendations")


if __name__ == "__main__":
    main()
