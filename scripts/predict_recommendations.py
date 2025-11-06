#!/usr/bin/env python3
"""
Product Recommendation API - Ensemble Inference

Given a customer_id, returns 20 recommended products:
- 80% (16 products) = REPURCHASE (products they bought before, likely to reorder)
- 20% (4 products) = DISCOVERY (new products from collaborative filtering)

Usage:
    python3 scripts/predict_recommendations.py --customer_id 410187
    python3 scripts/predict_recommendations.py --customer_id 410187 --top_n 20
"""

import pickle
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
import argparse
import logging
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/ml_features")
MODEL_DIR_CF = Path("models/collaborative_filtering")
MODEL_DIR_SURVIVAL = Path("models/survival_analysis")

DUCKDB_PATH = DATA_DIR / "concord_ml.duckdb"
ALS_MODEL_PATH = MODEL_DIR_CF / "als_model_v2.pkl"
MAPPINGS_PATH = MODEL_DIR_CF / "id_mappings_v2.pkl"
SURVIVAL_MODEL_PATH = MODEL_DIR_SURVIVAL / "weibull_repurchase_model.pkl"


class RecommendationEngine:
    """Ensemble recommendation engine combining Survival Analysis + Collaborative Filtering"""

    def __init__(self):
        self.als_model = None
        self.survival_model = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_item_map = {}
        self.user_item_matrix = None

    def load_models(self):
        """Load both trained models and preload interaction matrix for fast inference"""
        logger.info("Loading models...")

        # Load Collaborative Filtering (ALS)
        with open(ALS_MODEL_PATH, 'rb') as f:
            self.als_model = pickle.load(f)

        with open(MAPPINGS_PATH, 'rb') as f:
            mappings = pickle.load(f)
            self.user_id_map = mappings['user_id_map']
            self.item_id_map = mappings['item_id_map']
            self.reverse_item_map = mappings['reverse_item_map']

        # Load Survival Analysis
        with open(SURVIVAL_MODEL_PATH, 'rb') as f:
            self.survival_model = pickle.load(f)

        logger.info(f"✓ ALS model loaded: {self.als_model.factors} factors")
        logger.info(f"✓ Survival model loaded: C-index {self.survival_model.concordance_index_:.4f}")

        # PERFORMANCE FIX: Preload interaction matrix on startup
        logger.info("Preloading interaction matrix for fast inference...")
        self._preload_interaction_matrix()
        logger.info(f"✓ Matrix preloaded: {self.user_item_matrix.shape} ({self.user_item_matrix.nnz:,} entries)")

    def _preload_interaction_matrix(self):
        """Preload interaction matrix from DuckDB (called once on startup)"""
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        query = """
        SELECT
            customer_id,
            product_id,
            implicit_rating
        FROM ml_features.interaction_matrix
        WHERE implicit_rating > 0
        """

        all_interactions = conn.execute(query).df()
        conn.close()

        # Map to indices
        all_interactions['user_idx'] = all_interactions['customer_id'].map(self.user_id_map)
        all_interactions['item_idx'] = all_interactions['product_id'].map(self.item_id_map)

        # Filter out rows where mapping failed
        all_interactions = all_interactions.dropna(subset=['user_idx', 'item_idx'])
        all_interactions['user_idx'] = all_interactions['user_idx'].astype(int)
        all_interactions['item_idx'] = all_interactions['item_idx'].astype(int)

        # Create sparse matrix (user x item) - CACHED
        self.user_item_matrix = csr_matrix(
            (all_interactions['implicit_rating'].values,
             (all_interactions['user_idx'].values, all_interactions['item_idx'].values)),
            shape=(len(self.user_id_map), len(self.item_id_map))
        )

    def load_user_history(self, customer_id):
        """Load customer's purchase history from DuckDB"""
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        # Get interaction matrix for this customer
        query = f"""
        SELECT
            product_id,
            num_purchases,
            total_spent,
            implicit_rating,
            days_since_last_purchase,
            purchase_span_days
        FROM ml_features.interaction_matrix
        WHERE customer_id = '{customer_id}'
        """

        interactions = conn.execute(query).df()

        # Get customer features (RFM)
        customer_query = f"""
        SELECT
            recency_score,
            frequency_score,
            monetary_score,
            customer_tier,
            customer_segment
        FROM ml_features.customer_features
        WHERE customer_id = '{customer_id}'
        """

        customer_features = conn.execute(customer_query).df()

        conn.close()

        return interactions, customer_features

    def get_repurchase_recommendations(self, customer_id, interactions, customer_features, top_n=16):
        """
        Get top N products for REPURCHASE using Survival Analysis

        Returns products customer already bought, ranked by urgency to reorder
        """
        if len(interactions) == 0:
            logger.warning(f"No purchase history for customer {customer_id}")
            return []

        predictions = []

        for idx, row in interactions.iterrows():
            # Skip products with only 1 purchase (need 2+ for interval calculation)
            if row['num_purchases'] < 2:
                continue

            # Calculate avg purchase interval
            avg_interval = row['purchase_span_days'] / (row['num_purchases'] - 1)

            # Prepare features for survival model (7 features)
            features = {
                'num_purchases': row['num_purchases'],
                'implicit_rating': row['implicit_rating'],
                'recency_score': customer_features['recency_score'].iloc[0] if len(customer_features) > 0 else 3,
                'frequency_score': customer_features['frequency_score'].iloc[0] if len(customer_features) > 0 else 3,
                'monetary_score': customer_features['monetary_score'].iloc[0] if len(customer_features) > 0 else 3,
                'total_spent_log': np.log1p(row['total_spent']),
                'avg_purchase_interval_log': np.log1p(avg_interval)
            }

            feature_df = pd.DataFrame([features])

            # Predict time to next purchase
            try:
                predicted_days = self.survival_model.predict_median(feature_df).iloc[0]
                predicted_days = min(predicted_days, 730)  # Cap at 2 years

                days_overdue = row['days_since_last_purchase'] - predicted_days

                # Priority score: Use absolute days_overdue for sorting (CRITICAL FIX)
                # Higher = more urgent (customer is more overdue)
                # Also factor in implicit_rating to prioritize high-affinity products when equally overdue
                urgency_score = days_overdue + (row['implicit_rating'] * 0.1)  # Add small rating boost

                predictions.append({
                    'product_id': row['product_id'],
                    'type': 'REPURCHASE',
                    'score': urgency_score,
                    'days_overdue': days_overdue,
                    'predicted_days_to_repurchase': predicted_days,
                    'num_purchases': row['num_purchases'],
                    'implicit_rating': row['implicit_rating'],
                    'reason': f"Reorder due ({int(days_overdue)} days overdue)" if days_overdue > 0 else f"Reorder soon ({int(abs(days_overdue))} days)"
                })
            except Exception as e:
                logger.warning(f"Prediction failed for product {row['product_id']}: {e}")
                continue

        # CRITICAL FIX: Sort by absolute days_overdue (most overdue first)
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

        # CRITICAL FIX: Deduplicate by product_id (keep first occurrence = most urgent)
        seen_products = set()
        deduplicated = []
        for pred in predictions:
            if pred['product_id'] not in seen_products:
                deduplicated.append(pred)
                seen_products.add(pred['product_id'])

        return deduplicated[:top_n]

    def get_discovery_recommendations(self, customer_id, purchased_product_ids, top_n=4):
        """
        Get top N NEW products using Collaborative Filtering (ALS)

        Returns products customer has NOT bought yet, based on similar customers
        """
        # Check if customer exists in training data
        if customer_id not in self.user_id_map:
            logger.warning(f"Customer {customer_id} not in ALS training data (cold start)")
            # Use similar customers approach instead of popular
            return self._get_similar_customer_products(purchased_product_ids, top_n)

        user_idx = self.user_id_map[customer_id]

        # Matrix already preloaded on startup (performance fix)
        # Get ALS recommendations using item similarity
        try:
            # Get user's current preferences (user vector)
            user_items = self.user_item_matrix[user_idx]

            # Recommend items (model was trained on item x user, so it handles this correctly)
            item_ids, scores = self.als_model.recommend(
                user_idx,
                user_items,
                N=min(100, len(self.item_id_map)),  # Get many candidates
                filter_already_liked_items=False  # We'll filter manually
            )

            recommendations = []
            seen_products = set()  # CRITICAL FIX: Track seen products
            for item_idx, score in zip(item_ids, scores):
                product_id = self.reverse_item_map[item_idx]

                # Only include products they HAVEN'T bought AND not already recommended
                if product_id not in purchased_product_ids and product_id not in seen_products:
                    recommendations.append({
                        'product_id': product_id,
                        'type': 'DISCOVERY',
                        'score': float(score),
                        'days_overdue': None,
                        'predicted_days_to_repurchase': None,
                        'num_purchases': 0,
                        'implicit_rating': None,
                        'reason': f"ML predicted based on similar buyers (confidence: {score:.2f})"
                    })
                    seen_products.add(product_id)

                if len(recommendations) >= top_n:
                    break

            # If we still don't have enough, use similar customers
            if len(recommendations) < top_n:
                logger.info(f"   Only found {len(recommendations)} ALS recs, filling with similar customers...")
                similar_recs = self._get_similar_customer_products(
                    purchased_product_ids | {r['product_id'] for r in recommendations},
                    top_n - len(recommendations)
                )
                recommendations.extend(similar_recs)

            return recommendations[:top_n]

        except Exception as e:
            logger.warning(f"ALS recommendation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to similar customers (still ML-based)
            return self._get_similar_customer_products(purchased_product_ids, top_n)

    def _get_similar_customer_products(self, exclude_product_ids, top_n=4):
        """
        ML-based fallback: Find products from similar customers using ALS user embeddings

        Uses cosine similarity of user vectors to find similar customers,
        then recommends products those customers bought.
        """
        try:
            # Get user factors (embeddings) from ALS model
            user_factors = self.als_model.user_factors  # Shape: (num_users, num_factors)
            item_factors = self.als_model.item_factors  # Shape: (num_items, num_factors)

            # For cold start, use item-item similarity instead
            # Find items similar to items in exclude_product_ids (if any)
            if len(exclude_product_ids) > 0:
                # Get item indices for excluded products
                exclude_indices = []
                for prod_id in list(exclude_product_ids)[:10]:  # Limit to 10 for performance
                    if prod_id in self.item_id_map:
                        exclude_indices.append(self.item_id_map[prod_id])

                if exclude_indices:
                    # Compute item-item similarity using item factors
                    # Get average vector of excluded items
                    avg_vector = np.mean(item_factors[exclude_indices], axis=0).reshape(1, -1)

                    # Find similar items
                    similarities = cosine_similarity(avg_vector, item_factors)[0]

                    # Get top similar items
                    similar_indices = np.argsort(similarities)[::-1]

                    recommendations = []
                    for item_idx in similar_indices:
                        product_id = self.reverse_item_map[item_idx]

                        if product_id not in exclude_product_ids:
                            recommendations.append({
                                'product_id': product_id,
                                'type': 'DISCOVERY',
                                'score': float(similarities[item_idx]),
                                'days_overdue': None,
                                'predicted_days_to_repurchase': None,
                                'num_purchases': 0,
                                'implicit_rating': None,
                                'reason': f"ML predicted from similar products (similarity: {similarities[item_idx]:.2f})"
                            })

                        if len(recommendations) >= top_n:
                            break

                    if len(recommendations) >= top_n:
                        return recommendations[:top_n]

            # If still not enough, get popular products as last resort
            return self._get_popular_products(exclude_product_ids, top_n)

        except Exception as e:
            logger.warning(f"Similar customer recommendation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_popular_products(exclude_product_ids, top_n)

    def _get_popular_products(self, exclude_product_ids, top_n=4):
        """Last resort fallback: Get popular products (for cold start users)"""
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        # Get most popular products overall
        if len(exclude_product_ids) > 0:
            exclude_list = "','".join(str(p) for p in exclude_product_ids)
            where_clause = f"WHERE product_id NOT IN ('{exclude_list}')"
        else:
            where_clause = ""

        query = f"""
        SELECT
            product_id,
            COUNT(DISTINCT customer_id) as customer_count,
            SUM(num_purchases) as total_purchases
        FROM ml_features.interaction_matrix
        {where_clause}
        GROUP BY product_id
        ORDER BY total_purchases DESC
        LIMIT {top_n}
        """

        popular = conn.execute(query).df()
        conn.close()

        recommendations = []
        for idx, row in popular.iterrows():
            recommendations.append({
                'product_id': row['product_id'],
                'type': 'POPULAR',
                'score': float(row['total_purchases']),
                'days_overdue': None,
                'predicted_days_to_repurchase': None,
                'num_purchases': 0,
                'implicit_rating': None,
                'reason': f"Popular fallback ({int(row['customer_count'])} customers)"
            })

        return recommendations

    def get_recommendations(self, customer_id, top_n=20, repurchase_ratio=0.8):
        """
        Get ensemble recommendations for a customer

        Args:
            customer_id: Customer identifier
            top_n: Total number of recommendations (default: 20)
            repurchase_ratio: % of repurchase vs discovery (default: 0.8 = 80% repurchase)

        Returns:
            List of recommended products with metadata
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Getting recommendations for customer: {customer_id}")
        logger.info(f"{'='*80}")

        # Calculate split
        num_repurchase = int(top_n * repurchase_ratio)
        num_discovery = top_n - num_repurchase

        logger.info(f"Target: {num_repurchase} repurchase + {num_discovery} discovery = {top_n} total")

        # Load customer history
        interactions, customer_features = self.load_user_history(customer_id)

        if len(interactions) == 0:
            logger.warning("No purchase history - returning popular products only")
            return self._get_popular_products([], top_n)

        purchased_product_ids = set(interactions['product_id'].tolist())

        logger.info(f"\nCustomer profile:")
        if len(customer_features) > 0:
            logger.info(f"  Tier: {customer_features['customer_tier'].iloc[0]}")
            logger.info(f"  Segment: {customer_features['customer_segment'].iloc[0]}")
        logger.info(f"  Products purchased: {len(purchased_product_ids)}")

        # Get repurchase recommendations (products they already bought)
        logger.info(f"\n1. Generating {num_repurchase} REPURCHASE recommendations...")
        repurchase_recs = self.get_repurchase_recommendations(
            customer_id, interactions, customer_features, top_n=num_repurchase
        )
        logger.info(f"   ✓ Found {len(repurchase_recs)} repurchase candidates")

        # Get discovery recommendations (new products)
        logger.info(f"\n2. Generating {num_discovery} DISCOVERY recommendations...")
        discovery_recs = self.get_discovery_recommendations(
            customer_id, purchased_product_ids, top_n=num_discovery
        )
        logger.info(f"   ✓ Found {len(discovery_recs)} discovery candidates")

        # Combine
        all_recommendations = repurchase_recs + discovery_recs

        # If we don't have enough, fill with popular products
        if len(all_recommendations) < top_n:
            logger.info(f"\n3. Filling remaining slots with popular products...")
            remaining = top_n - len(all_recommendations)
            popular_recs = self._get_popular_products(
                purchased_product_ids | {r['product_id'] for r in all_recommendations},
                top_n=remaining
            )
            all_recommendations.extend(popular_recs)

        # CRITICAL FIX: Final deduplication safety check (keep first occurrence)
        seen_final = set()
        deduplicated_final = []
        for rec in all_recommendations:
            if rec['product_id'] not in seen_final:
                deduplicated_final.append(rec)
                seen_final.add(rec['product_id'])
        all_recommendations = deduplicated_final

        # Add rank
        for idx, rec in enumerate(all_recommendations[:top_n], 1):
            rec['rank'] = idx

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Generated {len(all_recommendations[:top_n])} recommendations")
        logger.info(f"{'='*80}")

        return all_recommendations[:top_n]


def main():
    parser = argparse.ArgumentParser(description='Get product recommendations for a customer')
    parser.add_argument('--customer_id', type=str, required=True, help='Customer ID')
    parser.add_argument('--top_n', type=int, default=20, help='Number of recommendations (default: 20)')
    parser.add_argument('--repurchase_ratio', type=float, default=0.8, help='Repurchase ratio (default: 0.8 = 80%)')
    parser.add_argument('--output', type=str, help='Output CSV file (optional)')

    args = parser.parse_args()

    # Initialize engine
    engine = RecommendationEngine()
    engine.load_models()

    # Get recommendations
    recommendations = engine.get_recommendations(
        customer_id=args.customer_id,
        top_n=args.top_n,
        repurchase_ratio=args.repurchase_ratio
    )

    # Display results
    print(f"\n{'='*80}")
    print(f"TOP {args.top_n} RECOMMENDATIONS FOR CUSTOMER {args.customer_id}")
    print(f"{'='*80}\n")

    df = pd.DataFrame(recommendations)

    # Print formatted table
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx:2d}. Product: {rec['product_id']:12s} | Type: {rec['type']:10s} | {rec['reason']}")

    # Summary by type
    print(f"\n{'='*80}")
    print("SUMMARY BY TYPE")
    print(f"{'='*80}")
    type_counts = df['type'].value_counts()
    for rec_type, count in type_counts.items():
        print(f"  {rec_type}: {count}")

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
