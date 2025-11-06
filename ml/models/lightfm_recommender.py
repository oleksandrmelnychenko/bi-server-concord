"""
LightFM Recommender Model Wrapper

Provides easy-to-use interface for product recommendations
"""

import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
import duckdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightFMRecommender:
    """Wrapper for LightFM recommendation model"""

    def __init__(self, model_path: str, duckdb_path: str):
        """
        Initialize recommender

        Args:
            model_path: Path to pickled LightFM model
            duckdb_path: Path to DuckDB database with product info
        """
        self.model_path = model_path
        self.duckdb_path = duckdb_path
        self.model = None
        self.dataset = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_item_map = {}
        self.user_features_matrix = None
        self.item_features_matrix = None

        self._load_model()

    def _load_model(self):
        """Load model from disk"""
        logger.info(f"Loading recommendation model from {self.model_path}...")

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.dataset = model_data['dataset']
        self.user_id_map = model_data['user_id_map']
        self.item_id_map = model_data['item_id_map']
        self.user_features_matrix = model_data['user_features_matrix']
        self.item_features_matrix = model_data['item_features_matrix']

        # Create reverse item map (internal_id -> product_id)
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Users: {len(self.user_id_map):,}")
        logger.info(f"  Items: {len(self.item_id_map):,}")

    def recommend(
        self,
        customer_id: str,
        num_recommendations: int = 20,
        in_stock_only: bool = False,
        exclude_purchased: bool = False
    ) -> List[Dict]:
        """
        Get product recommendations for a customer

        Args:
            customer_id: Customer ID
            num_recommendations: Number of products to recommend (default: 20)
            in_stock_only: Filter to only active products (default: False)
            exclude_purchased: Exclude already purchased products (default: False)

        Returns:
            List of recommended products with scores and metadata
        """
        # Check if customer exists
        if customer_id not in self.user_id_map:
            logger.warning(f"Customer {customer_id} not found in training data")
            return self._recommend_popular(num_recommendations, in_stock_only)

        # Get internal user ID
        user_internal_id = self.user_id_map[customer_id]

        # Get all items
        n_items = len(self.item_id_map)
        all_item_ids = np.arange(n_items)

        # Get predictions
        scores = self.model.predict(
            user_internal_id,
            all_item_ids,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix
        )

        # Get product info from DuckDB
        conn = duckdb.connect(self.duckdb_path, read_only=True)

        # Get products
        products_df = conn.execute("""
            SELECT
                product_id,
                product_name,
                current_price,
                num_analogues,
                times_ordered,
                product_status,
                is_for_sale,
                is_for_web
            FROM ml_features.product_features
        """).df()

        # Filter: in-stock only (Active products)
        if in_stock_only:
            active_products = products_df[products_df['product_status'] == 'Active']['product_id'].tolist()
            # Mask out non-active products
            for internal_id, product_id in self.reverse_item_map.items():
                if product_id not in active_products:
                    scores[internal_id] = -np.inf

        # Filter: exclude purchased
        if exclude_purchased:
            # Get customer's purchase history
            purchased_df = conn.execute(f"""
                SELECT DISTINCT product_id
                FROM ml_features.interaction_matrix
                WHERE customer_id = '{customer_id}'
            """).df()

            if len(purchased_df) > 0:
                purchased_products = purchased_df['product_id'].tolist()
                # Mask out purchased products
                for internal_id, product_id in self.reverse_item_map.items():
                    if product_id in purchased_products:
                        scores[internal_id] = -np.inf

        conn.close()

        # Get top N recommendations
        top_indices = np.argsort(-scores)[:num_recommendations * 2]  # Get 2x in case some filtered out

        # Build recommendations list
        recommendations = []
        for internal_id in top_indices:
            if len(recommendations) >= num_recommendations:
                break

            if internal_id not in self.reverse_item_map:
                continue

            product_id = self.reverse_item_map[internal_id]
            score = float(scores[internal_id])

            # Skip if score is -inf (filtered out)
            if score == -np.inf:
                continue

            # Get product details
            product_row = products_df[products_df['product_id'] == product_id]

            if len(product_row) == 0:
                continue

            product_row = product_row.iloc[0]

            recommendations.append({
                'product_id': product_id,
                'product_name': product_row['product_name'],
                'score': score,
                'reason': 'personalized',
                'metadata': {
                    'price': float(product_row['current_price']) if product_row['current_price'] else None,
                    'num_analogues': int(product_row['num_analogues']),
                    'times_ordered': int(product_row['times_ordered']),
                    'product_status': product_row['product_status']
                }
            })

        return recommendations

    def _recommend_popular(self, num_recommendations: int, in_stock_only: bool) -> List[Dict]:
        """
        Fallback: Recommend popular products (for customers not in training data)

        Args:
            num_recommendations: Number of products to recommend
            in_stock_only: Filter to only active products

        Returns:
            List of popular products
        """
        conn = duckdb.connect(self.duckdb_path, read_only=True)

        query = """
            SELECT
                product_id,
                product_name,
                current_price,
                num_analogues,
                times_ordered,
                product_status,
                is_for_sale,
                is_for_web
            FROM ml_features.product_features
            WHERE times_ordered > 0
        """

        if in_stock_only:
            query += " AND product_status = 'Active'"

        query += f"""
            ORDER BY times_ordered DESC, total_revenue DESC
            LIMIT {num_recommendations}
        """

        products_df = conn.execute(query).df()
        conn.close()

        recommendations = []
        for _, row in products_df.iterrows():
            recommendations.append({
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'score': float(row['times_ordered']) / 100.0,  # Normalize to 0-1 range
                'reason': 'popular',
                'metadata': {
                    'price': float(row['current_price']) if row['current_price'] else None,
                    'num_analogues': int(row['num_analogues']),
                    'times_ordered': int(row['times_ordered']),
                    'product_status': row['product_status']
                }
            })

        return recommendations

    def get_customer_info(self, customer_id: str) -> Optional[Dict]:
        """
        Get customer information

        Args:
            customer_id: Customer ID

        Returns:
            Customer info dict or None if not found
        """
        conn = duckdb.connect(self.duckdb_path, read_only=True)

        result = conn.execute(f"""
            SELECT
                customer_id,
                customer_name,
                customer_segment,
                rfm_segment,
                total_orders,
                lifetime_value,
                customer_status
            FROM ml_features.customer_features
            WHERE customer_id = '{customer_id}'
        """).df()

        conn.close()

        if len(result) == 0:
            return None

        row = result.iloc[0]
        return {
            'customer_id': row['customer_id'],
            'customer_name': row['customer_name'],
            'customer_segment': row['customer_segment'],
            'rfm_segment': row['rfm_segment'],
            'total_orders': int(row['total_orders']),
            'lifetime_value': float(row['lifetime_value']),
            'customer_status': row['customer_status']
        }
