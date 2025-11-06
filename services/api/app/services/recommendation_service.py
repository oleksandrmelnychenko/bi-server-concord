"""
Recommendation Service - Handles product recommendations using LightFM
"""
import logging
from typing import List, Optional
from datetime import datetime
import pickle
import numpy as np
from pathlib import Path
import duckdb

from app.config import settings
from app.schemas.recommendations import ProductRecommendation

logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "/opt/dagster/app/models/lightfm/recommendation_v1.pkl"
DUCKDB_PATH = "/opt/dagster/app/data/dbt/concord_bi.duckdb"


class RecommendationService:
    """Service for generating product recommendations using LightFM"""

    def __init__(self):
        self.model_version = "v1"
        self.model = None
        self.dataset = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_item_map = {}
        self.user_features_matrix = None
        self.item_features_matrix = None
        self._load_model()

    def _load_model(self):
        """Load LightFM recommendation model"""
        try:
            logger.info(f"Loading LightFM model from {MODEL_PATH}...")

            model_path = Path(MODEL_PATH)
            if not model_path.exists():
                logger.warning(f"Model file not found: {MODEL_PATH}")
                return

            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.dataset = model_data['dataset']
            self.user_id_map = model_data['user_id_map']
            self.item_id_map = model_data['item_id_map']
            self.user_features_matrix = model_data.get('user_features_matrix')
            self.item_features_matrix = model_data.get('item_features_matrix')

            # Create reverse item map
            self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}

            logger.info(f"✓ LightFM model loaded successfully")
            logger.info(f"  Users: {len(self.user_id_map):,}")
            logger.info(f"  Items: {len(self.item_id_map):,}")

        except Exception as e:
            logger.error(f"Could not load LightFM model: {e}", exc_info=True)
            self.model = None

    async def get_recommendations(
        self,
        customer_id: str,
        n_recommendations: int = 10,
        exclude_purchased: bool = True,
        category_filter: Optional[str] = None
    ) -> List[ProductRecommendation]:
        """
        Generate product recommendations for a customer using LightFM

        Args:
            customer_id: Customer identifier
            n_recommendations: Number of recommendations to return
            exclude_purchased: Whether to exclude already purchased products
            category_filter: Optional category filter (not yet implemented)

        Returns:
            List of product recommendations
        """
        try:
            logger.info(f"Generating {n_recommendations} recommendations for customer {customer_id}")

            if self.model is None:
                logger.warning("Model not loaded, returning empty recommendations")
                return []

            # Check if customer exists
            if customer_id not in self.user_id_map:
                logger.info(f"Customer {customer_id} not in training data, returning popular products")
                return await self._get_popular_products(n_recommendations, category_filter)

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
            conn = duckdb.connect(DUCKDB_PATH, read_only=True)

            # Get products
            products_df = conn.execute("""
                SELECT
                    product_id,
                    product_name,
                    current_price,
                    num_analogues,
                    times_ordered,
                    product_status
                FROM ml_features.product_features
            """).df()

            # Filter: exclude purchased
            if exclude_purchased:
                purchased_df = conn.execute(f"""
                    SELECT DISTINCT product_id
                    FROM ml_features.interaction_matrix
                    WHERE customer_id = '{customer_id}'
                """).df()

                if len(purchased_df) > 0:
                    purchased_products = purchased_df['product_id'].tolist()
                    for internal_id, product_id in self.reverse_item_map.items():
                        if product_id in purchased_products:
                            scores[internal_id] = -np.inf

            conn.close()

            # Get top N recommendations
            top_indices = np.argsort(-scores)[:n_recommendations * 2]

            # Build recommendations list
            recommendations = []
            for internal_id in top_indices:
                if len(recommendations) >= n_recommendations:
                    break

                if internal_id not in self.reverse_item_map:
                    continue

                product_id = self.reverse_item_map[internal_id]
                score = float(scores[internal_id])

                if score == -np.inf:
                    continue

                # Get product details
                product_row = products_df[products_df['product_id'] == product_id]

                if len(product_row) == 0:
                    continue

                product_row = product_row.iloc[0]

                # Normalize score to 0-1 range
                normalized_score = min(1.0, max(0.0, (score + 5) / 10))

                recommendations.append(
                    ProductRecommendation(
                        product_id=product_id,
                        product_name=product_row['product_name'],
                        score=normalized_score,
                        category=None,  # TODO: Add category from product data
                        price=float(product_row['current_price']) if product_row['current_price'] else None,
                        reason="Personalized based on your purchase history"
                    )
                )

            logger.info(f"Generated {len(recommendations)} recommendations for {customer_id}")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            raise

    async def _get_popular_products(
        self,
        n_recommendations: int,
        category_filter: Optional[str] = None
    ) -> List[ProductRecommendation]:
        """
        Fallback: Get popular products for customers not in training data

        Args:
            n_recommendations: Number of products to return
            category_filter: Optional category filter

        Returns:
            List of popular products
        """
        try:
            conn = duckdb.connect(DUCKDB_PATH, read_only=True)

            query = """
                SELECT
                    product_id,
                    product_name,
                    current_price,
                    times_ordered,
                    product_status
                FROM ml_features.product_features
                WHERE times_ordered > 0
                    AND product_status = 'Active'
                ORDER BY times_ordered DESC, total_revenue DESC
                LIMIT ?
            """

            products_df = conn.execute(query, [n_recommendations]).df()
            conn.close()

            recommendations = []
            for _, row in products_df.iterrows():
                recommendations.append(
                    ProductRecommendation(
                        product_id=row['product_id'],
                        product_name=row['product_name'],
                        score=min(1.0, float(row['times_ordered']) / 50.0),  # Normalize
                        category=None,
                        price=float(row['current_price']) if row['current_price'] else None,
                        reason="Popular product"
                    )
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            return []

    async def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_name": "LightFM",
            "model_version": self.model_version,
            "model_loaded": self.model is not None,
            "num_users": len(self.user_id_map),
            "num_items": len(self.item_id_map),
            "algorithm": "WARP (Weighted Approximate-Rank Pairwise)",
            "features": "Hybrid (user + item features)"
        }

    @staticmethod
    def get_current_timestamp() -> datetime:
        """Get current UTC timestamp"""
        return datetime.utcnow()
