#!/usr/bin/env python3
"""
Train LightFM Hybrid Recommendation Model

Uses:
- User-item interactions (customer purchases)
- User features (RFM scores, customer segment)
- Item features (product popularity, sales metrics)

Model: LightFM with WARP loss (Weighted Approximate-Rank Pairwise)
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from datetime import datetime
from lightfm import LightFM
from lightfm.data import Dataset
from scipy import sparse
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DUCKDB_PATH = "/opt/dagster/app/data/dbt/concord_bi.duckdb"
MODEL_DIR = Path("/opt/dagster/app/models/lightfm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class RecommendationTrainer:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.user_features_map = {}
        self.item_features_map = {}

    def load_data_from_duckdb(self):
        """Load interaction matrix and features from DuckDB"""
        logger.info("Loading data from DuckDB...")

        conn = duckdb.connect(DUCKDB_PATH, read_only=True)

        # Load interactions
        logger.info("  Loading interaction matrix...")
        self.interactions_df = conn.execute("""
            SELECT
                customer_id,
                product_id,
                implicit_rating,
                num_purchases,
                total_spent
            FROM ml_features.interaction_matrix
        """).df()
        logger.info(f"  ✓ Loaded {len(self.interactions_df):,} interactions")

        # Load user features
        logger.info("  Loading customer features...")
        self.user_features_df = conn.execute("""
            SELECT
                customer_id,
                recency_score,
                frequency_score,
                monetary_score,
                customer_segment,
                total_orders,
                lifetime_value
            FROM ml_features.customer_features
            WHERE customer_id IN (
                SELECT DISTINCT customer_id
                FROM ml_features.interaction_matrix
            )
        """).df()
        logger.info(f"  ✓ Loaded features for {len(self.user_features_df):,} customers")

        # Load item features
        logger.info("  Loading product features...")
        self.item_features_df = conn.execute("""
            SELECT
                product_id,
                times_ordered,
                unique_customers,
                total_revenue,
                num_analogues,
                product_status,
                is_for_sale,
                is_for_web
            FROM ml_features.product_features
            WHERE product_id IN (
                SELECT DISTINCT product_id
                FROM ml_features.interaction_matrix
            )
        """).df()
        logger.info(f"  ✓ Loaded features for {len(self.item_features_df):,} products")

        conn.close()

    def prepare_lightfm_dataset(self):
        """Prepare LightFM Dataset with features"""
        logger.info("\nPreparing LightFM dataset...")

        # Create dataset
        self.dataset = Dataset()

        # Get unique users and items
        unique_users = self.interactions_df['customer_id'].unique()
        unique_items = self.interactions_df['product_id'].unique()

        logger.info(f"  Users: {len(unique_users):,}")
        logger.info(f"  Items: {len(unique_items):,}")

        # Define user features
        user_feature_names = [
            'recency_1', 'recency_2', 'recency_3', 'recency_4', 'recency_5',
            'frequency_1', 'frequency_2', 'frequency_3', 'frequency_4', 'frequency_5',
            'monetary_1', 'monetary_2', 'monetary_3', 'monetary_4', 'monetary_5',
            'segment_Champions', 'segment_Loyal', 'segment_BigSpenders',
            'segment_Potential', 'segment_Recent', 'segment_AtRisk',
            'segment_CannotLose', 'segment_Lost', 'segment_New'
        ]

        # Define item features
        item_feature_names = [
            'popularity_high', 'popularity_medium', 'popularity_low',
            'has_analogues', 'status_Active', 'status_SlowMoving',
            'status_DeadStock', 'status_NeverSold',
            'for_sale', 'for_web'
        ]

        # Fit dataset
        self.dataset.fit(
            users=unique_users,
            items=unique_items,
            user_features=user_feature_names,
            item_features=item_feature_names
        )

        # Store mappings
        self.user_id_map = self.dataset.mapping()[0]
        self.item_id_map = self.dataset.mapping()[2]

        logger.info("  ✓ Dataset prepared")

    def build_interactions_matrix(self):
        """Build interaction matrix"""
        logger.info("\nBuilding interaction matrix...")

        # Prepare interactions (user_id, item_id, weight)
        interactions = [
            (row['customer_id'], row['product_id'], row['implicit_rating'])
            for _, row in self.interactions_df.iterrows()
        ]

        # Build matrix
        self.interactions_matrix, _ = self.dataset.build_interactions(interactions)

        logger.info(f"  Matrix shape: {self.interactions_matrix.shape}")
        logger.info(f"  Density: {self.interactions_matrix.nnz / (self.interactions_matrix.shape[0] * self.interactions_matrix.shape[1]) * 100:.2f}%")
        logger.info("  ✓ Interaction matrix built")

    def build_user_features_matrix(self):
        """Build user features matrix"""
        logger.info("\nBuilding user features matrix...")

        user_features = []

        for _, row in self.user_features_df.iterrows():
            customer_id = row['customer_id']
            features = []

            # RFM scores (one-hot encoded)
            features.append(f"recency_{int(row['recency_score'])}")
            features.append(f"frequency_{int(row['frequency_score'])}")
            features.append(f"monetary_{int(row['monetary_score'])}")

            # Customer segment (simplified)
            segment = row['customer_segment']
            if 'Champion' in segment:
                features.append('segment_Champions')
            elif 'Loyal' in segment:
                features.append('segment_Loyal')
            elif 'Spender' in segment:
                features.append('segment_BigSpenders')
            elif 'Potential' in segment:
                features.append('segment_Potential')
            elif 'Recent' in segment:
                features.append('segment_Recent')
            elif 'Risk' in segment:
                features.append('segment_AtRisk')
            elif 'Cannot' in segment:
                features.append('segment_CannotLose')
            elif 'Lost' in segment:
                features.append('segment_Lost')
            else:
                features.append('segment_New')

            user_features.append((customer_id, features))

        self.user_features_matrix = self.dataset.build_user_features(user_features)

        logger.info(f"  User features shape: {self.user_features_matrix.shape}")
        logger.info("  ✓ User features built")

    def build_item_features_matrix(self):
        """Build item features matrix"""
        logger.info("\nBuilding item features matrix...")

        item_features = []

        for _, row in self.item_features_df.iterrows():
            product_id = row['product_id']
            features = []

            # Popularity tier
            if row['times_ordered'] >= 10:
                features.append('popularity_high')
            elif row['times_ordered'] >= 3:
                features.append('popularity_medium')
            else:
                features.append('popularity_low')

            # Has analogues
            if row['num_analogues'] > 0:
                features.append('has_analogues')

            # Product status
            status = row['product_status']
            if 'Active' in status:
                features.append('status_Active')
            elif 'Slow' in status:
                features.append('status_SlowMoving')
            elif 'Dead' in status:
                features.append('status_DeadStock')
            else:
                features.append('status_NeverSold')

            # Sale flags
            if row['is_for_sale']:
                features.append('for_sale')
            if row['is_for_web']:
                features.append('for_web')

            item_features.append((product_id, features))

        self.item_features_matrix = self.dataset.build_item_features(item_features)

        logger.info(f"  Item features shape: {self.item_features_matrix.shape}")
        logger.info("  ✓ Item features built")

    def train_model(self, epochs=30, num_threads=4):
        """Train LightFM model"""
        logger.info(f"\nTraining LightFM model ({epochs} epochs)...")

        # Initialize model with WARP loss (good for implicit feedback)
        self.model = LightFM(
            loss='warp',
            no_components=64,  # Embedding size
            learning_rate=0.05,
            item_alpha=0.0,
            user_alpha=0.0,
            max_sampled=10,
            random_state=42
        )

        start_time = datetime.now()

        # Train
        self.model.fit(
            interactions=self.interactions_matrix,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            epochs=epochs,
            num_threads=num_threads,
            verbose=True
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"  ✓ Training complete in {duration:.1f}s")

    def save_model(self, version="v1"):
        """Save model and mappings"""
        logger.info(f"\nSaving model (version {version})...")

        model_path = MODEL_DIR / f"recommendation_{version}.pkl"

        model_data = {
            'model': self.model,
            'dataset': self.dataset,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'user_features_df': self.user_features_df,
            'item_features_df': self.item_features_df,
            'interactions_matrix': self.interactions_matrix,
            'user_features_matrix': self.user_features_matrix,
            'item_features_matrix': self.item_features_matrix,
            'training_date': datetime.now().isoformat(),
            'num_users': len(self.user_id_map),
            'num_items': len(self.item_id_map),
            'num_interactions': len(self.interactions_df)
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"  ✓ Model saved: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")

        return model_path

def main():
    logger.info("=" * 70)
    logger.info("LIGHTFM RECOMMENDATION MODEL TRAINING")
    logger.info("=" * 70)

    trainer = RecommendationTrainer()

    # Step 1: Load data
    trainer.load_data_from_duckdb()

    # Step 2: Prepare dataset
    trainer.prepare_lightfm_dataset()

    # Step 3: Build matrices
    trainer.build_interactions_matrix()
    trainer.build_user_features_matrix()
    trainer.build_item_features_matrix()

    # Step 4: Train model
    trainer.train_model(epochs=30, num_threads=4)

    # Step 5: Save model
    model_path = trainer.save_model(version="v1")

    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\nModel saved: {model_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Evaluate model performance")
    logger.info("  2. Build API endpoint")
    logger.info("  3. Test recommendations")

    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
