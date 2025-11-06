#!/usr/bin/env python3
"""
Train Collaborative Filtering Model V2 (Implicit ALS)

Using implicit library instead of LightFM:
- Better for implicit feedback (purchases, not ratings)
- CPU-optimized (faster than LightFM on CPU)
- Simpler API, proven for production

Trains on 183K interactions from DuckDB.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/ml_features")
MODEL_DIR = Path("models/collaborative_filtering")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = DATA_DIR / "concord_ml.duckdb"
MODEL_PATH = MODEL_DIR / "als_model_v2.pkl"
MAPPINGS_PATH = MODEL_DIR / "id_mappings_v2.pkl"


class CollaborativeFilteringTrainer:
    """Train ALS Collaborative Filtering model"""

    def __init__(self, factors=100, regularization=0.01, iterations=15, alpha=40):
        """
        Initialize ALS model parameters

        Args:
            factors: Number of latent factors (embeddings dimension)
            regularization: L2 regularization
            iterations: Number of ALS iterations
            alpha: Confidence scaling factor for implicit feedback
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            use_gpu=False,  # CPU-only
            num_threads=0  # Use all CPU cores
        )

        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

    def load_data(self):
        """Load interaction matrix from DuckDB"""
        logger.info("\n" + "="*80)
        logger.info("LOADING DATA FROM DUCKDB")
        logger.info("="*80)

        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        # Load interaction matrix
        query = """
        SELECT
            customer_id,
            product_id,
            implicit_rating as rating,
            num_purchases,
            total_spent
        FROM ml_features.interaction_matrix
        WHERE implicit_rating > 0
        """

        logger.info(f"Loading from {DUCKDB_PATH}...")
        df = conn.execute(query).df()
        conn.close()

        logger.info(f"✓ Loaded {len(df):,} interactions")
        logger.info(f"  Unique customers: {df['customer_id'].nunique():,}")
        logger.info(f"  Unique products: {df['product_id'].nunique():,}")
        logger.info(f"  Mean rating: {df['rating'].mean():.2f}")
        logger.info(f"  Rating range: [{df['rating'].min():.2f}, {df['rating'].max():.2f}]")

        return df

    def prepare_sparse_matrix(self, df):
        """Convert interactions to sparse matrix"""
        logger.info("\n" + "="*80)
        logger.info("PREPARING SPARSE MATRIX")
        logger.info("="*80)

        # Create user and item mappings
        unique_users = df['customer_id'].unique()
        unique_items = df['product_id'].unique()

        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}

        logger.info(f"Created mappings:")
        logger.info(f"  Users: {len(self.user_id_map):,}")
        logger.info(f"  Items: {len(self.item_id_map):,}")

        # Map to indices
        df['user_idx'] = df['customer_id'].map(self.user_id_map)
        df['item_idx'] = df['product_id'].map(self.item_id_map)

        # Create sparse matrix (users x items)
        # Using implicit ratings as confidence weights
        user_item_matrix = csr_matrix(
            (df['rating'].values, (df['user_idx'].values, df['item_idx'].values)),
            shape=(len(self.user_id_map), len(self.item_id_map))
        )

        logger.info(f"✓ Created sparse matrix:")
        logger.info(f"  Shape: {user_item_matrix.shape}")
        logger.info(f"  Density: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.4f}%")
        logger.info(f"  Non-zero entries: {user_item_matrix.nnz:,}")

        return user_item_matrix, df

    def train_test_split_data(self, df, test_size=0.2):
        """Split data into train/test sets"""
        logger.info("\n" + "="*80)
        logger.info("CREATING TRAIN/TEST SPLIT")
        logger.info("="*80)

        # Split by interactions (not by users)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        logger.info(f"Train set: {len(train_df):,} interactions")
        logger.info(f"Test set: {len(test_df):,} interactions")

        # Create train matrix
        train_matrix = csr_matrix(
            (train_df['rating'].values, (train_df['user_idx'].values, train_df['item_idx'].values)),
            shape=(len(self.user_id_map), len(self.item_id_map))
        )

        # Create test matrix
        test_matrix = csr_matrix(
            (test_df['rating'].values, (test_df['user_idx'].values, test_df['item_idx'].values)),
            shape=(len(self.user_id_map), len(self.item_id_map))
        )

        return train_matrix, test_matrix

    def train(self, train_matrix):
        """Train ALS model"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING ALS MODEL")
        logger.info("="*80)
        logger.info(f"Model parameters:")
        logger.info(f"  Factors: {self.model.factors}")
        logger.info(f"  Regularization: {self.model.regularization}")
        logger.info(f"  Iterations: {self.model.iterations}")
        logger.info(f"  Alpha: {self.model.alpha}")

        start_time = datetime.now()
        logger.info(f"\nTraining started at {start_time}...")

        # Fit model (implicit expects item x user matrix)
        self.model.fit(train_matrix.T)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ Training complete in {duration:.1f} seconds")

    def evaluate(self, train_matrix, test_matrix):
        """Evaluate model performance (manual evaluation)"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING MODEL")
        logger.info("="*80)

        # Manual precision@K evaluation
        k_values = [10, 20, 50]

        for k in k_values:
            logger.info(f"\n--- Metrics @ K={k} ---")

            # Simple precision@K: sample 100 users, check if test items are in top-K recs
            sample_size = min(100, len(self.user_id_map))
            sample_users = np.random.choice(len(self.user_id_map), sample_size, replace=False)

            precisions = []

            for user_idx in sample_users:
                try:
                    # Get test items for this user
                    test_items = set(test_matrix[user_idx].indices)

                    if len(test_items) == 0:
                        continue

                    # Get top-K recommendations
                    item_ids, scores = self.model.recommend(user_idx, train_matrix[user_idx], N=k)
                    recommended_set = set(item_ids)

                    # Calculate precision
                    hits = len(recommended_set & test_items)
                    precision = hits / k if k > 0 else 0
                    precisions.append(precision)

                except Exception:
                    continue

            if precisions:
                avg_precision = np.mean(precisions)
                logger.info(f"Precision@{k}: {avg_precision:.4f} (sampled {len(precisions)} users)")
            else:
                logger.info(f"Precision@{k}: Unable to calculate")

        # Coverage (what % of items can we recommend?)
        logger.info(f"\n--- Coverage ---")
        num_items = len(self.item_id_map)

        # Sample recommendations for 100 users
        sample_users = np.random.choice(len(self.user_id_map), min(100, len(self.user_id_map)), replace=False)
        recommended_items = set()

        for user_idx in sample_users:
            try:
                # Get recommendations
                item_ids, scores = self.model.recommend(user_idx, train_matrix[user_idx], N=50)
                recommended_items.update(item_ids)
            except Exception:
                continue

        coverage = len(recommended_items) / num_items
        logger.info(f"Coverage (100 users, top-50 recs): {coverage*100:.2f}% ({len(recommended_items):,} / {num_items:,} items)")

    def save_model(self):
        """Save trained model and mappings"""
        logger.info("\n" + "="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)

        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"✓ Model saved to {MODEL_PATH}")

        # Save ID mappings
        mappings = {
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map
        }
        with open(MAPPINGS_PATH, 'wb') as f:
            pickle.dump(mappings, f)
        logger.info(f"✓ Mappings saved to {MAPPINGS_PATH}")

    def get_recommendations(self, customer_id, train_matrix, n=20):
        """Get recommendations for a customer"""
        if customer_id not in self.user_id_map:
            logger.warning(f"Customer {customer_id} not in training data")
            return []

        user_idx = self.user_id_map[customer_id]

        # Get recommendations
        item_indices, scores = self.model.recommend(
            user_idx,
            train_matrix[user_idx],
            N=n,
            filter_already_liked_items=True
        )

        # Convert back to product IDs
        recommendations = [
            {
                'product_id': self.reverse_item_map[item_idx],
                'score': float(score)
            }
            for item_idx, score in zip(item_indices, scores)
        ]

        return recommendations

    def run(self):
        """Main training pipeline"""
        try:
            start_time = datetime.now()
            logger.info("\n" + "="*80)
            logger.info("COLLABORATIVE FILTERING TRAINING - V2 (ALS)")
            logger.info("="*80)
            logger.info(f"Start time: {start_time}")

            # Load data
            df = self.load_data()

            # Prepare sparse matrix
            full_matrix, df = self.prepare_sparse_matrix(df)

            # Train/test split
            train_matrix, test_matrix = self.train_test_split_data(df)

            # Train model
            self.train(train_matrix)

            # Evaluate
            self.evaluate(train_matrix, test_matrix)

            # Save model
            self.save_model()

            # Demo: Get recommendations for top customer
            logger.info("\n" + "="*80)
            logger.info("DEMO RECOMMENDATIONS")
            logger.info("="*80)
            sample_customer = df['customer_id'].iloc[0]
            logger.info(f"Getting recommendations for customer: {sample_customer}")

            recs = self.get_recommendations(sample_customer, train_matrix, n=10)
            logger.info(f"Top 10 recommendations:")
            for i, rec in enumerate(recs, 1):
                logger.info(f"  {i}. Product {rec['product_id']} (score: {rec['score']:.3f})")

            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "="*80)
            logger.info("✅ TRAINING COMPLETE!")
            logger.info("="*80)
            logger.info(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"\nModel files:")
            logger.info(f"  {MODEL_PATH}")
            logger.info(f"  {MAPPINGS_PATH}")
            logger.info(f"\nNext step: Deploy to FastAPI for serving!")

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    trainer = CollaborativeFilteringTrainer(
        factors=100,  # Latent dimensions
        regularization=0.01,
        iterations=15,
        alpha=40  # Confidence scaling for implicit feedback
    )

    success = trainer.run()
    sys.exit(0 if success else 1)
