"""
Product Recommendation Model Training

Trains a collaborative filtering model using LightFM for product recommendations.
"""
import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from lightfm.data import Dataset
import mlflow
import mlflow.pyfunc
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for LightFM recommendation model"""

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def predict(self, context, model_input):
        """
        Generate recommendations for customers

        Args:
            model_input: DataFrame with 'customer_id' and 'n_recommendations' columns

        Returns:
            DataFrame with recommendations
        """
        results = []

        for _, row in model_input.iterrows():
            customer_id = row['customer_id']
            n_recommendations = row.get('n_recommendations', 10)

            # Get user internal ID
            try:
                user_id_internal = self.dataset.mapping()[0][customer_id]
            except KeyError:
                logger.warning(f"Unknown customer: {customer_id}")
                continue

            # Get all items
            n_items = self.dataset.interactions_shape()[1]

            # Predict scores
            scores = self.model.predict(user_id_internal, np.arange(n_items))

            # Get top N
            top_items = np.argsort(-scores)[:n_recommendations]

            # Map back to external IDs
            item_mapping_reverse = {v: k for k, v in self.dataset.mapping()[2].items()}

            recommendations = []
            for item_id in top_items:
                if item_id in item_mapping_reverse:
                    recommendations.append({
                        'product_id': item_mapping_reverse[item_id],
                        'score': float(scores[item_id])
                    })

            results.append({
                'customer_id': customer_id,
                'recommendations': recommendations
            })

        return pd.DataFrame(results)


def load_interaction_data() -> pd.DataFrame:
    """
    Load customer-product interaction data

    Returns:
        DataFrame with customer_id, product_id, interaction_weight columns
    """
    # TODO: Load from PostgreSQL/Delta Lake
    # For now, generate synthetic data

    logger.info("Loading interaction data...")

    # Synthetic data
    np.random.seed(42)
    n_customers = 1000
    n_products = 500
    n_interactions = 10000

    data = {
        'customer_id': [f'CUST-{i:04d}' for i in np.random.randint(0, n_customers, n_interactions)],
        'product_id': [f'PROD-{i:04d}' for i in np.random.randint(0, n_products, n_interactions)],
        'interaction_weight': np.random.randint(1, 6, n_interactions)  # 1-5 rating or purchase count
    }

    df = pd.DataFrame(data)

    # Aggregate multiple interactions
    df = df.groupby(['customer_id', 'product_id'])['interaction_weight'].sum().reset_index()

    logger.info(f"Loaded {len(df)} interactions")

    return df


def create_train_test_split(
    interactions: pd.DataFrame,
    test_percentage: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train and test sets

    Args:
        interactions: Interaction data
        test_percentage: Percentage for test set

    Returns:
        Train and test DataFrames
    """
    # Random split
    test_idx = np.random.rand(len(interactions)) < test_percentage

    train = interactions[~test_idx]
    test = interactions[test_idx]

    logger.info(f"Train: {len(train)}, Test: {len(test)}")

    return train, test


def train_model(
    train_interactions: pd.DataFrame,
    n_components: int = 64,
    loss: str = 'warp',
    epochs: int = 30
) -> Tuple[LightFM, Dataset]:
    """
    Train LightFM recommendation model

    Args:
        train_interactions: Training data
        n_components: Number of latent dimensions
        loss: Loss function ('warp', 'bpr', 'logistic')
        epochs: Number of training epochs

    Returns:
        Trained model and dataset
    """
    logger.info("Training recommendation model...")

    # Create dataset
    dataset = Dataset()
    dataset.fit(
        users=train_interactions['customer_id'].unique(),
        items=train_interactions['product_id'].unique()
    )

    # Build interaction matrix
    (interactions, weights) = dataset.build_interactions(
        ((row['customer_id'], row['product_id'], row['interaction_weight'])
         for _, row in train_interactions.iterrows())
    )

    logger.info(f"Interaction matrix shape: {interactions.shape}")

    # Initialize model
    model = LightFM(
        no_components=n_components,
        loss=loss,
        random_state=42
    )

    # Train
    logger.info(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.fit_partial(interactions, sample_weight=weights, epochs=1)

        if (epoch + 1) % 10 == 0:
            train_precision = precision_at_k(model, interactions, k=10).mean()
            logger.info(f"Epoch {epoch + 1}: Precision@10 = {train_precision:.4f}")

    logger.info("Training complete!")

    return model, dataset


def evaluate_model(
    model: LightFM,
    test_interactions: pd.DataFrame,
    dataset: Dataset
) -> Dict[str, float]:
    """
    Evaluate model performance

    Args:
        model: Trained model
        test_interactions: Test data
        dataset: Dataset object

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")

    # Build test interaction matrix
    (test_matrix, _) = dataset.build_interactions(
        ((row['customer_id'], row['product_id'], row['interaction_weight'])
         for _, row in test_interactions.iterrows())
    )

    # Calculate metrics
    metrics = {
        'precision_at_5': precision_at_k(model, test_matrix, k=5).mean(),
        'precision_at_10': precision_at_k(model, test_matrix, k=10).mean(),
        'recall_at_5': recall_at_k(model, test_matrix, k=5).mean(),
        'recall_at_10': recall_at_k(model, test_matrix, k=10).mean(),
        'auc': auc_score(model, test_matrix).mean()
    }

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics


def main():
    """Main training pipeline"""

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

    # Create experiment
    experiment_name = "product-recommendations"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Log parameters
        params = {
            'n_components': 64,
            'loss': 'warp',
            'epochs': 30,
            'test_percentage': 0.2
        }
        mlflow.log_params(params)

        # Load data
        interactions = load_interaction_data()
        mlflow.log_metric('total_interactions', len(interactions))
        mlflow.log_metric('unique_customers', interactions['customer_id'].nunique())
        mlflow.log_metric('unique_products', interactions['product_id'].nunique())

        # Split data
        train, test = create_train_test_split(interactions, params['test_percentage'])

        # Train model
        model, dataset = train_model(
            train,
            n_components=params['n_components'],
            loss=params['loss'],
            epochs=params['epochs']
        )

        # Evaluate
        metrics = evaluate_model(model, test, dataset)
        mlflow.log_metrics(metrics)

        # Save model
        wrapped_model = RecommendationModelWrapper(model, dataset)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,
            pip_requirements=[
                "lightfm==1.17",
                "numpy==1.26.2",
                "scipy==1.11.4",
                "pandas==2.1.4"
            ]
        )

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_details = mlflow.register_model(model_uri, "product-recommendation")

        logger.info(f"Model registered: {model_details.name} version {model_details.version}")

        print("\n=== Training Complete ===")
        print(f"Model: {model_details.name}")
        print(f"Version: {model_details.version}")
        print(f"Metrics: {metrics}")
        print(f"MLflow UI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
