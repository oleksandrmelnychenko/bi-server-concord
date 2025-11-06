#!/usr/bin/env python3
"""
Evaluate LightFM Recommendation Model

Metrics:
- Precision@K
- Recall@K
- NDCG@K
- Coverage (% of items recommended)
"""

import pickle
import numpy as np
import logging
from pathlib import Path
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/opt/dagster/app/models/lightfm/recommendation_v1.pkl"

def ndcg_at_k(model, test_interactions, train_interactions, k=10, user_features=None, item_features=None):
    """
    Calculate NDCG@K for the model
    NDCG (Normalized Discounted Cumulative Gain) measures ranking quality
    """
    n_users, n_items = test_interactions.shape

    ndcg_scores = []

    for user_id in range(n_users):
        # Get test items for this user
        test_items = test_interactions[user_id].indices

        if len(test_items) == 0:
            continue

        # Get predictions for all items
        scores = model.predict(
            user_id,
            np.arange(n_items),
            user_features=user_features,
            item_features=item_features
        )

        # Mask out training items
        train_items = train_interactions[user_id].indices
        scores[train_items] = -np.inf

        # Get top K recommendations
        top_k_items = np.argsort(-scores)[:k]

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_items):
            if item in test_items:
                # Relevant item, add to DCG
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # Calculate ideal DCG (all relevant items at top)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))

        # Normalized DCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def evaluate_model():
    """Evaluate the trained model"""
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)

    # Load model
    logger.info(f"\nLoading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    dataset = model_data['dataset']
    interactions_matrix = model_data.get('interactions_matrix')

    # If interactions not in model, rebuild from dataset
    if interactions_matrix is None:
        logger.info("Rebuilding interactions matrix...")
        import duckdb
        conn = duckdb.connect("/opt/dagster/app/data/dbt/concord_bi.duckdb", read_only=True)
        interactions_df = conn.execute("""
            SELECT customer_id, product_id, implicit_rating
            FROM ml_features.interaction_matrix
        """).df()
        conn.close()

        interactions = [
            (row['customer_id'], row['product_id'], row['implicit_rating'])
            for _, row in interactions_df.iterrows()
        ]
        interactions_matrix, _ = dataset.build_interactions(interactions)

    user_features_matrix = model_data.get('user_features_matrix')
    item_features_matrix = model_data.get('item_features_matrix')

    logger.info(f"✓ Model loaded")
    logger.info(f"  Users: {model_data['num_users']:,}")
    logger.info(f"  Items: {model_data['num_items']:,}")
    logger.info(f"  Interactions: {model_data['num_interactions']:,}")

    # Create train/test split (80/20)
    logger.info("\nCreating train/test split (80/20)...")

    # Convert sparse matrix to coordinate format for splitting
    from scipy import sparse
    interactions_coo = interactions_matrix.tocoo()

    # Create train/test masks
    np.random.seed(42)
    n_interactions = interactions_coo.nnz
    test_mask = np.random.rand(n_interactions) < 0.2
    train_mask = ~test_mask

    # Create train and test matrices
    train_interactions = sparse.coo_matrix(
        (interactions_coo.data[train_mask],
         (interactions_coo.row[train_mask], interactions_coo.col[train_mask])),
        shape=interactions_matrix.shape
    ).tocsr()

    test_interactions = sparse.coo_matrix(
        (interactions_coo.data[test_mask],
         (interactions_coo.row[test_mask], interactions_coo.col[test_mask])),
        shape=interactions_matrix.shape
    ).tocsr()

    logger.info(f"  Train interactions: {train_interactions.nnz:,}")
    logger.info(f"  Test interactions: {test_interactions.nnz:,}")

    # Evaluate metrics
    logger.info("\nEvaluating model performance...")

    # Precision@K
    logger.info("  Calculating Precision@K...")
    precision_10 = precision_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        k=10
    ).mean()

    precision_20 = precision_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        k=20
    ).mean()

    # Recall@K
    logger.info("  Calculating Recall@K...")
    recall_10 = recall_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        k=10
    ).mean()

    recall_20 = recall_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        k=20
    ).mean()

    # NDCG@K
    logger.info("  Calculating NDCG@K...")
    ndcg_10 = ndcg_at_k(
        model,
        test_interactions,
        train_interactions,
        k=10,
        user_features=user_features_matrix,
        item_features=item_features_matrix
    )

    ndcg_20 = ndcg_at_k(
        model,
        test_interactions,
        train_interactions,
        k=20,
        user_features=user_features_matrix,
        item_features=item_features_matrix
    )

    # AUC Score
    logger.info("  Calculating AUC score...")
    auc = auc_score(
        model,
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix
    ).mean()

    # Coverage - what % of items get recommended at least once
    logger.info("  Calculating coverage...")
    n_users, n_items = interactions_matrix.shape
    recommended_items = set()

    for user_id in range(min(100, n_users)):  # Sample 100 users for speed
        scores = model.predict(
            user_id,
            np.arange(n_items),
            user_features=user_features_matrix,
            item_features=item_features_matrix
        )
        top_20 = np.argsort(-scores)[:20]
        recommended_items.update(top_20)

    coverage = len(recommended_items) / n_items

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    logger.info("\n📊 Ranking Metrics:")
    logger.info(f"  Precision@10:  {precision_10:.4f} {'✅' if precision_10 > 0.15 else '⚠️'} (target: > 0.15)")
    logger.info(f"  Precision@20:  {precision_20:.4f}")
    logger.info(f"  Recall@10:     {recall_10:.4f} {'✅' if recall_10 > 0.10 else '⚠️'} (target: > 0.10)")
    logger.info(f"  Recall@20:     {recall_20:.4f}")
    logger.info(f"  NDCG@10:       {ndcg_10:.4f} {'✅' if ndcg_10 > 0.25 else '⚠️'} (target: > 0.25)")
    logger.info(f"  NDCG@20:       {ndcg_20:.4f}")

    logger.info(f"\n📈 Other Metrics:")
    logger.info(f"  AUC Score:     {auc:.4f}")
    logger.info(f"  Coverage@20:   {coverage:.2%} ({len(recommended_items)}/{n_items} items)")

    # Interpretation
    logger.info("\n💡 Interpretation:")

    if precision_10 >= 0.15:
        logger.info("  ✅ Precision@10 is good - model recommends relevant products")
    else:
        logger.info(f"  ⚠️  Precision@10 is low - model struggles with relevance")

    if recall_10 >= 0.10:
        logger.info("  ✅ Recall@10 is acceptable - model finds relevant products")
    else:
        logger.info(f"  ⚠️  Recall@10 is low - model misses relevant products")

    if ndcg_10 >= 0.25:
        logger.info("  ✅ NDCG@10 is good - ranking quality is solid")
    else:
        logger.info(f"  ⚠️  NDCG@10 is low - ranking needs improvement")

    if coverage >= 0.50:
        logger.info(f"  ✅ Coverage is good - diverse recommendations")
    else:
        logger.info(f"  ⚠️  Coverage is low - recommendations too narrow")

    # Overall assessment
    logger.info("\n🎯 Overall Assessment:")

    metrics_passed = sum([
        precision_10 >= 0.15,
        recall_10 >= 0.10,
        ndcg_10 >= 0.25
    ])

    if metrics_passed >= 2:
        logger.info("  ✅ Model performs well enough for production")
        logger.info("  ✅ Ready to deploy API endpoint")
    else:
        logger.info("  ⚠️  Model may need improvement")
        logger.info("  Consider:")
        logger.info("    - More training epochs")
        logger.info("    - Different hyperparameters")
        logger.info("    - Additional features")

    logger.info("\n" + "=" * 70)

    return {
        'precision_10': precision_10,
        'precision_20': precision_20,
        'recall_10': recall_10,
        'recall_20': recall_20,
        'ndcg_10': ndcg_10,
        'ndcg_20': ndcg_20,
        'auc': auc,
        'coverage': coverage
    }

if __name__ == "__main__":
    evaluate_model()
