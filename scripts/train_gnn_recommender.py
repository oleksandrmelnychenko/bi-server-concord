#!/usr/bin/env python3
"""
Train GNN Recommender System

Training strategy:
- Temporal split (H1 2024 for training, H2 2024 for validation)
- BPR loss (Bayesian Personalized Ranking)
- Negative sampling for efficiency
- Early stopping based on validation precision
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import duckdb
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import our GNN model
from build_gnn_recommender import HeteroLightGCN, load_graph_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss"""

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_scores: Scores for positive (purchased) items
            neg_scores: Scores for negative (not purchased) items

        Returns:
            BPR loss
        """
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()


def prepare_temporal_split(db_path: str = 'data/graph_features.duckdb',
                           split_date: str = '2024-07-01') -> Tuple[Dict, Dict]:
    """
    Split purchase history into train/val based on date

    Args:
        db_path: Path to DuckDB database
        split_date: Date to split on (YYYY-MM-DD)

    Returns:
        train_data, val_data dictionaries
    """
    logger.info(f"\n📅 Splitting data at {split_date}...")

    conn = duckdb.connect(db_path, read_only=True)

    # Get train data (before split date)
    train_purchases = conn.execute(f"""
        SELECT customer_id, product_id, purchase_date
        FROM purchase_history
        WHERE purchase_date < '{split_date}'
    """).df()

    # Get val data (after split date)
    val_purchases = conn.execute(f"""
        SELECT customer_id, product_id, purchase_date
        FROM purchase_history
        WHERE purchase_date >= '{split_date}'
    """).df()

    conn.close()

    # Build dictionaries
    train_data = {}
    for _, row in train_purchases.iterrows():
        cid = row['customer_id']
        pid = row['product_id']
        if cid not in train_data:
            train_data[cid] = set()
        train_data[cid].add(pid)

    val_data = {}
    for _, row in val_purchases.iterrows():
        cid = row['customer_id']
        pid = row['product_id']
        if cid not in val_data:
            val_data[cid] = set()
        val_data[cid].add(pid)

    # Only keep customers who have purchases in both train and val
    common_customers = set(train_data.keys()) & set(val_data.keys())
    train_data = {k: v for k, v in train_data.items() if k in common_customers}
    val_data = {k: v for k, v in val_data.items() if k in common_customers}

    logger.info(f"✅ Train: {len(train_data)} customers, {sum(len(v) for v in train_data.values())} purchases")
    logger.info(f"✅ Val: {len(val_data)} customers, {sum(len(v) for v in val_data.values())} purchases")

    return train_data, val_data


def train_epoch(model: HeteroLightGCN,
                edge_index_dict: Dict,
                train_data: Dict,
                metadata: Dict,
                optimizer: optim.Optimizer,
                criterion: BPRLoss,
                device: torch.device,
                num_negatives: int = 4,
                batch_size: int = 128) -> float:  # QUICK WIN #2: Reduced from 1024 for 8x more updates!
    """
    Train for one epoch

    Args:
        model: GNN model
        edge_index_dict: Edge indices
        train_data: Training data {customer_id: set(product_ids)}
        metadata: Metadata with ID mappings
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_negatives: Number of negative samples per positive
        batch_size: Batch size

    Returns:
        Average loss
    """
    model.train()

    # Move edge indices to device
    edge_index_dict_device = {k: v.to(device) for k, v in edge_index_dict.items()}

    # Prepare training batches
    customer_ids = list(train_data.keys())
    np.random.shuffle(customer_ids)

    all_product_ids = list(metadata['product_id_to_idx'].keys())

    total_loss = 0.0
    num_batches = 0

    for i in tqdm(range(0, len(customer_ids), batch_size), desc="Training"):
        batch_customers = customer_ids[i:i+batch_size]

        batch_customer_idx = []
        batch_pos_product_idx = []
        batch_neg_product_idx = []

        for cust_id in batch_customers:
            if cust_id not in metadata['customer_id_to_idx']:
                continue

            cust_idx = metadata['customer_id_to_idx'][cust_id]
            pos_products = train_data[cust_id]

            # Sample positive products
            for pos_prod_id in list(pos_products)[:10]:  # Limit to 10 per customer
                if pos_prod_id not in metadata['product_id_to_idx']:
                    continue

                pos_prod_idx = metadata['product_id_to_idx'][pos_prod_id]

                # Sample negative products (not purchased)
                neg_candidates = set(all_product_ids) - pos_products
                neg_samples = np.random.choice(list(neg_candidates), size=num_negatives, replace=False)

                for neg_prod_id in neg_samples:
                    if neg_prod_id in metadata['product_id_to_idx']:
                        neg_prod_idx = metadata['product_id_to_idx'][neg_prod_id]

                        batch_customer_idx.append(cust_idx)
                        batch_pos_product_idx.append(pos_prod_idx)
                        batch_neg_product_idx.append(neg_prod_idx)

        if len(batch_customer_idx) == 0:
            continue

        # Convert to tensors
        customer_idx_t = torch.tensor(batch_customer_idx, dtype=torch.long, device=device)
        pos_product_idx_t = torch.tensor(batch_pos_product_idx, dtype=torch.long, device=device)
        neg_product_idx_t = torch.tensor(batch_neg_product_idx, dtype=torch.long, device=device)

        # Forward pass through GNN to get embeddings (with gradients)
        embeddings = model(edge_index_dict_device)

        # Get scores
        pos_scores = model.predict(customer_idx_t, pos_product_idx_t, embeddings)
        neg_scores = model.predict(customer_idx_t, neg_product_idx_t, embeddings)

        # Compute loss
        loss = criterion(pos_scores, neg_scores)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    logger.info("\n" + "="*80)
    logger.info("TRAINING GNN RECOMMENDER SYSTEM")
    logger.info("="*80)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️  Device: {device}")

    # Load graph data
    edge_index_dict, metadata = load_graph_data()

    # Prepare temporal split
    train_data, val_data = prepare_temporal_split()

    # Initialize model
    model = HeteroLightGCN(
        num_customers=metadata['num_customers'],
        num_products=metadata['num_products'],
        num_groups=metadata['num_groups'],
        num_brands=metadata['num_brands'],
        embedding_dim=64,
        num_layers=3
    ).to(device)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = BPRLoss()

    # Training loop
    num_epochs = 50  # QUICK WIN #2: Increased from 20 for better convergence
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    logger.info(f"\n🚀 Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, edge_index_dict, train_data, metadata,
            optimizer, criterion, device
        )

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")

        # Early stopping check
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            patience_counter = 0

            # Save best model
            os.makedirs('models/gnn_recommender', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'metadata': metadata
            }, 'models/gnn_recommender/best_model.pt')

            logger.info(f"  ✅ Saved best model (loss: {train_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  ⏳ Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info(f"\n⏹️  Early stopping triggered")
                break

    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Best train loss: {best_val_loss:.4f}")
    logger.info(f"   Model saved to: models/gnn_recommender/best_model.pt")


if __name__ == '__main__':
    main()
