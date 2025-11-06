#!/usr/bin/env python3
"""
Neural Collaborative Filtering for B2B Recommendation System

Uses deep learning with customer/product embeddings + metadata features
to predict purchase probability. Much better for sparse B2B data than
traditional collaborative filtering.

Architecture:
- Customer embedding (128 dim)
- Product embedding (128 dim)
- Customer metadata features (RFM, purchase stats)
- Product metadata features (category, price, etc.)
- Deep neural network (256 → 128 → 64 → 1)
- Output: Purchase probability

Training:
- Positive samples: Actual H2 2024 purchases
- Negative samples: Random non-purchases (sampled)
- Loss: Binary cross-entropy
- Optimizer: Adam with learning rate scheduling

Expected Performance: >50% precision (beat 42.5% baseline)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import pymssql
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SPLIT_DATE = '2024-06-30'
VALIDATION_START = '2024-06-30'
VALIDATION_END = '2025-01-01'

# Model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIMS = [256, 128, 64]
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 512
EPOCHS = 20
NEGATIVE_SAMPLE_RATIO = 4  # For each positive, sample N negatives


class PurchaseDataset(Dataset):
    """Dataset for customer-product purchase pairs"""

    def __init__(self, interactions: pd.DataFrame,
                 customer_features: pd.DataFrame,
                 product_features: pd.DataFrame):
        """
        Args:
            interactions: DataFrame with columns [customer_id, product_id, purchased]
            customer_features: DataFrame with customer metadata
            product_features: DataFrame with product metadata
        """
        self.interactions = interactions.reset_index(drop=True)
        self.customer_features = customer_features.set_index('customer_id')
        self.product_features = product_features.set_index('product_id')

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        customer_id = row['customer_id']
        product_id = row['product_id']
        customer_idx = row['customer_idx']
        product_idx = row['product_idx']
        label = row['purchased']

        # Get customer features
        cust_feats = self.customer_features.loc[customer_id].values.astype(np.float32)

        # Get product features
        prod_feats = self.product_features.loc[product_id].values.astype(np.float32)

        return {
            'customer_idx': customer_idx,
            'product_idx': product_idx,
            'customer_features': torch.FloatTensor(cust_feats),
            'product_features': torch.FloatTensor(prod_feats),
            'label': torch.FloatTensor([label])
        }


class NeuralRecommender(nn.Module):
    """Neural network for B2B product recommendations"""

    def __init__(self, n_customers: int, n_products: int,
                 n_customer_features: int, n_product_features: int,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        super(NeuralRecommender, self).__init__()

        # Embedding layers
        self.customer_embedding = nn.Embedding(n_customers, embedding_dim)
        self.product_embedding = nn.Embedding(n_products, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.customer_embedding.weight, std=0.01)
        nn.init.normal_(self.product_embedding.weight, std=0.01)

        # Calculate input dimension for first dense layer
        input_dim = (embedding_dim * 2) + n_customer_features + n_product_features

        # Build deep network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, customer_ids, product_ids, customer_features, product_features):
        # Get embeddings
        customer_embed = self.customer_embedding(customer_ids)
        product_embed = self.product_embedding(product_ids)

        # Concatenate all features
        x = torch.cat([
            customer_embed,
            product_embed,
            customer_features,
            product_features
        ], dim=1)

        # Forward through network
        output = self.network(x)
        return output


class NeuralRecommenderTrainer:
    """Handles data extraction, training, and validation"""

    def __init__(self):
        self.conn = self._connect_mssql()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Encoders (will be fitted during training)
        self.customer_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.customer_scaler = StandardScaler()
        self.product_scaler = StandardScaler()

    def _connect_mssql(self):
        """Connect to MSSQL database"""
        return pymssql.connect(
            server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
            port=int(os.environ.get('MSSQL_PORT', '1433')),
            database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
            user=os.environ.get('MSSQL_USER', 'ef_migrator'),
            password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
        )

    def extract_training_data(self) -> pd.DataFrame:
        """Extract all purchases before split date"""
        logger.info("Extracting training interactions...")

        query = f"""
        SELECT
            CAST(ca.ClientID AS INT) as customer_id,
            CAST(oi.ProductID AS INT) as product_id,
            COUNT(DISTINCT o.ID) as num_purchases,
            MAX(o.Created) as last_purchase_date,
            MIN(o.Created) as first_purchase_date,
            SUM(oi.Qty * oi.PricePerItem) as total_spent
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.Created < '{SPLIT_DATE}'
            AND o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
            AND oi.Qty > 0
        GROUP BY ca.ClientID, oi.ProductID
        """

        df = pd.read_sql(query, self.conn)
        logger.info(f"✓ Loaded {len(df):,} training interactions")
        logger.info(f"  Customers: {df['customer_id'].nunique():,}")
        logger.info(f"  Products: {df['product_id'].nunique():,}")

        return df

    def extract_validation_purchases(self) -> pd.DataFrame:
        """Extract purchases in validation period (H2 2024)"""
        logger.info("Extracting validation period purchases...")

        query = f"""
        SELECT DISTINCT
            CAST(ca.ClientID AS INT) as customer_id,
            CAST(oi.ProductID AS INT) as product_id
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.Created >= '{VALIDATION_START}'
            AND o.Created < '{VALIDATION_END}'
            AND o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
            AND oi.Qty > 0
        """

        df = pd.read_sql(query, self.conn)
        logger.info(f"✓ Loaded {len(df):,} validation purchases")

        return df

    def extract_customer_features(self, customer_ids: List[int]) -> pd.DataFrame:
        """Extract RFM and other customer features"""
        logger.info("Extracting customer features...")

        query = f"""
        WITH CustomerStats AS (
            SELECT
                CAST(ca.ClientID AS INT) as customer_id,
                COUNT(DISTINCT o.ID) as total_orders,
                COUNT(DISTINCT oi.ProductID) as unique_products,
                SUM(oi.Qty * oi.PricePerItem) as lifetime_value,
                AVG(oi.Qty * oi.PricePerItem) as avg_order_value,
                MAX(o.Created) as last_order_date,
                MIN(o.Created) as first_order_date,
                DATEDIFF(day, MAX(o.Created), '{SPLIT_DATE}') as recency_days
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.Created < '{SPLIT_DATE}'
                AND o.Created IS NOT NULL
            GROUP BY ca.ClientID
        )
        SELECT
            customer_id,
            total_orders,
            unique_products,
            lifetime_value,
            avg_order_value,
            recency_days,
            DATEDIFF(day, first_order_date, last_order_date) as customer_age_days
        FROM CustomerStats
        WHERE customer_id IN ({','.join(map(str, customer_ids))})
        """

        df = pd.read_sql(query, self.conn)
        logger.info(f"✓ Extracted features for {len(df):,} customers")

        return df

    def extract_product_features(self, product_ids: List[int]) -> pd.DataFrame:
        """Extract product metadata features"""
        logger.info("Extracting product features...")

        # Get product sales statistics
        query = f"""
        SELECT
            CAST(oi.ProductID AS INT) as product_id,
            COUNT(DISTINCT o.ID) as total_orders,
            COUNT(DISTINCT ca.ClientID) as unique_customers,
            SUM(oi.Qty) as total_quantity_sold,
            AVG(oi.PricePerItem) as avg_price,
            MAX(oi.PricePerItem) as max_price,
            MIN(oi.PricePerItem) as min_price,
            MAX(o.Created) as last_sold_date
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        INNER JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
        WHERE o.Created < '{SPLIT_DATE}'
            AND o.Created IS NOT NULL
            AND oi.ProductID IN ({','.join(map(str, product_ids))})
        GROUP BY oi.ProductID
        """

        df = pd.read_sql(query, self.conn)

        # Add recency
        df['days_since_last_sold'] = (pd.to_datetime(SPLIT_DATE) - pd.to_datetime(df['last_sold_date'])).dt.days
        df = df.drop('last_sold_date', axis=1)

        logger.info(f"✓ Extracted features for {len(df):,} products")

        return df

    def generate_negative_samples(self, train_df: pd.DataFrame,
                                  all_customers: List[int],
                                  all_products: List[int],
                                  ratio: int = 4) -> pd.DataFrame:
        """Generate negative samples (products NOT purchased) - OPTIMIZED"""
        logger.info(f"Generating negative samples (ratio={ratio}:1)...")

        # Convert to sets for fast operations
        all_products_set = set(all_products)

        negative_samples = []

        # Group products by customer for faster lookup
        customer_products = train_df.groupby('customer_id')['product_id'].apply(set).to_dict()

        for customer_id in all_customers:
            # Get products this customer bought
            purchased = customer_products.get(customer_id, set())

            # Get products NOT purchased (set difference)
            not_purchased = list(all_products_set - purchased)

            # How many negatives to sample
            n_positives = len(purchased)
            n_negatives = min(n_positives * ratio, len(not_purchased))

            if n_negatives > 0 and len(not_purchased) > 0:
                # Sample directly from non-purchased products (MUCH FASTER!)
                sampled_products = np.random.choice(not_purchased, size=n_negatives, replace=False)

                for product_id in sampled_products:
                    negative_samples.append({
                        'customer_id': customer_id,
                        'product_id': product_id,
                        'purchased': 0
                    })

        neg_df = pd.DataFrame(negative_samples)
        logger.info(f"✓ Generated {len(neg_df):,} negative samples")

        return neg_df

    def prepare_training_data(self):
        """Prepare all data for training"""
        logger.info("\n" + "="*80)
        logger.info("PREPARING TRAINING DATA")
        logger.info("="*80)

        # Extract training interactions
        train_interactions = self.extract_training_data()

        # Get unique customers and products
        all_customers = sorted(train_interactions['customer_id'].unique())
        all_products = sorted(train_interactions['product_id'].unique())

        logger.info(f"\nTotal customers: {len(all_customers):,}")
        logger.info(f"Total products: {len(all_products):,}")

        # Extract features
        customer_features = self.extract_customer_features(all_customers)
        product_features = self.extract_product_features(all_products)

        # Encode IDs
        logger.info("\nEncoding customer and product IDs...")
        self.customer_encoder.fit(all_customers)
        self.product_encoder.fit(all_products)

        # Normalize features
        logger.info("Normalizing features...")
        feature_cols_customer = [c for c in customer_features.columns if c != 'customer_id']
        feature_cols_product = [c for c in product_features.columns if c != 'product_id']

        customer_features[feature_cols_customer] = self.customer_scaler.fit_transform(
            customer_features[feature_cols_customer].fillna(0)
        )
        product_features[feature_cols_product] = self.product_scaler.fit_transform(
            product_features[feature_cols_product].fillna(0)
        )

        # Create positive samples (actual purchases)
        positive_samples = train_interactions[['customer_id', 'product_id']].copy()
        positive_samples['purchased'] = 1

        # Generate negative samples
        negative_samples = self.generate_negative_samples(
            train_interactions, all_customers, all_products,
            ratio=NEGATIVE_SAMPLE_RATIO
        )

        # Combine positive and negative samples
        all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)

        # Shuffle
        all_samples = all_samples.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"\n✓ Total training samples: {len(all_samples):,}")
        logger.info(f"  Positive: {(all_samples['purchased'] == 1).sum():,}")
        logger.info(f"  Negative: {(all_samples['purchased'] == 0).sum():,}")

        return all_samples, customer_features, product_features

    def train(self):
        """Train the neural recommender model"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING NEURAL RECOMMENDER")
        logger.info("="*80)

        # Prepare data
        interactions, customer_features, product_features = self.prepare_training_data()

        # Encode customer and product IDs in interactions
        interactions['customer_idx'] = self.customer_encoder.transform(interactions['customer_id'])
        interactions['product_idx'] = self.product_encoder.transform(interactions['product_id'])

        # Split train/validation
        train_df, val_df = train_test_split(interactions, test_size=0.1, random_state=42, stratify=interactions['purchased'])

        logger.info(f"\nTrain set: {len(train_df):,} samples")
        logger.info(f"Validation set: {len(val_df):,} samples")

        # Create datasets
        train_dataset = PurchaseDataset(train_df, customer_features, product_features)
        val_dataset = PurchaseDataset(val_df, customer_features, product_features)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Initialize model
        n_customers = len(self.customer_encoder.classes_)
        n_products = len(self.product_encoder.classes_)
        n_customer_features = len([c for c in customer_features.columns if c != 'customer_id'])
        n_product_features = len([c for c in product_features.columns if c != 'product_id'])

        model = NeuralRecommender(
            n_customers=n_customers,
            n_products=n_products,
            n_customer_features=n_customer_features,
            n_product_features=n_product_features,
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT_RATE
        ).to(self.device)

        logger.info(f"\nModel architecture:")
        logger.info(f"  Customers: {n_customers:,} → Embedding: {EMBEDDING_DIM}")
        logger.info(f"  Products: {n_products:,} → Embedding: {EMBEDDING_DIM}")
        logger.info(f"  Customer features: {n_customer_features}")
        logger.info(f"  Product features: {n_product_features}")
        logger.info(f"  Hidden layers: {HIDDEN_DIMS}")

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        # Training loop
        logger.info(f"\nTraining for {EPOCHS} epochs...")
        logger.info("="*80)

        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                customer_idx = batch['customer_idx'].to(self.device)
                product_idx = batch['product_idx'].to(self.device)
                customer_feats = batch['customer_features'].to(self.device)
                product_feats = batch['product_features'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                outputs = model(customer_idx, product_idx, customer_feats, product_feats)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    customer_idx = batch['customer_idx'].to(self.device)
                    product_idx = batch['product_idx'].to(self.device)
                    customer_feats = batch['customer_features'].to(self.device)
                    product_feats = batch['product_features'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = model(customer_idx, product_idx, customer_feats, product_feats)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Log progress
            logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'models/neural_recommender/best_model.pt')

        logger.info(f"\n✓ Training complete! Best validation loss: {best_val_loss:.4f}")

        # Save encoders and scalers
        os.makedirs('models/neural_recommender', exist_ok=True)

        with open('models/neural_recommender/encoders.pkl', 'wb') as f:
            pickle.dump({
                'customer_encoder': self.customer_encoder,
                'product_encoder': self.product_encoder,
                'customer_scaler': self.customer_scaler,
                'product_scaler': self.product_scaler,
                'customer_features': customer_features,
                'product_features': product_features
            }, f)

        logger.info("✓ Saved encoders and scalers")

        return model


def main():
    """Main training pipeline"""
    print("="*80)
    print("NEURAL COLLABORATIVE FILTERING - B2B RECOMMENDATION SYSTEM")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print(f"Split date: {SPLIT_DATE}")
    print()

    trainer = NeuralRecommenderTrainer()
    model = trainer.train()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("Next step: Run validation script to test on 20 customers")
    print("Expected: >50% precision (better than 42.5% frequency baseline)")


if __name__ == '__main__':
    main()
