#!/usr/bin/env python3
"""
Generate ML Features with Temporal Split (Clean Training Data)

This script generates TWO feature sets to avoid temporal contamination:
1. TRAINING set: Only purchases BEFORE split_date (for model training)
2. FULL set: All purchases (for validation/production inference)

This prevents the trained models from seeing validation period data.

Usage:
    python scripts/generate_ml_features_temporal.py --split_date 2024-06-30

Output:
    - interaction_matrix_train (clean training data before split)
    - interaction_matrix_full (complete data for inference)
    - customer_features, product_graph_features
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymssql
import duckdb
from datetime import datetime, timedelta
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MSSQL Configuration
MSSQL_CONFIG = {
    "host": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": os.getenv("MSSQL_PORT", "1433"),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb_v5"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD", "Grimm_jow92"),
}

# Output paths
OUTPUT_DIR = Path("data/ml_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_PATH = OUTPUT_DIR / "concord_ml_temporal.duckdb"


class TemporalMLFeatureGenerator:
    """Generate ML features with proper temporal split"""

    def __init__(self, split_date: str):
        """
        Args:
            split_date: Date string (YYYY-MM-DD) to split training/validation
        """
        self.split_date = split_date
        self.conn = None
        self.duckdb_conn = None

    def connect_mssql(self):
        """Connect to MSSQL"""
        logger.info(f"Connecting to MSSQL {MSSQL_CONFIG['database']}...")
        self.conn = pymssql.connect(
            server=MSSQL_CONFIG['host'],
            port=int(MSSQL_CONFIG['port']),
            user=MSSQL_CONFIG['user'],
            password=MSSQL_CONFIG['password'],
            database=MSSQL_CONFIG['database'],
            tds_version='7.0'
        )
        logger.info("✓ Connected to MSSQL")

    def connect_duckdb(self):
        """Connect to DuckDB for feature storage"""
        logger.info(f"Connecting to DuckDB at {DUCKDB_PATH}...")
        self.duckdb_conn = duckdb.connect(str(DUCKDB_PATH))
        self.duckdb_conn.execute("CREATE SCHEMA IF NOT EXISTS ml_features")
        logger.info("✓ Connected to DuckDB")

    def extract_interaction_matrix(self, max_date: str = None, table_suffix: str = ""):
        """
        Extract customer-product interactions with optional date filter

        Args:
            max_date: Only include purchases before this date (None = all time)
            table_suffix: Suffix for table name (e.g., "_train", "_full")
        """
        logger.info("\n" + "="*80)
        logger.info(f"EXTRACTING INTERACTION MATRIX{table_suffix.upper()}")
        if max_date:
            logger.info(f"Date Filter: Purchases BEFORE {max_date}")
        else:
            logger.info(f"Date Filter: ALL TIME")
        logger.info("="*80)

        # Build date filter
        date_filter = f"AND o.Created < '{max_date}'" if max_date else ""

        query = f"""
        SELECT
            CAST(ca.ClientID AS VARCHAR(50)) as customer_id,
            CAST(oi.ProductID AS VARCHAR(50)) as product_id,

            -- Interaction metrics
            COUNT(DISTINCT o.ID) as num_purchases,
            SUM(oi.Qty) as total_quantity,
            SUM(oi.Qty * oi.PricePerItem) as total_spent,
            AVG(oi.Qty) as avg_quantity_per_order,
            AVG(oi.PricePerItem) as avg_price_per_item,

            -- Temporal metrics (relative to max_date or current date)
            MAX(o.Created) as last_purchase_date,
            MIN(o.Created) as first_purchase_date,
            DATEDIFF(day, MAX(o.Created), {f"'{max_date}'" if max_date else "GETDATE()"}) as days_since_last_purchase,
            DATEDIFF(day, MIN(o.Created), MAX(o.Created)) as purchase_span_days

        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID

        WHERE o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
            AND oi.Qty > 0
            {date_filter}

        GROUP BY ca.ClientID, oi.ProductID
        """

        logger.info("Executing query...")
        df = pd.read_sql(query, self.conn)
        logger.info(f"✓ Extracted {len(df):,} customer-product interactions")

        # Convert date columns
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])

        # Calculate derived features
        logger.info("Calculating derived features...")

        # Implicit rating (1-5 scale)
        df['implicit_rating'] = np.minimum(5, np.maximum(1,
            (df['num_purchases'] * 0.3) +
            np.where(df['days_since_last_purchase'] <= 30, 2.5,
            np.where(df['days_since_last_purchase'] <= 60, 2.0,
            np.where(df['days_since_last_purchase'] <= 90, 1.5,
            np.where(df['days_since_last_purchase'] <= 180, 1.0,
            np.where(df['days_since_last_purchase'] <= 365, 0.5, 0.2))))) +
            np.minimum(2.0, np.log(1 + df['total_spent']) / 5)
        ))

        # Repurchase likelihood
        df['repurchase_likelihood'] = np.where(
            df['num_purchases'] >= 2,
            np.minimum(1.0, np.maximum(0.0,
                1.0 - (df['days_since_last_purchase'] / (df['purchase_span_days'] / (df['num_purchases'] - 1)))
            )),
            0.0
        )

        # Behavioral flags
        df['is_repeat_customer'] = df['num_purchases'] >= 5
        df['is_at_risk'] = df['days_since_last_purchase'] > 180
        df['revenue_potential'] = np.where(df['repurchase_likelihood'] > 0.7, 'High',
                                           np.where(df['repurchase_likelihood'] > 0.4, 'Medium', 'Low'))

        logger.info(f"  Mean implicit rating: {df['implicit_rating'].mean():.2f}")
        logger.info(f"  Repeat customers: {df['is_repeat_customer'].sum():,}")
        logger.info(f"  High revenue potential: {(df['revenue_potential'] == 'High').sum():,}")

        return df

    def generate_customer_features(self, interaction_df):
        """Generate customer-level features"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING CUSTOMER FEATURES")
        logger.info("="*80)

        customer_features = interaction_df.groupby('customer_id').agg({
            'product_id': 'count',
            'num_purchases': 'sum',
            'total_spent': 'sum',
            'last_purchase_date': 'max',
            'first_purchase_date': 'min',
            'implicit_rating': 'mean',
            'repurchase_likelihood': 'mean',
            'is_repeat_customer': 'any',
            'revenue_potential': lambda x: (x == 'High').sum()
        }).reset_index()

        customer_features.columns = [
            'customer_id', 'num_unique_products', 'total_orders', 'lifetime_value',
            'last_purchase_date', 'first_purchase_date', 'avg_implicit_rating',
            'avg_repurchase_likelihood', 'is_repeat_customer', 'high_revenue_products'
        ]

        # Calculate recency, frequency, monetary
        customer_features['recency_days'] = (
            pd.Timestamp.now() - customer_features['last_purchase_date']
        ).dt.days
        customer_features['tenure_days'] = (
            customer_features['last_purchase_date'] - customer_features['first_purchase_date']
        ).dt.days

        # RFM Segmentation
        customer_features['rfm_score'] = (
            pd.qcut(customer_features['recency_days'], q=5, labels=[5,4,3,2,1], duplicates='drop').astype(float) +
            pd.qcut(customer_features['total_orders'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(float) +
            pd.qcut(customer_features['lifetime_value'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(float)
        )

        # Customer segments
        def segment_customer(row):
            if row['rfm_score'] >= 12: return 'Champions'
            elif row['rfm_score'] >= 10: return 'Loyal Customers'
            elif row['rfm_score'] >= 8: return 'Potential Loyalists'
            elif row['rfm_score'] >= 6: return 'Recent Customers'
            elif row['rfm_score'] >= 5: return 'At Risk'
            elif row['rfm_score'] >= 4: return 'Cannot Lose Them'
            else: return 'Lost'

        customer_features['segment'] = customer_features.apply(segment_customer, axis=1)

        # Customer tiers
        customer_features['tier'] = pd.cut(
            customer_features['lifetime_value'],
            bins=[0, 1000, 5000, 10000, float('inf')],
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )

        logger.info(f"✓ Generated features for {len(customer_features):,} customers")
        logger.info(f"  Customer segments: {dict(customer_features['segment'].value_counts())}")
        logger.info(f"  Customer tiers: {dict(customer_features['tier'].value_counts())}")

        return customer_features

    def generate_product_graph_features(self):
        """Generate product-level features (simplified - just product list)"""
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING PRODUCT GRAPH FEATURES")
        logger.info("="*80)

        query = """
        SELECT
            CAST(p.ID AS VARCHAR(50)) as product_id,
            p.Title as product_name
        FROM dbo.Product p
        """

        logger.info("Executing query...")
        df = pd.read_sql(query, self.conn)
        logger.info(f"✓ Extracted features for {len(df):,} products")

        # Add simple features
        df['has_analogues'] = False
        df['is_hub_product'] = False

        return df

    def save_to_duckdb(self, train_df, full_df, customer_df):
        """Save features to DuckDB"""
        logger.info("\n" + "="*80)
        logger.info("SAVING TO DUCKDB")
        logger.info("="*80)

        # Save training interaction matrix
        logger.info("Saving interaction_matrix_train...")
        self.duckdb_conn.execute("DROP TABLE IF EXISTS ml_features.interaction_matrix_train")
        self.duckdb_conn.execute(
            "CREATE TABLE ml_features.interaction_matrix_train AS SELECT * FROM train_df"
        )
        logger.info(f"  ✓ Saved {len(train_df):,} training interactions")

        # Save full interaction matrix
        logger.info("Saving interaction_matrix_full...")
        self.duckdb_conn.execute("DROP TABLE IF EXISTS ml_features.interaction_matrix_full")
        self.duckdb_conn.execute(
            "CREATE TABLE ml_features.interaction_matrix_full AS SELECT * FROM full_df"
        )
        logger.info(f"  ✓ Saved {len(full_df):,} full interactions")

        # Save customer features
        logger.info("Saving customer_features...")
        self.duckdb_conn.execute("DROP TABLE IF EXISTS ml_features.customer_features")
        self.duckdb_conn.execute(
            "CREATE TABLE ml_features.customer_features AS SELECT * FROM customer_df"
        )
        logger.info(f"  ✓ Saved {len(customer_df):,} customer records")

        # Save CSV exports
        logger.info("\nSaving CSV exports...")
        train_df.to_csv(OUTPUT_DIR / "interaction_matrix_train.csv", index=False)
        full_df.to_csv(OUTPUT_DIR / "interaction_matrix_full.csv", index=False)
        customer_df.to_csv(OUTPUT_DIR / "customer_features.csv", index=False)
        logger.info(f"  ✓ CSVs saved to {OUTPUT_DIR}")

    def run(self):
        """Execute full feature generation pipeline"""
        start_time = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("ML FEATURE GENERATION - TEMPORAL SPLIT VERSION")
        logger.info("="*80)
        logger.info(f"Start time: {start_time}")
        logger.info(f"Split date: {self.split_date}")

        # Connect to databases
        self.connect_mssql()
        self.connect_duckdb()

        # Extract TRAINING interaction matrix (before split_date)
        train_df = self.extract_interaction_matrix(
            max_date=self.split_date,
            table_suffix="_train"
        )

        # Extract FULL interaction matrix (all time)
        full_df = self.extract_interaction_matrix(
            max_date=None,
            table_suffix="_full"
        )

        # Generate customer features (based on full data)
        customer_df = self.generate_customer_features(full_df)

        # Skip product features - not needed for ML training
        logger.info("\n✓ Skipping product features (not needed for ML training)")

        # Save everything
        self.save_to_duckdb(train_df, full_df, customer_df)

        # Close connections
        self.conn.close()
        self.duckdb_conn.close()

        # Summary
        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("✅ TEMPORAL FEATURE GENERATION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"\nFeatures saved to:")
        logger.info(f"  DuckDB: {DUCKDB_PATH}")
        logger.info(f"  CSVs:   {OUTPUT_DIR}")
        logger.info(f"\nFeature counts:")
        logger.info(f"  Training interactions: {len(train_df):,} (before {self.split_date})")
        logger.info(f"  Full interactions: {len(full_df):,} (all time)")
        logger.info(f"  Customers: {len(customer_df):,}")
        logger.info(f"\n✅ Ready for temporal model training!")


def main():
    parser = argparse.ArgumentParser(description='Generate ML features with temporal split')
    parser.add_argument(
        '--split_date',
        type=str,
        default='2024-06-30',
        help='Split date for training/validation (YYYY-MM-DD, default: 2024-06-30)'
    )
    args = parser.parse_args()

    generator = TemporalMLFeatureGenerator(split_date=args.split_date)
    generator.run()


if __name__ == "__main__":
    main()
