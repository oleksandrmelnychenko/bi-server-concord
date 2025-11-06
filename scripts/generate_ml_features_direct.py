#!/usr/bin/env python3
"""
Generate ML Features Directly from MSSQL (Quick Prototype Path)

This script bypasses Delta Lake and dbt to quickly generate ML features:
1. Connects directly to MSSQL ConcordDb_v5
2. Extracts 183K customer-product interactions
3. Generates all ML features (interaction_matrix, customer_features, product_graph_features)
4. Saves to DuckDB for immediate model training

Usage:
    python scripts/generate_ml_features_direct.py

Output:
    - data/ml_features/concord_ml.duckdb (DuckDB database with all features)
    - data/ml_features/*.csv (CSV exports for inspection)
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
DUCKDB_PATH = OUTPUT_DIR / "concord_ml.duckdb"


class MLFeatureGenerator:
    """Generate ML features directly from MSSQL"""

    def __init__(self):
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

        # Create schema
        self.duckdb_conn.execute("CREATE SCHEMA IF NOT EXISTS ml_features")
        logger.info("✓ Connected to DuckDB")

    def extract_interaction_matrix(self):
        """Extract customer-product interactions (183K+)"""
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING INTERACTION MATRIX (183K+ interactions)")
        logger.info("="*80)

        query = """
        SELECT
            CAST(ca.ClientID AS VARCHAR(50)) as customer_id,
            CAST(oi.ProductID AS VARCHAR(50)) as product_id,

            -- Interaction metrics (aggregate across ALL customer agreements)
            COUNT(DISTINCT o.ID) as num_purchases,
            SUM(oi.Qty) as total_quantity,
            SUM(oi.Qty * oi.PricePerItem) as total_spent,
            AVG(oi.Qty) as avg_quantity_per_order,
            AVG(oi.PricePerItem) as avg_price_per_item,

            -- Temporal metrics
            MAX(o.Created) as last_purchase_date,
            MIN(o.Created) as first_purchase_date,
            DATEDIFF(day, MAX(o.Created), GETDATE()) as days_since_last_purchase,
            DATEDIFF(day, MIN(o.Created), MAX(o.Created)) as purchase_span_days

        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID

        WHERE o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
            AND oi.Qty > 0

        GROUP BY ca.ClientID, oi.ProductID
        """

        logger.info("Executing query...")
        df = pd.read_sql(query, self.conn)
        logger.info(f"✓ Extracted {len(df):,} customer-product interactions")

        # Convert date columns to datetime
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])

        # Calculate derived features
        logger.info("Calculating derived features...")

        # Implicit rating (1-5 scale) with temporal decay
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

    def extract_customer_features(self, interaction_df):
        """Generate customer-level features"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING CUSTOMER FEATURES")
        logger.info("="*80)

        # Aggregate by customer
        customer_features = interaction_df.groupby('customer_id').agg({
            'num_purchases': 'sum',
            'total_quantity': 'sum',
            'total_spent': 'sum',
            'product_id': 'nunique',
            'last_purchase_date': 'max',
            'first_purchase_date': 'min',
            'days_since_last_purchase': 'min'
        }).reset_index()

        customer_features.columns = [
            'customer_id', 'total_orders', 'total_quantity_purchased',
            'lifetime_value', 'unique_products_purchased',
            'last_purchase_date', 'first_purchase_date', 'days_since_last_purchase'
        ]

        # Calculate customer lifespan
        customer_features['customer_lifespan_days'] = (
            customer_features['last_purchase_date'] - customer_features['first_purchase_date']
        ).dt.days

        # Orders per month
        customer_features['orders_per_month'] = customer_features['total_orders'] / np.maximum(
            1, customer_features['customer_lifespan_days'] / 30
        )

        # RFM Scores
        customer_features['recency_score'] = np.where(customer_features['days_since_last_purchase'] <= 30, 5,
                                              np.where(customer_features['days_since_last_purchase'] <= 60, 4,
                                              np.where(customer_features['days_since_last_purchase'] <= 90, 3,
                                              np.where(customer_features['days_since_last_purchase'] <= 180, 2, 1))))

        customer_features['frequency_score'] = np.where(customer_features['total_orders'] >= 10, 5,
                                                np.where(customer_features['total_orders'] >= 5, 4,
                                                np.where(customer_features['total_orders'] >= 3, 3,
                                                np.where(customer_features['total_orders'] >= 2, 2, 1))))

        customer_features['monetary_score'] = np.where(customer_features['lifetime_value'] >= 50000, 5,
                                               np.where(customer_features['lifetime_value'] >= 20000, 4,
                                               np.where(customer_features['lifetime_value'] >= 10000, 3,
                                               np.where(customer_features['lifetime_value'] >= 5000, 2, 1))))

        # Customer segment
        def classify_segment(row):
            if row['recency_score'] >= 4 and row['frequency_score'] >= 4 and row['monetary_score'] >= 4:
                return 'Champions'
            elif row['recency_score'] >= 4 and row['frequency_score'] >= 3:
                return 'Loyal Customers'
            elif row['recency_score'] >= 4 and row['monetary_score'] >= 4:
                return 'Big Spenders'
            elif row['recency_score'] >= 3 and row['frequency_score'] >= 3:
                return 'Potential Loyalists'
            elif row['recency_score'] >= 4:
                return 'Recent Customers'
            elif row['recency_score'] == 3:
                return 'At Risk'
            elif row['recency_score'] == 2 and row['frequency_score'] >= 2:
                return 'Cannot Lose Them'
            elif row['recency_score'] <= 2:
                return 'Lost'
            else:
                return 'Needs Attention'

        customer_features['customer_segment'] = customer_features.apply(classify_segment, axis=1)

        # Customer tier
        customer_features['customer_tier'] = np.where(customer_features['lifetime_value'] >= 100000, 'Platinum',
                                              np.where(customer_features['lifetime_value'] >= 50000, 'Gold',
                                              np.where(customer_features['lifetime_value'] >= 20000, 'Silver',
                                              np.where(customer_features['lifetime_value'] >= 5000, 'Bronze', 'New'))))

        # Engagement score
        customer_features['engagement_score'] = np.minimum(100, np.maximum(0,
            (customer_features['recency_score'] * 15) +
            (customer_features['frequency_score'] * 10) +
            (customer_features['monetary_score'] * 15)
        ))

        logger.info(f"✓ Generated features for {len(customer_features):,} customers")
        logger.info(f"  Customer segments: {customer_features['customer_segment'].value_counts().to_dict()}")
        logger.info(f"  Customer tiers: {customer_features['customer_tier'].value_counts().to_dict()}")

        return customer_features

    def extract_product_graph_features(self):
        """Extract product graph features (1.7M analogues)"""
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING PRODUCT GRAPH FEATURES")
        logger.info("="*80)

        query = """
        SELECT
            CAST(p.ID AS VARCHAR(50)) as product_id,
            p.Name as product_name,
            CAST(p.IsForSale AS BIT) as is_for_sale,
            CAST(p.IsForWeb AS BIT) as is_for_web,
            CAST(p.HasAnalogue AS BIT) as has_analogue,
            p.Weight as weight,

            -- Count analogues
            (SELECT COUNT(*) FROM dbo.ProductAnalogue pa
             WHERE pa.BaseProductID = p.ID AND pa.Deleted = 0) as num_analogues,

            -- Purchase metrics
            (SELECT COUNT(DISTINCT oi.OrderID) FROM dbo.OrderItem oi
             WHERE oi.ProductID = p.ID AND oi.Deleted = 0) as times_ordered,

            (SELECT COUNT(DISTINCT o.ClientAgreementID) FROM dbo.OrderItem oi
             INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
             WHERE oi.ProductID = p.ID AND oi.Deleted = 0 AND o.Deleted = 0) as num_customers

        FROM dbo.Product p
        WHERE p.Deleted = 0
            AND p.ID IS NOT NULL
        """

        logger.info("Executing query...")
        df = pd.read_sql(query, self.conn)

        # Calculate graph centrality (normalized)
        df['graph_centrality'] = np.minimum(1.0, df['num_analogues'] / 100.0)

        # Popularity tier
        df['popularity_tier'] = np.where(df['times_ordered'] >= 100, 'Very Popular',
                                np.where(df['times_ordered'] >= 50, 'Popular',
                                np.where(df['times_ordered'] >= 10, 'Moderate',
                                np.where(df['times_ordered'] >= 1, 'Low', 'Never Sold'))))

        # Graph role
        df['graph_role'] = np.where((df['num_analogues'] >= 100) & (df['times_ordered'] >= 50), 'Hub Product',
                           np.where((df['num_analogues'] >= 50) & (df['times_ordered'] >= 10), 'Popular Node',
                           np.where(df['num_analogues'] >= 10, 'Connected Node', 'Isolated Node')))

        logger.info(f"✓ Extracted features for {len(df):,} products")
        logger.info(f"  Products with analogues: {(df['num_analogues'] > 0).sum():,}")
        logger.info(f"  Hub products: {(df['graph_role'] == 'Hub Product').sum():,}")
        logger.info(f"  Popularity tiers: {df['popularity_tier'].value_counts().to_dict()}")

        return df

    def save_to_duckdb(self, interaction_df, customer_df, product_df):
        """Save all features to DuckDB"""
        logger.info("\n" + "="*80)
        logger.info("SAVING TO DUCKDB")
        logger.info("="*80)

        # Save interaction matrix
        logger.info("Saving interaction_matrix...")
        self.duckdb_conn.execute("DROP TABLE IF EXISTS ml_features.interaction_matrix")
        self.duckdb_conn.execute("""
            CREATE TABLE ml_features.interaction_matrix AS
            SELECT * FROM interaction_df
        """)
        logger.info(f"  ✓ Saved {len(interaction_df):,} interactions")

        # Save customer features
        logger.info("Saving customer_features...")
        self.duckdb_conn.execute("DROP TABLE IF EXISTS ml_features.customer_features")
        self.duckdb_conn.execute("""
            CREATE TABLE ml_features.customer_features AS
            SELECT * FROM customer_df
        """)
        logger.info(f"  ✓ Saved {len(customer_df):,} customer records")

        # Save product features
        logger.info("Saving product_graph_features...")
        self.duckdb_conn.execute("DROP TABLE IF EXISTS ml_features.product_graph_features")
        self.duckdb_conn.execute("""
            CREATE TABLE ml_features.product_graph_features AS
            SELECT * FROM product_df
        """)
        logger.info(f"  ✓ Saved {len(product_df):,} product records")

    def save_to_csv(self, interaction_df, customer_df, product_df):
        """Save CSVs for inspection"""
        logger.info("\nSaving CSV exports...")

        interaction_df.to_csv(OUTPUT_DIR / "interaction_matrix.csv", index=False)
        customer_df.to_csv(OUTPUT_DIR / "customer_features.csv", index=False)
        product_df.to_csv(OUTPUT_DIR / "product_graph_features.csv", index=False)

        logger.info(f"  ✓ CSVs saved to {OUTPUT_DIR}")

    def run(self):
        """Main execution"""
        try:
            start_time = datetime.now()
            logger.info("\n" + "="*80)
            logger.info("ML FEATURE GENERATION - QUICK PROTOTYPE PATH")
            logger.info("="*80)
            logger.info(f"Start time: {start_time}")

            # Connect
            self.connect_mssql()
            self.connect_duckdb()

            # Extract features
            interaction_df = self.extract_interaction_matrix()
            customer_df = self.extract_customer_features(interaction_df)
            product_df = self.extract_product_graph_features()

            # Save
            self.save_to_duckdb(interaction_df, customer_df, product_df)
            self.save_to_csv(interaction_df, customer_df, product_df)

            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "="*80)
            logger.info("✅ FEATURE GENERATION COMPLETE!")
            logger.info("="*80)
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"\nFeatures saved to:")
            logger.info(f"  DuckDB: {DUCKDB_PATH}")
            logger.info(f"  CSVs:   {OUTPUT_DIR}")
            logger.info(f"\nFeature counts:")
            logger.info(f"  Interactions: {len(interaction_df):,}")
            logger.info(f"  Customers:    {len(customer_df):,}")
            logger.info(f"  Products:     {len(product_df):,}")
            logger.info(f"\nNext step: Train LightFM v2 with these features!")

            return True

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return False

        finally:
            if self.conn:
                self.conn.close()
            if self.duckdb_conn:
                self.duckdb_conn.close()


if __name__ == "__main__":
    generator = MLFeatureGenerator()
    success = generator.run()
    sys.exit(0 if success else 1)