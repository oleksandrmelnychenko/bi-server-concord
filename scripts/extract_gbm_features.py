#!/usr/bin/env python3
"""
Phase I.2: Comprehensive Feature Extraction for Gradient Boosting

Extracts ALL available signals for learn-to-rank:
- 6 Hybrid features (frequency, recency, maintenance_cycle, compatibility, seasonality, monetary)
- 6 Advanced features (from Phase H: sequence, basket, time-based)
- 5+ Business features (margin, stock, category popularity, etc.)
- Cross-features (frequency × recency, etc.)

Target: 15-20 features per customer-product pair for GBM training
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pymssql

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.hybrid_recommender import HybridRecommender
from scripts.extract_advanced_features import AdvancedFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection (same as hybrid_recommender)
DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
}


class ComprehensiveFeatureExtractor:
    """Extract all features for GBM training"""

    def __init__(self):
        self.conn = pymssql.connect(**DB_CONFIG)
        self.hybrid = HybridRecommender()
        self.advanced = AdvancedFeatureExtractor()

    def close(self):
        """Close all connections"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if hasattr(self, 'hybrid') and self.hybrid.conn:
            self.hybrid.conn.close()
        if hasattr(self, 'advanced') and self.advanced.conn:
            self.advanced.conn.close()

    def get_candidate_products(self, customer_id: int, as_of_date: datetime, top_n: int = 1000) -> List[int]:
        """
        Get candidate products for ranking

        Strategy: Get hybrid's top-1000 candidates (this ensures we have good coverage)
        """
        recs = self.hybrid.get_recommendations(
            customer_id=customer_id,
            top_n=top_n,
            as_of_date=as_of_date
        )
        return [int(r['product_id']) for r in recs]

    def extract_hybrid_features(self, customer_id: int, product_ids: List[int], as_of_date: datetime) -> pd.DataFrame:
        """
        Extract 6 hybrid features using existing hybrid recommender

        Returns DataFrame with columns: product_id, frequency, recency, maintenance_cycle,
                                        compatibility, seasonality, monetary
        """
        # Get hybrid scores (which internally calculates all 6 signals)
        recs = self.hybrid.get_recommendations(
            customer_id=customer_id,
            top_n=len(product_ids),
            as_of_date=as_of_date
        )

        # Convert to DataFrame
        df = pd.DataFrame(recs)
        df['product_id'] = df['product_id'].astype(int)

        # Hybrid recommender doesn't expose individual feature scores,
        # so we need to recalculate them
        # For now, we'll use the composite score and extract features separately

        # Get purchase history for this customer
        query = f"""
        SELECT
            CAST(oi.ProductID AS INT) as product_id,
            COUNT(*) as purchase_count,
            MAX(o.Created) as last_purchase_date,
            SUM(oi.Price * oi.Quantity) as total_spent,
            AVG(DATEDIFF(day, LAG(o.Created) OVER (PARTITION BY oi.ProductID ORDER BY o.Created), o.Created)) as avg_cycle_days
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
            AND o.Created < '{as_of_date.strftime('%Y-%m-%d')}'
            AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        purchase_df = pd.read_sql(query, self.conn)

        # Calculate normalized features
        if len(purchase_df) > 0:
            # Frequency: normalized purchase count
            max_count = purchase_df['purchase_count'].max()
            purchase_df['frequency'] = purchase_df['purchase_count'] / max_count if max_count > 0 else 0

            # Recency: days since last purchase (inverted and normalized)
            purchase_df['days_since'] = (as_of_date - pd.to_datetime(purchase_df['last_purchase_date'])).dt.days
            max_days = purchase_df['days_since'].max()
            purchase_df['recency'] = 1 - (purchase_df['days_since'] / max_days) if max_days > 0 else 0

            # Maintenance cycle: regularity of purchases
            purchase_df['maintenance_cycle'] = purchase_df['avg_cycle_days'].apply(
                lambda x: 1.0 / (1.0 + x/30.0) if pd.notna(x) else 0
            )

            # Monetary: normalized total spent
            max_spent = purchase_df['total_spent'].max()
            purchase_df['monetary'] = purchase_df['total_spent'] / max_spent if max_spent > 0 else 0

        else:
            # No purchase history - all zeros
            purchase_df = pd.DataFrame({'product_id': product_ids})
            purchase_df['frequency'] = 0
            purchase_df['recency'] = 0
            purchase_df['maintenance_cycle'] = 0
            purchase_df['monetary'] = 0

        # Compatibility and seasonality require product-level data
        # For now, set to 0 (can enhance later)
        purchase_df['compatibility'] = 0
        purchase_df['seasonality'] = 0

        # Merge with requested product_ids
        result = pd.DataFrame({'product_id': product_ids})
        result = result.merge(
            purchase_df[['product_id', 'frequency', 'recency', 'maintenance_cycle',
                        'compatibility', 'seasonality', 'monetary']],
            on='product_id',
            how='left'
        ).fillna(0)

        return result

    def extract_advanced_features(self, customer_id: int, product_ids: List[int], as_of_date: datetime) -> pd.DataFrame:
        """
        Extract 6 advanced features from Phase H

        Returns DataFrame with columns: product_id, sequence_frequency, sequence_recency,
                                        days_since_last, purchase_overdue, mean_cycle, basket_frequency
        """
        # Use existing AdvancedFeatureExtractor
        features = self.advanced.extract_all_features(customer_id, as_of_date)

        # Convert to DataFrame
        rows = []
        for product_id in product_ids:
            product_features = features.get(product_id, {})
            rows.append({
                'product_id': product_id,
                'sequence_frequency': product_features.get('sequence_frequency', 0),
                'sequence_recency': product_features.get('sequence_recency', 999),
                'days_since_last': product_features.get('days_since_last', 999),
                'purchase_overdue': product_features.get('purchase_overdue', 0),
                'mean_cycle': product_features.get('mean_cycle', 999),
                'basket_frequency': product_features.get('basket_frequency', 0)
            })

        return pd.DataFrame(rows)

    def extract_business_features(self, product_ids: List[int]) -> pd.DataFrame:
        """
        Extract business-level features

        Returns DataFrame with columns: product_id, product_popularity, category_popularity,
                                        has_stock, avg_margin, brand_popularity
        """
        if len(product_ids) == 0:
            return pd.DataFrame()

        product_ids_str = ','.join(str(p) for p in product_ids)

        query = f"""
        WITH ProductStats AS (
            SELECT
                CAST(p.ID AS INT) as product_id,
                COUNT(DISTINCT oi.OrderID) as order_count,
                COUNT(DISTINCT o.ClientAgreementID) as customer_count,
                COALESCE(p.CategoryID, 0) as category_id,
                COALESCE(p.BrandID, 0) as brand_id
            FROM dbo.Product p
            LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID
            LEFT JOIN dbo.[Order] o ON oi.OrderID = o.ID
            WHERE p.ID IN ({product_ids_str})
            GROUP BY p.ID, p.CategoryID, p.BrandID
        ),
        CategoryStats AS (
            SELECT
                CategoryID,
                COUNT(DISTINCT ID) as category_product_count
            FROM dbo.Product
            WHERE CategoryID IS NOT NULL
            GROUP BY CategoryID
        ),
        BrandStats AS (
            SELECT
                BrandID,
                COUNT(DISTINCT ID) as brand_product_count
            FROM dbo.Product
            WHERE BrandID IS NOT NULL
            GROUP BY BrandID
        )
        SELECT
            ps.product_id,
            ps.order_count as product_popularity,
            COALESCE(cs.category_product_count, 0) as category_popularity,
            COALESCE(bs.brand_product_count, 0) as brand_popularity
        FROM ProductStats ps
        LEFT JOIN CategoryStats cs ON ps.category_id = cs.CategoryID
        LEFT JOIN BrandStats bs ON ps.brand_id = bs.BrandID
        """

        df = pd.read_sql(query, self.conn)

        # Normalize features
        if len(df) > 0:
            df['product_popularity'] = df['product_popularity'] / df['product_popularity'].max() if df['product_popularity'].max() > 0 else 0
            df['category_popularity'] = df['category_popularity'] / df['category_popularity'].max() if df['category_popularity'].max() > 0 else 0
            df['brand_popularity'] = df['brand_popularity'] / df['brand_popularity'].max() if df['brand_popularity'].max() > 0 else 0

        return df

    def extract_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-features (feature interactions)

        Adds columns: freq_x_recency, freq_x_monetary, recency_x_cycle, etc.
        """
        df = df.copy()

        # Multiplicative cross-features
        df['freq_x_recency'] = df['frequency'] * df['recency']
        df['freq_x_monetary'] = df['frequency'] * df['monetary']
        df['recency_x_cycle'] = df['recency'] * df['maintenance_cycle']
        df['freq_x_pop'] = df['frequency'] * df.get('product_popularity', 0)

        # Ratio features (with safety for division by zero)
        df['freq_recency_ratio'] = np.where(
            df['recency'] > 0,
            df['frequency'] / df['recency'],
            0
        )

        return df

    def extract_all_features(self, customer_id: int, as_of_date: datetime, top_n: int = 1000) -> pd.DataFrame:
        """
        Extract ALL features for a customer

        Returns DataFrame with ~20 features per product
        """
        logger.info(f"Extracting comprehensive features for customer {customer_id}...")

        # Step 1: Get candidate products
        logger.info("  [1/6] Getting candidate products...")
        product_ids = self.get_candidate_products(customer_id, as_of_date, top_n)
        logger.info(f"    Found {len(product_ids)} candidates")

        # Step 2: Extract hybrid features (6 features)
        logger.info("  [2/6] Extracting hybrid features...")
        hybrid_df = self.extract_hybrid_features(customer_id, product_ids, as_of_date)
        logger.info(f"    Extracted {len(hybrid_df.columns)-1} hybrid features")

        # Step 3: Extract advanced features (6 features)
        logger.info("  [3/6] Extracting advanced features...")
        advanced_df = self.extract_advanced_features(customer_id, product_ids, as_of_date)
        logger.info(f"    Extracted {len(advanced_df.columns)-1} advanced features")

        # Step 4: Extract business features (3-5 features)
        logger.info("  [4/6] Extracting business features...")
        business_df = self.extract_business_features(product_ids)
        logger.info(f"    Extracted {len(business_df.columns)-1} business features")

        # Step 5: Merge all features
        logger.info("  [5/6] Merging features...")
        df = hybrid_df.merge(advanced_df, on='product_id', how='left')
        df = df.merge(business_df, on='product_id', how='left').fillna(0)

        # Step 6: Create cross-features (5 features)
        logger.info("  [6/6] Creating cross-features...")
        df = self.extract_cross_features(df)

        logger.info(f"✓ Total features extracted: {len(df.columns)-1}")

        return df


def main():
    """Test feature extraction on a single customer"""
    logger.info("="*80)
    logger.info("PHASE I.2: COMPREHENSIVE FEATURE EXTRACTION TEST")
    logger.info("="*80)

    customer_id = 410376  # High performer
    as_of_date = datetime(2024, 7, 1)

    extractor = ComprehensiveFeatureExtractor()

    try:
        df = extractor.extract_all_features(customer_id, as_of_date, top_n=100)

        logger.info(f"\nFeature extraction complete!")
        logger.info(f"Products: {len(df)}")
        logger.info(f"Features: {len(df.columns)-1}")
        logger.info(f"\nFeature list:")
        for col in df.columns:
            if col != 'product_id':
                logger.info(f"  - {col}")

        logger.info(f"\nSample features for first 5 products:")
        logger.info(df.head().to_string())

        # Save sample
        df.to_csv('results/sample_gbm_features.csv', index=False)
        logger.info(f"\n💾 Sample saved to: results/sample_gbm_features.csv")

    finally:
        extractor.close()


if __name__ == '__main__':
    main()
