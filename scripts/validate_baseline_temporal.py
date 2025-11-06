#!/usr/bin/env python3
"""
Validate Frequency Baseline with Proper Temporal Split

Query MSSQL directly for training data (before split) and validation data (after split)
to avoid temporal contamination from aggregated DuckDB data.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymssql
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

SPLIT_DATE = "2024-06-30"
TEST_CUSTOMERS = [410376, 411706, 410767, 410258]


class TemporalBaselineValidator:
    """Validate frequency baseline with proper temporal split"""

    def __init__(self):
        self.conn = None

    def connect_mssql(self):
        """Connect to MSSQL"""
        logger.info(f"Connecting to MSSQL...")
        self.conn = pymssql.connect(
            server=MSSQL_CONFIG['host'],
            port=int(MSSQL_CONFIG['port']),
            user=MSSQL_CONFIG['user'],
            password=MSSQL_CONFIG['password'],
            database=MSSQL_CONFIG['database'],
            tds_version='7.0'
        )
        logger.info("✓ Connected")

    def get_customer_purchases(self, customer_id, start_date=None, end_date=None):
        """Get purchases for a customer"""
        where_clauses = []
        where_clauses.append(f"ca.ClientID = {customer_id}")

        if start_date:
            where_clauses.append(f"o.Created >= '{start_date}'")
        if end_date:
            where_clauses.append(f"o.Created < '{end_date}'")

        query = f"""
        SELECT
            ca.ClientID as customer_id,
            oi.ProductID as product_id,
            o.Created as order_date,
            oi.Qty as quantity,
            oi.PricePerItem * oi.Qty as total_price
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE {' AND '.join(where_clauses)}
        ORDER BY o.Created ASC
        """

        df = pd.read_sql(query, self.conn)
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df

    def generate_frequency_recommendations(self, training_data, top_n=50):
        """
        Generate recommendations based on purchase frequency in training data

        Args:
            training_data: DataFrame with columns [product_id, quantity, total_price]
            top_n: Number of recommendations

        Returns:
            List of recommended product IDs
        """
        # Aggregate by product
        agg = training_data.groupby('product_id').agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'order_date': ['min', 'max', 'count']
        }).reset_index()

        agg.columns = ['product_id', 'total_qty', 'total_spent', 'first_date', 'last_date', 'num_orders']

        # Calculate days since last purchase
        max_date = training_data['order_date'].max()
        agg['days_since_last'] = (max_date - agg['last_date']).dt.days

        # Score: frequency (70%) + recency (20%) + monetary (10%)
        agg['freq_score'] = agg['num_orders'] / agg['num_orders'].max()
        agg['recency_score'] = 1 - (agg['days_since_last'] / agg['days_since_last'].max())
        agg['monetary_score'] = agg['total_spent'] / agg['total_spent'].max()

        agg['final_score'] = (
            agg['freq_score'] * 0.7 +
            agg['recency_score'] * 0.2 +
            agg['monetary_score'] * 0.1
        )

        # Sort by score and return top N
        top_products = agg.nlargest(top_n, 'final_score')
        return top_products['product_id'].astype(str).tolist()

    def test_customer(self, customer_id, top_n=50):
        """Test baseline for one customer with proper temporal split"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Customer {customer_id}")
        logger.info(f"{'='*80}")

        # Get training data (before split)
        train_df = self.get_customer_purchases(
            customer_id,
            start_date=None,
            end_date=SPLIT_DATE
        )
        logger.info(f"Training: {len(train_df)} purchases, {train_df['product_id'].nunique()} unique products")

        if train_df.empty:
            logger.warning(f"No training data for customer {customer_id}")
            return None

        # Generate recommendations
        recommendations = self.generate_frequency_recommendations(train_df, top_n=top_n)
        logger.info(f"Generated {len(recommendations)} recommendations")

        # Get validation data (after split)
        validation_df = self.get_customer_purchases(
            customer_id,
            start_date=SPLIT_DATE,
            end_date='2025-01-01'
        )
        actual_products = set(validation_df['product_id'].astype(str).unique())
        logger.info(f"Validation: {len(validation_df)} purchases, {len(actual_products)} unique products")

        # Calculate metrics
        recommended_set = set(recommendations)
        hits = recommended_set & actual_products

        hit_rate = 1.0 if len(hits) > 0 else 0.0
        precision = len(hits) / len(recommended_set) if len(recommended_set) > 0 else 0.0
        recall = len(hits) / len(actual_products) if len(actual_products) > 0 else 0.0

        logger.info(f"\n📊 RESULTS:")
        logger.info(f"  Hit Rate: {hit_rate:.1%} ({'✅ HIT' if hit_rate > 0 else '❌ MISS'})")
        logger.info(f"  Precision@{top_n}: {precision:.2%} ({len(hits)}/{len(recommended_set)})")
        logger.info(f"  Recall@{top_n}: {recall:.2%} ({len(hits)}/{len(actual_products)})")

        if hits:
            logger.info(f"\n  ✅ CORRECT PREDICTIONS (first 10):")
            for i, product_id in enumerate(list(hits)[:10], 1):
                logger.info(f"     {i}. Product {product_id}")
            if len(hits) > 10:
                logger.info(f"     ... and {len(hits) - 10} more")

        return {
            'customer_id': customer_id,
            'top_n': top_n,
            'train_products': train_df['product_id'].nunique(),
            'validation_products': len(actual_products),
            'num_hits': len(hits),
            'hit_rate': hit_rate,
            'precision': precision,
            'recall': recall,
            'hits': list(hits)
        }

    def run_validation(self, top_n=50):
        """Run validation on all test customers"""
        logger.info("="*80)
        logger.info("FREQUENCY BASELINE TEMPORAL VALIDATION")
        logger.info("="*80)
        logger.info(f"Split Date: {SPLIT_DATE}")
        logger.info(f"Training: All purchases BEFORE {SPLIT_DATE}")
        logger.info(f"Validation: Purchases from {SPLIT_DATE} to 2025-01-01")
        logger.info(f"Top-N: {top_n} recommendations\n")

        self.connect_mssql()

        results = []
        for customer_id in TEST_CUSTOMERS:
            result = self.test_customer(customer_id, top_n=top_n)
            if result:
                results.append(result)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("OVERALL SUMMARY")
        logger.info("="*80)

        df = pd.DataFrame(results)
        avg_hit_rate = df['hit_rate'].mean()
        avg_precision = df['precision'].mean()
        avg_recall = df['recall'].mean()
        total_hits = df['num_hits'].sum()

        logger.info(f"\nCustomers Tested: {len(results)}")
        logger.info(f"Average Hit Rate: {avg_hit_rate:.1%}")
        logger.info(f"Average Precision@{top_n}: {avg_precision:.2%}")
        logger.info(f"Average Recall@{top_n}: {avg_recall:.2%}")
        logger.info(f"Total Hits: {total_hits}")

        if avg_hit_rate > 0:
            logger.info(f"\n✅ **FREQUENCY BASELINE WORKS!**")
            logger.info(f"   {int(avg_hit_rate * 100)}% of customers got useful recommendations")
            logger.info(f"   Simple frequency beats complex ML (which got 0%)")
        else:
            logger.info(f"\n⚠️  Still 0% - fundamental data issue")

        return results


def main():
    validator = TemporalBaselineValidator()
    validator.run_validation(top_n=50)


if __name__ == "__main__":
    main()
