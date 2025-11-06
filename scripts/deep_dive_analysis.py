#!/usr/bin/env python3
"""
Deep Dive Analysis - Why 0% Hit Rate?

This script investigates the root cause of 0% validation hit rate by analyzing:
1. What products were recommended vs what was actually purchased
2. Whether customers have ANY repurchase behavior
3. Product catalog size and diversity
4. Baseline random prediction success rate
5. Whether the task is fundamentally solvable
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymssql
from datetime import datetime
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

# Test customer IDs
TEST_CUSTOMERS = [410376, 411706, 410767, 410258]


class DeepDiveAnalyzer:
    """Analyze why ML validation failed"""

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
            o.ID as order_id
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE {' AND '.join(where_clauses)}
        ORDER BY o.Created ASC
        """

        df = pd.read_sql(query, self.conn)
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df

    def analyze_repurchase_behavior(self, customer_id):
        """
        Analyze if customer has ANY repurchase behavior
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"REPURCHASE BEHAVIOR ANALYSIS - Customer {customer_id}")
        logger.info(f"{'='*80}")

        # Get all purchases before and after split
        train = self.get_customer_purchases(customer_id, end_date=SPLIT_DATE)
        validation = self.get_customer_purchases(customer_id, start_date=SPLIT_DATE, end_date='2025-01-01')

        train_products = set(train['product_id'].unique())
        validation_products = set(validation['product_id'].unique())

        # Check overlap (repurchased products)
        repurchased = train_products & validation_products
        new_products = validation_products - train_products

        logger.info(f"\nH1 2024 (training):")
        logger.info(f"  Orders: {train['order_id'].nunique()}")
        logger.info(f"  Items purchased: {len(train)}")
        logger.info(f"  Unique products: {len(train_products)}")

        logger.info(f"\nH2 2024 (validation):")
        logger.info(f"  Orders: {validation['order_id'].nunique()}")
        logger.info(f"  Items purchased: {len(validation)}")
        logger.info(f"  Unique products: {len(validation_products)}")

        logger.info(f"\nREPURCHASE ANALYSIS:")
        logger.info(f"  Products repurchased: {len(repurchased)} ({len(repurchased)/len(validation_products)*100:.1f}% of H2 purchases)")
        logger.info(f"  New products: {len(new_products)} ({len(new_products)/len(validation_products)*100:.1f}% of H2 purchases)")

        if len(repurchased) > 0:
            logger.info(f"\n✅ Customer DOES repurchase products!")
            logger.info(f"  If we recommended ALL {len(train_products)} previously-purchased products,")
            logger.info(f"  we would have hit {len(repurchased)} out of {len(validation_products)} ({len(repurchased)/len(validation_products)*100:.1f}%)")
            logger.info(f"\n  But we only recommend 20 products, so max possible recall is:")
            logger.info(f"  {min(20, len(repurchased))}/{len(validation_products)} = {min(20, len(repurchased))/len(validation_products)*100:.1f}%")
        else:
            logger.info(f"\n❌ Customer does NOT repurchase any products!")
            logger.info(f"  Every purchase in H2 was a NEW product they'd never bought before")
            logger.info(f"  Repurchase model cannot work for this customer")

        # Check frequency of repurchases
        if len(repurchased) > 0:
            repurchase_counts = {}
            all_purchases = pd.concat([train, validation])
            for product_id in repurchased:
                count = len(all_purchases[all_purchases['product_id'] == product_id])
                repurchase_counts[product_id] = count

            sorted_counts = sorted(repurchase_counts.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"\nTOP 10 MOST FREQUENTLY REPURCHASED:")
            for product_id, count in sorted_counts[:10]:
                logger.info(f"  Product {product_id}: {count} purchases")

        return {
            'train_products': len(train_products),
            'validation_products': len(validation_products),
            'repurchased': len(repurchased),
            'new_products': len(new_products),
            'repurchase_rate': len(repurchased) / len(validation_products) if len(validation_products) > 0 else 0
        }

    def analyze_all_customers(self):
        """Analyze repurchase behavior for all test customers"""
        logger.info("\n" + "="*80)
        logger.info("ANALYZING ALL TEST CUSTOMERS")
        logger.info("="*80)

        results = []
        for customer_id in TEST_CUSTOMERS:
            result = self.analyze_repurchase_behavior(customer_id)
            result['customer_id'] = customer_id
            results.append(result)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)

        df = pd.DataFrame(results)
        logger.info(f"\nAverage repurchase rate: {df['repurchase_rate'].mean():.1%}")
        logger.info(f"Customers with 0% repurchase: {len(df[df['repurchase_rate'] == 0])}/{len(df)}")
        logger.info(f"Customers with >50% repurchase: {len(df[df['repurchase_rate'] > 0.5])}/{len(df)}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Per Customer:")
        for _, row in df.iterrows():
            logger.info(f"  Customer {row['customer_id']}: "
                       f"{row['repurchased']}/{row['validation_products']} repurchased "
                       f"({row['repurchase_rate']:.1%})")

        return results

    def calculate_baseline_metrics(self):
        """
        Calculate baseline metrics to understand task difficulty

        Question: If we recommended 20 RANDOM products, what hit rate would we get?
        """
        logger.info("\n" + "="*80)
        logger.info("BASELINE RANDOM RECOMMENDATION METRICS")
        logger.info("="*80)

        # Get total number of products in catalog
        query = "SELECT COUNT(DISTINCT ID) as total_products FROM dbo.Product"
        result = pd.read_sql(query, self.conn)
        total_products = result['total_products'].iloc[0]

        logger.info(f"\nTotal products in catalog: {total_products:,}")

        # For each test customer, calculate random baseline
        for customer_id in TEST_CUSTOMERS:
            validation = self.get_customer_purchases(customer_id, start_date=SPLIT_DATE, end_date='2025-01-01')
            validation_products = validation['product_id'].nunique()

            # Probability of hitting at least 1 product if we recommend 20 random products
            # P(hit) = 1 - P(all miss)
            # P(all miss) = (catalog_size - validation_products) / catalog_size * ... (20 times)

            # Simplified: P(hit) ≈ 20 * validation_products / catalog_size (assuming no overlap)
            baseline_precision = (20 * validation_products) / (total_products ** 2) * 100
            baseline_hit_rate = min(1.0, (20 * validation_products) / total_products)

            logger.info(f"\nCustomer {customer_id}:")
            logger.info(f"  Validation products: {validation_products}")
            logger.info(f"  Random recommendation baseline:")
            logger.info(f"    Expected hit rate: {baseline_hit_rate:.2%}")
            logger.info(f"    Expected precision: {baseline_precision:.3%}")

    def investigate_ml_recommendations(self, customer_id):
        """
        Get actual ML recommendations and compare with actual purchases
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ML RECOMMENDATIONS vs ACTUAL PURCHASES - Customer {customer_id}")
        logger.info(f"{'='*80}")

        # Import recommendation engine
        sys.path.insert(0, str(Path(__file__).parent))
        from predict_recommendations import RecommendationEngine

        # Get ML recommendations
        engine = RecommendationEngine()
        engine.load_models()
        recommendations = engine.get_recommendations(str(customer_id), top_n=20)

        # Get actual purchases
        validation = self.get_customer_purchases(customer_id, start_date=SPLIT_DATE, end_date='2025-01-01')
        actual_products = set(validation['product_id'].astype(str).unique())

        logger.info(f"\nML RECOMMENDED (Top 20):")
        for i, rec in enumerate(recommendations[:20], 1):
            logger.info(f"  {i}. Product {rec['product_id']} | {rec['type']} | Score: {rec['score']:.3f}")

        logger.info(f"\nACTUAL PURCHASES (first 20 of {len(actual_products)} unique products):")
        for i, product_id in enumerate(list(actual_products)[:20], 1):
            logger.info(f"  {i}. Product {product_id}")

        # Check overlap
        recommended_ids = set([str(rec['product_id']) for rec in recommendations])
        overlap = recommended_ids & actual_products

        if overlap:
            logger.info(f"\n✅ OVERLAP FOUND: {overlap}")
        else:
            logger.info(f"\n❌ NO OVERLAP - Not a single recommended product was purchased")

    def run_deep_dive(self):
        """Run full deep dive analysis"""
        self.connect_mssql()

        # Step 1: Analyze repurchase behavior
        self.analyze_all_customers()

        # Step 2: Calculate baseline metrics
        self.calculate_baseline_metrics()

        # Step 3: Investigate ML recommendations for one customer
        self.investigate_ml_recommendations(TEST_CUSTOMERS[0])

        logger.info("\n" + "="*80)
        logger.info("CONCLUSIONS")
        logger.info("="*80)
        logger.info("""
1. Check repurchase rates above
2. If customers have <20% repurchase rate, repurchase model will struggle
3. With huge product catalogs (>10K products), random baseline is <1% hit rate
4. ML must be MUCH better than random to be useful
5. Consider:
   - Recommending product categories instead of specific products
   - Using complementary products (bought together)
   - Focusing on high-frequency repurchase items only
   - Increasing top-N from 20 to 100
""")


def main():
    analyzer = DeepDiveAnalyzer()
    analyzer.run_deep_dive()


if __name__ == "__main__":
    main()
