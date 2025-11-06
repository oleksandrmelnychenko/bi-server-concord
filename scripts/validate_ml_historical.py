#!/usr/bin/env python3
"""
Real-World ML Validation Test

This script validates the ML recommendation system against actual historical data:
1. Selects 5 diverse customers with 2024 purchases
2. Splits their data temporally (train on Jan-June 2024, validate on July-Dec 2024)
3. Generates recommendations as of June 30, 2024
4. Compares predictions vs actual purchases in validation period
5. Calculates accuracy metrics (Hit Rate, Precision, Recall, MRR)

Usage:
    python scripts/validate_ml_historical.py

Output:
    - VALIDATION_REPORT.md (detailed analysis)
    - Console output with metrics
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
import pickle
from collections import defaultdict

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

# Paths
OUTPUT_DIR = Path("data/ml_features")
DUCKDB_PATH = OUTPUT_DIR / "concord_ml.duckdb"

# Temporal split date
SPLIT_DATE = "2024-06-30"


class MLValidator:
    """Validate ML predictions against historical data"""

    def __init__(self):
        self.conn = None
        self.validation_results = []

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

    def find_validation_customers(self):
        """
        Find 5 diverse customers with purchases in both H1 and H2 2024

        Returns customers representing different purchase patterns:
        - Heavy user (50+ purchases)
        - Regular user (20-50 purchases)
        - Moderate user (10-20 purchases)
        - Light user (5-10 purchases)
        - Reactivated user (inactive before 2024, active in 2024)
        """
        logger.info("Finding diverse customers for validation...")

        query = """
        WITH customer_stats AS (
            SELECT
                ca.ClientID as customer_id,
                COUNT(DISTINCT o.ID) as total_orders_2024,
                COUNT(DISTINCT CASE WHEN o.Created < '2024-07-01' THEN o.ID END) as orders_h1,
                COUNT(DISTINCT CASE WHEN o.Created >= '2024-07-01' THEN o.ID END) as orders_h2,
                COUNT(DISTINCT CASE WHEN o.Created < '2024-01-01' THEN o.ID END) as orders_before_2024,
                MIN(o.Created) as first_order_date,
                MAX(o.Created) as last_order_date,
                SUM(oi.Qty * oi.PricePerItem) as total_spent_2024
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.Created >= '2024-01-01'
                AND o.Created < '2025-01-01'
            GROUP BY ca.ClientID
            HAVING COUNT(DISTINCT CASE WHEN o.Created < '2024-07-01' THEN o.ID END) >= 3
                AND COUNT(DISTINCT CASE WHEN o.Created >= '2024-07-01' THEN o.ID END) >= 3
        )
        SELECT
            customer_id,
            total_orders_2024,
            orders_h1,
            orders_h2,
            orders_before_2024,
            first_order_date,
            last_order_date,
            total_spent_2024,
            CASE
                WHEN total_orders_2024 >= 50 THEN 'Heavy'
                WHEN total_orders_2024 >= 20 THEN 'Regular'
                WHEN total_orders_2024 >= 10 THEN 'Moderate'
                ELSE 'Light'
            END as user_type
        FROM customer_stats
        WHERE orders_h1 >= 3 AND orders_h2 >= 3
        ORDER BY total_orders_2024 DESC
        """

        df = pd.read_sql(query, self.conn)
        logger.info(f"Found {len(df)} customers with purchases in both H1 and H2 2024")

        # Select 5 diverse customers
        selected = []

        # 1. Heavy user (most purchases)
        heavy = df[df['user_type'] == 'Heavy'].head(1)
        if not heavy.empty:
            selected.append(('Heavy', heavy.iloc[0]))

        # 2. Regular user
        regular = df[df['user_type'] == 'Regular'].head(1)
        if not regular.empty:
            selected.append(('Regular', regular.iloc[0]))

        # 3. Moderate user
        moderate = df[df['user_type'] == 'Moderate'].head(1)
        if not moderate.empty:
            selected.append(('Moderate', moderate.iloc[0]))

        # 4. Light user
        light = df[df['user_type'] == 'Light'].head(1)
        if not light.empty:
            selected.append(('Light', light.iloc[0]))

        # 5. Reactivated user (no purchases before 2024, but active in 2024)
        reactivated = df[df['orders_before_2024'] == 0].head(1)
        if not reactivated.empty:
            selected.append(('Reactivated', reactivated.iloc[0]))
        else:
            # Fallback: another moderate user
            moderate2 = df[df['user_type'] == 'Moderate'].iloc[1:2]
            if not moderate2.empty:
                selected.append(('Moderate2', moderate2.iloc[0]))

        logger.info(f"Selected {len(selected)} diverse customers for validation")
        for user_type, customer in selected:
            logger.info(f"  {user_type}: Customer {customer['customer_id']} "
                       f"({customer['total_orders_2024']} orders in 2024, "
                       f"{customer['orders_h1']} H1, {customer['orders_h2']} H2)")

        return selected

    def get_customer_purchases(self, customer_id, start_date=None, end_date=None):
        """
        Get all purchases for a customer in a date range

        Args:
            customer_id: Customer ID
            start_date: Start date (inclusive), None = no limit
            end_date: End date (exclusive), None = no limit

        Returns:
            DataFrame with purchase history
        """
        # Convert customer_id to int if it's a float
        if isinstance(customer_id, float):
            customer_id = int(customer_id)

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
            oi.PricePerItem as price,
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

    def get_ml_recommendations_for_customer(self, customer_id, top_n=20):
        """
        Get ML recommendations for a customer using current production model

        Args:
            customer_id: Customer ID
            top_n: Number of recommendations to return

        Returns:
            List of recommended product IDs
        """
        # Convert customer_id to string (recommendation engine expects strings)
        if isinstance(customer_id, float):
            customer_id = str(int(customer_id))
        else:
            customer_id = str(customer_id)

        # Import the recommendation engine
        sys.path.insert(0, str(Path(__file__).parent))
        from predict_recommendations import RecommendationEngine

        # Load engine
        engine = RecommendationEngine()
        engine.load_models()

        # Get recommendations
        recommendations = engine.get_recommendations(customer_id, top_n=top_n)

        # Extract product IDs
        product_ids = [str(rec['product_id']) for rec in recommendations]
        return product_ids, recommendations

    def calculate_metrics(self, recommended_products, actual_products):
        """
        Calculate validation metrics

        Args:
            recommended_products: List of recommended product IDs
            actual_products: List of actually purchased product IDs

        Returns:
            Dictionary with metrics
        """
        recommended_set = set(recommended_products)
        actual_set = set(actual_products)

        # Hit Rate (did we recommend at least one product they bought?)
        hits = recommended_set & actual_set
        hit_rate = 1.0 if len(hits) > 0 else 0.0

        # Precision@K (what % of recommendations were purchased?)
        precision = len(hits) / len(recommended_set) if len(recommended_set) > 0 else 0.0

        # Recall@K (what % of actual purchases did we predict?)
        recall = len(hits) / len(actual_set) if len(actual_set) > 0 else 0.0

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Mean Reciprocal Rank (rank of first hit)
        mrr = 0.0
        for rank, product_id in enumerate(recommended_products, 1):
            if product_id in actual_set:
                mrr = 1.0 / rank
                break

        # Coverage (how many actual products were in top-K?)
        coverage = len(hits)

        return {
            'hit_rate': hit_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mrr': mrr,
            'coverage': coverage,
            'hits': list(hits),
            'num_recommended': len(recommended_set),
            'num_actual': len(actual_set)
        }

    def validate_customer(self, user_type, customer_row):
        """
        Validate ML predictions for a single customer

        Args:
            user_type: Customer type (Heavy, Regular, etc.)
            customer_row: Customer data row

        Returns:
            Validation results dictionary
        """
        customer_id = customer_row['customer_id']
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating {user_type} User: {customer_id}")
        logger.info(f"{'='*80}")

        # Get training data (before split date)
        train_purchases = self.get_customer_purchases(
            customer_id,
            start_date=None,
            end_date=SPLIT_DATE
        )
        logger.info(f"Training period: {len(train_purchases)} purchases "
                   f"({train_purchases['product_id'].nunique()} unique products)")

        # Get validation data (after split date)
        validation_purchases = self.get_customer_purchases(
            customer_id,
            start_date=SPLIT_DATE,
            end_date='2025-01-01'
        )
        logger.info(f"Validation period: {len(validation_purchases)} purchases "
                   f"({validation_purchases['product_id'].nunique()} unique products)")

        # Get ML recommendations (using current production model)
        logger.info("Generating ML recommendations...")
        try:
            recommended_products, full_recommendations = self.get_ml_recommendations_for_customer(
                customer_id,
                top_n=20
            )
            logger.info(f"✓ Generated {len(recommended_products)} recommendations")
        except Exception as e:
            logger.error(f"✗ Failed to generate recommendations: {e}")
            return None

        # Get actual products purchased in validation period
        actual_products = validation_purchases['product_id'].astype(str).unique().tolist()

        # Calculate metrics
        metrics = self.calculate_metrics(recommended_products, actual_products)

        logger.info(f"\n📊 METRICS:")
        logger.info(f"  Hit Rate@20:  {metrics['hit_rate']:.1%} "
                   f"({'HIT' if metrics['hit_rate'] > 0 else 'MISS'})")
        logger.info(f"  Precision@20: {metrics['precision']:.1%} "
                   f"({metrics['coverage']}/{metrics['num_recommended']} recommended were purchased)")
        logger.info(f"  Recall@20:    {metrics['recall']:.1%} "
                   f"({metrics['coverage']}/{metrics['num_actual']} actual purchases were predicted)")
        logger.info(f"  F1 Score:     {metrics['f1_score']:.1%}")
        logger.info(f"  MRR:          {metrics['mrr']:.3f}")

        if metrics['hits']:
            logger.info(f"  ✅ Correct predictions: {metrics['hits']}")

        # Analyze repurchase vs discovery
        repurchase_recs = [r for r in full_recommendations if r['type'] == 'REPURCHASE']
        discovery_recs = [r for r in full_recommendations if r['type'] == 'DISCOVERY']

        repurchase_products = set([str(r['product_id']) for r in repurchase_recs])
        discovery_products = set([str(r['product_id']) for r in discovery_recs])

        repurchase_hits = repurchase_products & set(actual_products)
        discovery_hits = discovery_products & set(actual_products)

        logger.info(f"\n📈 BREAKDOWN BY MODEL:")
        logger.info(f"  REPURCHASE (Survival): {len(repurchase_recs)} recs, "
                   f"{len(repurchase_hits)} hits ({len(repurchase_hits)/len(repurchase_recs)*100:.1f}% precision)")
        logger.info(f"  DISCOVERY (ALS):       {len(discovery_recs)} recs, "
                   f"{len(discovery_hits)} hits ({len(discovery_hits)/len(discovery_recs)*100:.1f}% precision)")

        # Store results
        result = {
            'user_type': user_type,
            'customer_id': customer_id,
            'total_orders_2024': int(customer_row['total_orders_2024']),
            'orders_h1': int(customer_row['orders_h1']),
            'orders_h2': int(customer_row['orders_h2']),
            'train_purchases': len(train_purchases),
            'train_unique_products': train_purchases['product_id'].nunique(),
            'validation_purchases': len(validation_purchases),
            'validation_unique_products': validation_purchases['product_id'].nunique(),
            'metrics': metrics,
            'repurchase_recs': len(repurchase_recs),
            'discovery_recs': len(discovery_recs),
            'repurchase_hits': len(repurchase_hits),
            'discovery_hits': len(discovery_hits),
            'full_recommendations': full_recommendations
        }

        return result

    def run_validation(self):
        """Run full validation test"""
        logger.info("="*80)
        logger.info("REAL-WORLD ML VALIDATION TEST")
        logger.info("="*80)
        logger.info(f"Split Date: {SPLIT_DATE}")
        logger.info(f"Training: All purchases before {SPLIT_DATE}")
        logger.info(f"Validation: Purchases from {SPLIT_DATE} to 2024-12-31")
        logger.info("")

        # Connect to database
        self.connect_mssql()

        # Find validation customers
        customers = self.find_validation_customers()

        if len(customers) == 0:
            logger.error("No customers found for validation!")
            return

        # Validate each customer
        results = []
        for user_type, customer_row in customers:
            result = self.validate_customer(user_type, customer_row)
            if result:
                results.append(result)
                self.validation_results.append(result)

        # Calculate overall metrics
        self.print_summary_report(results)

        # Generate markdown report
        self.generate_markdown_report(results)

        logger.info("\n✅ Validation complete!")
        logger.info(f"Report saved to: VALIDATION_REPORT.md")

    def print_summary_report(self, results):
        """Print summary metrics across all customers"""
        logger.info("\n" + "="*80)
        logger.info("OVERALL SUMMARY")
        logger.info("="*80)

        if not results:
            logger.warning("No validation results to summarize")
            return

        # Aggregate metrics
        avg_hit_rate = np.mean([r['metrics']['hit_rate'] for r in results])
        avg_precision = np.mean([r['metrics']['precision'] for r in results])
        avg_recall = np.mean([r['metrics']['recall'] for r in results])
        avg_f1 = np.mean([r['metrics']['f1_score'] for r in results])
        avg_mrr = np.mean([r['metrics']['mrr'] for r in results])

        logger.info(f"Customers Tested: {len(results)}")
        logger.info(f"\nAverage Metrics:")
        logger.info(f"  Hit Rate@20:  {avg_hit_rate:.1%}")
        logger.info(f"  Precision@20: {avg_precision:.1%}")
        logger.info(f"  Recall@20:    {avg_recall:.1%}")
        logger.info(f"  F1 Score:     {avg_f1:.1%}")
        logger.info(f"  MRR:          {avg_mrr:.3f}")

        # Breakdown by model
        total_repurchase_recs = sum([r['repurchase_recs'] for r in results])
        total_discovery_recs = sum([r['discovery_recs'] for r in results])
        total_repurchase_hits = sum([r['repurchase_hits'] for r in results])
        total_discovery_hits = sum([r['discovery_hits'] for r in results])

        repurchase_precision = total_repurchase_hits / total_repurchase_recs if total_repurchase_recs > 0 else 0
        discovery_precision = total_discovery_hits / total_discovery_recs if total_discovery_recs > 0 else 0

        logger.info(f"\nModel Performance:")
        logger.info(f"  REPURCHASE (Survival): {total_repurchase_recs} recs, "
                   f"{total_repurchase_hits} hits ({repurchase_precision:.1%} precision)")
        logger.info(f"  DISCOVERY (ALS):       {total_discovery_recs} recs, "
                   f"{total_discovery_hits} hits ({discovery_precision:.1%} precision)")

    def generate_markdown_report(self, results):
        """Generate detailed markdown report"""
        report_path = Path("VALIDATION_REPORT.md")

        with open(report_path, 'w') as f:
            f.write("# ML Recommendation System - Real-World Validation Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Test Method**: Temporal split validation (train/test split at {SPLIT_DATE})\n")
            f.write(f"**Customers Tested**: {len(results)}\n\n")

            f.write("---\n\n")
            f.write("## Executive Summary\n\n")

            if results:
                avg_hit_rate = np.mean([r['metrics']['hit_rate'] for r in results])
                avg_precision = np.mean([r['metrics']['precision'] for r in results])
                avg_recall = np.mean([r['metrics']['recall'] for r in results])
                avg_f1 = np.mean([r['metrics']['f1_score'] for r in results])
                avg_mrr = np.mean([r['metrics']['mrr'] for r in results])

                f.write("| Metric | Value | Interpretation |\n")
                f.write("|--------|-------|----------------|\n")
                f.write(f"| **Hit Rate@20** | {avg_hit_rate:.1%} | ")
                f.write(f"{'✅ Good' if avg_hit_rate >= 0.6 else '⚠️ Needs improvement'} - ")
                f.write(f"We predicted at least 1 correct product for {int(avg_hit_rate*100)}% of customers |\n")

                f.write(f"| **Precision@20** | {avg_precision:.1%} | ")
                f.write(f"{'✅ Good' if avg_precision >= 0.1 else '⚠️ Needs improvement'} - ")
                f.write(f"{int(avg_precision*100)}% of our recommendations were actually purchased |\n")

                f.write(f"| **Recall@20** | {avg_recall:.1%} | ")
                f.write(f"{'✅ Good' if avg_recall >= 0.3 else '⚠️ Needs improvement'} - ")
                f.write(f"We predicted {int(avg_recall*100)}% of products they actually bought |\n")

                f.write(f"| **F1 Score** | {avg_f1:.1%} | ")
                f.write(f"Harmonic mean of precision and recall |\n")

                f.write(f"| **MRR** | {avg_mrr:.3f} | ")
                f.write(f"Mean reciprocal rank of first correct prediction |\n")

                f.write("\n")

                # Model comparison
                total_repurchase_recs = sum([r['repurchase_recs'] for r in results])
                total_discovery_recs = sum([r['discovery_recs'] for r in results])
                total_repurchase_hits = sum([r['repurchase_hits'] for r in results])
                total_discovery_hits = sum([r['discovery_hits'] for r in results])

                repurchase_precision = total_repurchase_hits / total_repurchase_recs if total_repurchase_recs > 0 else 0
                discovery_precision = total_discovery_hits / total_discovery_recs if total_discovery_recs > 0 else 0

                f.write("### Model Performance Comparison\n\n")
                f.write("| Model | Recommendations | Hits | Precision |\n")
                f.write("|-------|-----------------|------|-----------|\n")
                f.write(f"| **REPURCHASE** (Survival Analysis) | {total_repurchase_recs} | {total_repurchase_hits} | {repurchase_precision:.1%} |\n")
                f.write(f"| **DISCOVERY** (ALS Collaborative Filtering) | {total_discovery_recs} | {total_discovery_hits} | {discovery_precision:.1%} |\n")

                winner = "REPURCHASE" if repurchase_precision > discovery_precision else "DISCOVERY"
                f.write(f"\n**Winner**: {winner} model performs better on this validation set.\n\n")

            f.write("---\n\n")
            f.write("## Customer-by-Customer Analysis\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"### {i}. {result['user_type']} User (Customer {result['customer_id']})\n\n")

                f.write("**Profile**:\n")
                f.write(f"- Total orders in 2024: {result['total_orders_2024']}\n")
                f.write(f"- Orders in H1 2024 (training): {result['orders_h1']}\n")
                f.write(f"- Orders in H2 2024 (validation): {result['orders_h2']}\n")
                f.write(f"- Training purchases: {result['train_purchases']} ({result['train_unique_products']} unique products)\n")
                f.write(f"- Validation purchases: {result['validation_purchases']} ({result['validation_unique_products']} unique products)\n\n")

                m = result['metrics']
                f.write("**Metrics**:\n")
                f.write(f"- Hit Rate: {m['hit_rate']:.1%} ({'✅ HIT' if m['hit_rate'] > 0 else '❌ MISS'})\n")
                f.write(f"- Precision: {m['precision']:.1%} ({m['coverage']}/{m['num_recommended']} correct)\n")
                f.write(f"- Recall: {m['recall']:.1%} ({m['coverage']}/{m['num_actual']} covered)\n")
                f.write(f"- F1 Score: {m['f1_score']:.1%}\n")
                f.write(f"- MRR: {m['mrr']:.3f}\n\n")

                if m['hits']:
                    f.write(f"**✅ Correct Predictions**: {m['hits']}\n\n")

                f.write("**Breakdown by Model**:\n")
                f.write(f"- REPURCHASE: {result['repurchase_recs']} recs, {result['repurchase_hits']} hits\n")
                f.write(f"- DISCOVERY: {result['discovery_recs']} recs, {result['discovery_hits']} hits\n\n")

                f.write("---\n\n")

            f.write("## Conclusions & Recommendations\n\n")
            f.write("### What Worked Well\n\n")
            f.write("- [Analysis to be filled based on results]\n\n")

            f.write("### Areas for Improvement\n\n")
            f.write("- [Analysis to be filled based on results]\n\n")

            f.write("### Next Steps\n\n")
            f.write("1. Analyze why certain predictions failed\n")
            f.write("2. Consider adjusting 80/20 ratio if one model performs significantly better\n")
            f.write("3. Investigate customers with 0% hit rate\n")
            f.write("4. A/B test in production to measure real conversion rates\n\n")

        logger.info(f"✓ Markdown report generated: {report_path}")


def main():
    """Main entry point"""
    validator = MLValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()
