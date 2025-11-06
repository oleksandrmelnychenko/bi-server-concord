#!/usr/bin/env python3
"""
Frequency-Only Baseline Recommender

This script implements the simplest possible recommendation strategy:
"Recommend the products the customer has purchased most frequently"

Purpose: Determine if our complex 6-signal hybrid model (39.4% precision)
         is justified compared to this naive baseline.

Expected Results:
- If baseline ≈ 35-38%: Our hybrid adds minimal value → simplify
- If baseline ≈ 25-30%: Our hybrid adds significant value → continue optimization
"""

import pymssql
import json
import logging
from typing import List, Dict, Tuple
from datetime import datetime
from collections import defaultdict

# Configuration
import os

DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
    'as_dict': True
}

AS_OF_DATE = '2024-07-01'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrequencyOnlyRecommender:
    """
    Simple recommender that ranks products by purchase frequency only.

    Logic:
    1. Count how many times each product was purchased before as_of_date
    2. Rank products by count (descending)
    3. Return top N products
    """

    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        """Connect to database"""
        try:
            self.conn = pymssql.connect(**DB_CONFIG)
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def get_customer_purchase_history(self, customer_id: int, as_of_date: str) -> Dict[int, int]:
        """
        Get purchase frequency for each product before as_of_date.

        Returns:
            Dict[product_id -> purchase_count]
        """
        query = f"""
        SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        ORDER BY purchase_count DESC
        """

        cursor = self.conn.cursor()
        cursor.execute(query)

        frequency_dict = {}
        for row in cursor:
            frequency_dict[row['ProductID']] = row['purchase_count']

        cursor.close()
        return frequency_dict

    def get_recommendations(self, customer_id: int, as_of_date: str, top_n: int = 50) -> List[Dict]:
        """
        Generate frequency-based recommendations.

        Returns:
            List of dicts with product_id, score (purchase_count)
        """
        frequency_dict = self.get_customer_purchase_history(customer_id, as_of_date)

        if not frequency_dict:
            logger.warning(f"No purchase history for customer {customer_id}")
            return []

        # Sort by frequency (descending)
        sorted_products = sorted(
            frequency_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top N
        recommendations = [
            {
                'product_id': product_id,
                'score': float(count),
                'rank': idx + 1
            }
            for idx, (product_id, count) in enumerate(sorted_products[:top_n])
        ]

        return recommendations

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Database connection closed")


def get_validation_products(conn, customer_id: int, as_of_date: str) -> List[int]:
    """Get products purchased after as_of_date (validation set)"""
    query = f"""
    SELECT DISTINCT oi.ProductID
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
          AND o.Created >= '{as_of_date}'
          AND oi.ProductID IS NOT NULL
    """

    cursor = conn.cursor()
    cursor.execute(query)
    products = [row['ProductID'] for row in cursor]
    cursor.close()

    return products


def calculate_precision(recommendations: List[Dict], validation_products: List[int]) -> Tuple[float, int]:
    """Calculate precision@50"""
    validation_set = set(validation_products)
    hits = sum(1 for rec in recommendations[:50] if rec['product_id'] in validation_set)
    precision = hits / 50 if len(recommendations) >= 50 else hits / len(recommendations)
    return precision, hits


def load_test_customers() -> List[int]:
    """Load test customers from real_world_validation_results.json"""
    try:
        with open('real_world_validation_results.json', 'r') as f:
            data = json.load(f)
        return [customer['customer_id'] for customer in data['customer_results']]
    except FileNotFoundError:
        logger.warning("real_world_validation_results.json not found, using default customers")
        return [411726, 410849, 410282, 410204, 411317, 414304, 410962, 411431, 411450, 410833]


def main():
    logger.info("=" * 80)
    logger.info("FREQUENCY-ONLY BASELINE RECOMMENDER")
    logger.info("=" * 80)
    logger.info(f"Strategy: Recommend most frequently purchased products")
    logger.info(f"As-of date: {AS_OF_DATE}")
    logger.info("")

    # Initialize recommender
    recommender = FrequencyOnlyRecommender()

    try:
        # Load test customers
        test_customers = load_test_customers()
        logger.info(f"Testing on {len(test_customers)} customers")
        logger.info("")

        # Test each customer
        results = []
        segment_results = defaultdict(list)

        for idx, customer_id in enumerate(test_customers, 1):
            logger.info(f"[{idx}/{len(test_customers)}] Testing customer {customer_id}...")

            # Get recommendations
            recommendations = recommender.get_recommendations(
                customer_id=customer_id,
                as_of_date=AS_OF_DATE,
                top_n=50
            )

            # Get validation products
            validation_products = get_validation_products(
                recommender.conn,
                customer_id,
                AS_OF_DATE
            )

            # Calculate precision
            precision, hits = calculate_precision(recommendations, validation_products)

            # Determine segment (simplified - based on training orders)
            query = f"""
            SELECT COUNT(DISTINCT o.ID) as orders_before
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            WHERE ca.ClientID = {customer_id}
                  AND o.Created < '{AS_OF_DATE}'
            """
            cursor = recommender.conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            orders_before = row['orders_before'] if row else 0
            cursor.close()

            if orders_before >= 500:
                segment = "HEAVY"
            elif orders_before >= 100:
                segment = "REGULAR"
            else:
                segment = "LIGHT"

            result = {
                'customer_id': customer_id,
                'segment': segment,
                'precision': precision,
                'hits': hits,
                'validation_products': len(validation_products),
                'recommendations': len(recommendations)
            }

            results.append(result)
            segment_results[segment].append(precision)

            logger.info(f"  Segment: {segment}")
            logger.info(f"  Precision@50: {precision:.1%} ({hits}/50 hits)")
            logger.info(f"  Validation products: {len(validation_products)}")
            logger.info("")

        # Calculate overall statistics
        logger.info("=" * 80)
        logger.info("BASELINE RESULTS")
        logger.info("=" * 80)

        overall_precision = sum(r['precision'] for r in results) / len(results)
        logger.info(f"Overall Precision@50: {overall_precision:.1%}")
        logger.info("")

        # Segment breakdown
        logger.info("SEGMENT BREAKDOWN:")
        for segment in ['HEAVY', 'REGULAR', 'LIGHT']:
            if segment in segment_results:
                precisions = segment_results[segment]
                avg_precision = sum(precisions) / len(precisions)
                logger.info(f"  {segment}: {avg_precision:.1%} (n={len(precisions)})")

        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPARISON TO HYBRID MODEL")
        logger.info("=" * 80)
        logger.info(f"Frequency-Only Baseline: {overall_precision:.1%}")
        logger.info(f"6-Signal Hybrid Model:   39.4%")

        difference = overall_precision - 0.394
        difference_pct = (difference / 0.394) * 100

        if overall_precision >= 0.38:
            logger.info("")
            logger.info("⚠️  BASELINE IS COMPARABLE TO HYBRID")
            logger.info(f"   Difference: {difference:+.1%} ({difference_pct:+.1f}%)")
            logger.info("")
            logger.info("IMPLICATION: Complex 6-signal model adds minimal value")
            logger.info("RECOMMENDATION: Simplify to frequency + 1-2 other signals")
        elif overall_precision >= 0.30:
            logger.info("")
            logger.info("✓ HYBRID SHOWS MODERATE IMPROVEMENT")
            logger.info(f"   Difference: {difference:+.1%} ({difference_pct:+.1f}%)")
            logger.info("")
            logger.info("IMPLICATION: Additional signals provide some value")
            logger.info("RECOMMENDATION: Continue with segment-specific optimization")
        else:
            logger.info("")
            logger.info("✓ HYBRID SHOWS SIGNIFICANT IMPROVEMENT")
            logger.info(f"   Difference: {difference:+.1%} ({difference_pct:+.1f}%)")
            logger.info("")
            logger.info("IMPLICATION: Complex model is justified")
            logger.info("RECOMMENDATION: Focus on segment-specific tuning")

        logger.info("")

        # Save results
        output = {
            'meta': {
                'test_date': datetime.now().isoformat(),
                'as_of_date': AS_OF_DATE,
                'strategy': 'frequency_only',
                'customers_tested': len(test_customers)
            },
            'results': results,
            'summary': {
                'overall_precision': overall_precision,
                'hybrid_precision': 0.394,
                'difference': difference,
                'difference_pct': difference_pct
            },
            'segment_summary': {
                segment: {
                    'count': len(precisions),
                    'avg_precision': sum(precisions) / len(precisions)
                }
                for segment, precisions in segment_results.items()
            }
        }

        output_file = 'frequency_only_baseline_results.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"✓ Results saved to {output_file}")

        # Customer-by-customer comparison table
        logger.info("")
        logger.info("=" * 80)
        logger.info("CUSTOMER-BY-CUSTOMER COMPARISON")
        logger.info("=" * 80)
        logger.info(f"{'Customer':<10} {'Segment':<10} {'Freq-Only':<12} {'Hybrid':<12} {'Difference'}")
        logger.info("-" * 80)

        # Load hybrid results for comparison
        try:
            with open('real_world_validation_results.json', 'r') as f:
                hybrid_data = json.load(f)
            hybrid_dict = {c['customer_id']: c['precision'] for c in hybrid_data['customer_results']}
        except:
            hybrid_dict = {}

        for result in sorted(results, key=lambda x: x['precision'], reverse=True):
            customer_id = result['customer_id']
            segment = result['segment']
            freq_precision = result['precision']
            hybrid_precision = hybrid_dict.get(customer_id, 0)
            diff = freq_precision - hybrid_precision

            logger.info(
                f"{customer_id:<10} {segment:<10} "
                f"{freq_precision:>10.1%}  {hybrid_precision:>10.1%}  "
                f"{diff:>+10.1%}"
            )

        logger.info("")

    finally:
        recommender.close()


if __name__ == '__main__':
    main()
