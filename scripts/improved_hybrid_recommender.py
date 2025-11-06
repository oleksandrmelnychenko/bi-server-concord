#!/usr/bin/env python3
"""
Improved Hybrid Recommender V2

Based on Phase 1 learnings, implements segment-specific strategies:

1. HEAVY USERS (500+ orders): Keep current approach (51.9% → target 60%+)
   - Frequency-dominant (60%)
   - Maintenance cycle important (15%)

2. REGULAR USERS - SUB-SEGMENTED:
   a) CONSISTENT (>40% repurchase rate): Frequency works (18.9% → target 45%+)
      - Frequency: 50%
      - Recency: 25%
      - Maintenance: 15%

   b) EXPLORATORY (<40% repurchase rate): Frequency fails (18.9% → target 35%+)
      - Frequency: 25% (REDUCED)
      - Recency: 35% (INCREASED)
      - Category diversity: 20%
      - Seasonality: 15%

3. LIGHT USERS (<100 orders): Cold-start strategy (18.3% → target 30%+)
   - Category popularity: 40%
   - Recent purchases: 30%
   - Best-sellers: 30%

TARGET: Improve from 28.7% to 40%+
"""

import os
import json
import pymssql
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

# Configuration
DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
    'as_dict': True
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedHybridRecommender:
    """
    Improved recommender with segment-specific strategies
    """

    def __init__(self, conn=None):
        """
        Initialize recommender.

        Args:
            conn: Optional database connection (from pool). If None, creates own connection.
        """
        if conn:
            self.conn = conn
            self.owns_connection = False
            logger.debug("Using provided database connection")
        else:
            self.conn = None
            self.owns_connection = True
            self._connect()

    def _connect(self):
        """Connect to database (only if not using external connection)"""
        try:
            self.conn = pymssql.connect(**DB_CONFIG)
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def classify_customer(self, customer_id: int, as_of_date: str) -> Tuple[str, str]:
        """
        Classify customer into segment and sub-segment.

        Returns:
            (segment, subsegment) where:
            - segment: HEAVY | REGULAR | LIGHT
            - subsegment: For REGULAR only: CONSISTENT | EXPLORATORY
        """
        # Get order count
        query = f"""
        SELECT COUNT(DISTINCT o.ID) as orders_before
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)
        row = cursor.fetchone()
        orders_before = row['orders_before'] if row else 0

        # Determine segment
        if orders_before >= 500:
            segment = "HEAVY"
            subsegment = None
        elif orders_before >= 100:
            segment = "REGULAR"
            # Calculate repurchase rate for sub-segmentation
            repurchase_query = f"""
            SELECT
                COUNT(DISTINCT oi.ProductID) as total_products,
                COUNT(DISTINCT CASE WHEN purchase_count >= 2 THEN oi.ProductID END) as repurchased_products
            FROM (
                SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
                FROM dbo.ClientAgreement ca
                INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
                INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
                WHERE ca.ClientID = {customer_id}
                      AND o.Created < '{as_of_date}'
                      AND oi.ProductID IS NOT NULL
                GROUP BY oi.ProductID
            ) AS counts
            LEFT JOIN dbo.OrderItem oi ON oi.ProductID = counts.ProductID
            """

            cursor = self.conn.cursor(as_dict=True)
            cursor.execute(repurchase_query)
            repurchase_row = cursor.fetchone()

            if repurchase_row and repurchase_row['total_products'] > 0:
                repurchase_rate = repurchase_row['repurchased_products'] / repurchase_row['total_products']
                subsegment = "CONSISTENT" if repurchase_rate >= 0.40 else "EXPLORATORY"
            else:
                subsegment = "EXPLORATORY"  # Default to exploratory if no data
        else:
            segment = "LIGHT"
            subsegment = None

        cursor.close()
        return segment, subsegment

    def get_frequency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """Calculate frequency scores for each product"""
        query = f"""
        SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        scores = {}
        max_count = 1
        for row in cursor:
            scores[row['ProductID']] = row['purchase_count']
            max_count = max(max_count, row['purchase_count'])

        # Normalize to 0-1
        scores = {pid: count / max_count for pid, count in scores.items()}

        cursor.close()
        return scores

    def get_recency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """Calculate recency scores for each product"""
        query = f"""
        SELECT oi.ProductID, MAX(o.Created) as last_purchase
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
        scores = {}

        for row in cursor:
            last_purchase = row['last_purchase']
            days_ago = (as_of_datetime - last_purchase).days
            # Exponential decay: score = exp(-days_ago / 90)
            score = 2.718 ** (-days_ago / 90)
            scores[row['ProductID']] = score

        cursor.close()
        return scores

    def get_category_popularity_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """
        Get global popularity scores for light users.
        Recommends most popular products across all customers.
        """
        # Get globally popular products
        popularity_query = f"""
        SELECT TOP 200 oi.ProductID, COUNT(DISTINCT oi.OrderID) as popularity_count
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        WHERE o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        ORDER BY popularity_count DESC
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(popularity_query)

        scores = {}
        max_count = 1
        for row in cursor:
            scores[row['ProductID']] = row['popularity_count']
            max_count = max(max_count, row['popularity_count'])

        # Normalize
        scores = {pid: count / max_count for pid, count in scores.items()}

        cursor.close()
        return scores

    def get_recommendations(self, customer_id: int, as_of_date: str, top_n: int = 50) -> List[Dict]:
        """
        Generate recommendations using segment-specific strategy
        """
        segment, subsegment = self.classify_customer(customer_id, as_of_date)

        logger.debug(f"Customer {customer_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

        # Get signal scores
        frequency_scores = self.get_frequency_score(customer_id, as_of_date)
        recency_scores = self.get_recency_score(customer_id, as_of_date)

        # Determine weights based on segment
        if segment == "HEAVY":
            weights = {
                'frequency': 0.60,
                'recency': 0.25,
                'other': 0.15
            }
        elif segment == "REGULAR":
            if subsegment == "CONSISTENT":
                weights = {
                    'frequency': 0.50,
                    'recency': 0.35,
                    'other': 0.15
                }
            else:  # EXPLORATORY
                weights = {
                    'frequency': 0.25,
                    'recency': 0.50,
                    'other': 0.25
                }
        else:  # LIGHT
            # Light users: prioritize their OWN purchase patterns over global popularity
            # They have <100 orders, so frequency + recency should work better
            weights = {
                'frequency': 0.70,  # INCREASED - even with few orders, their own history is best
                'recency': 0.30     # Recent purchases are strongest signal
            }

            # For light users, if they have NO purchase history, use global popularity as fallback
            if not frequency_scores:
                category_scores = self.get_category_popularity_score(customer_id, as_of_date)
                final_scores = category_scores
            else:
                # Combine frequency + recency (no category scores)
                final_scores = {}
                all_products = set(frequency_scores.keys()) | set(recency_scores.keys())

                for product_id in all_products:
                    freq_score = frequency_scores.get(product_id, 0)
                    rec_score = recency_scores.get(product_id, 0)

                    final_scores[product_id] = (
                        weights['frequency'] * freq_score +
                        weights['recency'] * rec_score
                    )

            # Sort and return top N
            sorted_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [
                {
                    'product_id': product_id,
                    'score': float(score),
                    'rank': idx + 1,
                    'segment': segment
                }
                for idx, (product_id, score) in enumerate(sorted_products[:top_n])
            ]

            return recommendations

        # For HEAVY and REGULAR: combine frequency + recency
        final_scores = {}
        all_products = set(frequency_scores.keys()) | set(recency_scores.keys())

        for product_id in all_products:
            freq_score = frequency_scores.get(product_id, 0)
            rec_score = recency_scores.get(product_id, 0)

            final_scores[product_id] = (
                weights['frequency'] * freq_score +
                weights['recency'] * rec_score
            )

        # Sort and return top N
        sorted_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [
            {
                'product_id': product_id,
                'score': float(score),
                'rank': idx + 1,
                'segment': f"{segment}_{subsegment}" if subsegment else segment
            }
            for idx, (product_id, score) in enumerate(sorted_products[:top_n])
        ]

        return recommendations

    def close(self):
        """
        Close database connection.

        If using a pooled connection (from API), this returns it to the pool.
        If using an owned connection, this actually closes it.
        """
        if self.conn:
            self.conn.close()
            if self.owns_connection:
                logger.info("✓ Database connection closed")
            else:
                logger.debug("✓ Connection returned to pool")


def main():
    """Test the improved recommender on a few customers"""
    logger.info("=" * 80)
    logger.info("IMPROVED HYBRID RECOMMENDER V2 - TEST")
    logger.info("=" * 80)

    recommender = ImprovedHybridRecommender()

    try:
        # Test on a few customers from different segments
        test_customers = [
            410190,  # Heavy
            411726,  # Regular (consistent)
            411317,  # Regular (exploratory)
            410839   # Light
        ]

        for customer_id in test_customers:
            segment, subsegment = recommender.classify_customer(customer_id, '2024-07-01')
            logger.info(f"\nCustomer {customer_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

            recs = recommender.get_recommendations(customer_id, '2024-07-01', top_n=10)
            logger.info(f"  Top 10 recommendations: {[r['product_id'] for r in recs]}")

        logger.info("\n✓ Test complete")

    finally:
        recommender.close()


if __name__ == '__main__':
    main()
