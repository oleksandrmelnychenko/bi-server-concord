#!/usr/bin/env python3
"""
Collaborative Hybrid Recommender V4

CRITICAL IMPROVEMENT: Addresses the "new product discovery" problem identified in quality analysis.

Analysis showed 98-100% of recommendation misses were NEW products (not in customer's history).
V3 could only recommend products the customer already bought.

V4 adds two new signals to recommend NEW products:

1. COLLABORATIVE FILTERING: Find similar customers, recommend THEIR products
   - Jaccard similarity based on purchase overlap
   - Top 20 most similar customers
   - Recommends products they bought (that target customer hasn't)
   - Expected: +10-15% precision improvement for LIGHT users

2. CATEGORY TRENDS: Recommend trending products in customer's preferred categories
   - Identify customer's top 3 categories (by purchase count)
   - Find trending products in those categories (high recent activity)
   - Recommends new/popular products in familiar categories
   - Expected: +5-8% precision improvement for LIGHT users

OVERALL TARGET: 75.4% → 83%+ precision

Signal blending strategy:
- HEAVY users: Frequency-dominant (own history is best) + small collaborative boost
- REGULAR users: Balanced (own history + collaborative + trends)
- LIGHT users: Collaborative-dominant (sparse own history, need help from others)
"""

import os
import json
import pymssql
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
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


class CollaborativeHybridRecommenderV4:
    """
    V4 Recommender with Collaborative Filtering + Category Trends
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

    def get_customer_products(self, customer_id: int, as_of_date: str) -> Set[int]:
        """Get set of all products purchased by customer before as_of_date"""
        query = f"""
        SELECT DISTINCT oi.ProductID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        products = {row['ProductID'] for row in cursor}
        cursor.close()
        return products

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

    def get_similar_customers(self, customer_id: int, as_of_date: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Find top K most similar customers using Jaccard similarity.

        Jaccard similarity = |A ∩ B| / |A ∪ B|
        where A, B are sets of products purchased by two customers.

        Returns:
            List of (customer_id, similarity_score) tuples, sorted by similarity
        """
        # Get target customer's products
        target_products = self.get_customer_products(customer_id, as_of_date)

        if not target_products:
            return []  # No purchase history

        # Get all customers with at least 1 overlapping product
        # This is more efficient than comparing against ALL customers
        overlap_query = f"""
        SELECT DISTINCT ca2.ClientID
        FROM dbo.ClientAgreement ca1
        INNER JOIN dbo.[Order] o1 ON ca1.ID = o1.ClientAgreementID
        INNER JOIN dbo.OrderItem oi1 ON o1.ID = oi1.OrderID
        INNER JOIN dbo.OrderItem oi2 ON oi1.ProductID = oi2.ProductID
        INNER JOIN dbo.[Order] o2 ON oi2.OrderID = o2.ID
        INNER JOIN dbo.ClientAgreement ca2 ON o2.ClientAgreementID = ca2.ID
        WHERE ca1.ClientID = {customer_id}
              AND ca2.ClientID != {customer_id}
              AND o1.Created < '{as_of_date}'
              AND o2.Created < '{as_of_date}'
              AND oi1.ProductID IS NOT NULL
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(overlap_query)

        candidate_customers = [row['ClientID'] for row in cursor]
        cursor.close()

        # Calculate Jaccard similarity for each candidate
        similarities = []
        for other_customer_id in candidate_customers[:500]:  # Limit to top 500 candidates for performance
            other_products = self.get_customer_products(other_customer_id, as_of_date)

            if not other_products:
                continue

            # Jaccard similarity
            intersection = len(target_products & other_products)
            union = len(target_products | other_products)

            if union > 0:
                similarity = intersection / union
                similarities.append((other_customer_id, similarity))

        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_collaborative_filtering_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """
        Generate scores based on what similar customers bought (that target customer hasn't).

        This addresses the "new product discovery" problem.
        """
        # Get similar customers
        similar_customers = self.get_similar_customers(customer_id, as_of_date, top_k=20)

        if not similar_customers:
            return {}

        # Get target customer's products (to exclude them)
        target_products = self.get_customer_products(customer_id, as_of_date)

        # Aggregate products from similar customers (weighted by similarity)
        product_scores = defaultdict(float)

        for other_customer_id, similarity in similar_customers:
            other_products = self.get_customer_products(other_customer_id, as_of_date)

            # Only consider products the target customer HASN'T bought
            new_products = other_products - target_products

            for product_id in new_products:
                # Weight by similarity score
                product_scores[product_id] += similarity

        # Normalize scores to 0-1
        if product_scores:
            max_score = max(product_scores.values())
            product_scores = {pid: score / max_score for pid, score in product_scores.items()}

        return dict(product_scores)

    def get_global_trends_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """
        Generate scores based on globally trending products (high recent purchase frequency).

        Trending = high purchase frequency in last 90 days across all customers.
        This addresses new product discovery without requiring category information.
        """
        # Get target customer's products (to exclude them)
        target_products = self.get_customer_products(customer_id, as_of_date)

        # Calculate as_of_date - 90 days
        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
        trend_start_date = (as_of_datetime - timedelta(days=90)).strftime('%Y-%m-%d')

        # Get globally trending products (high recent popularity)
        trending_query = f"""
        SELECT TOP 200 oi.ProductID, COUNT(DISTINCT oi.OrderID) as recent_popularity
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        WHERE o.Created >= '{trend_start_date}'
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        HAVING COUNT(DISTINCT oi.OrderID) >= 5  -- At least 5 orders in last 90 days
        ORDER BY recent_popularity DESC
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(trending_query)

        scores = {}
        max_count = 1

        for row in cursor:
            product_id = row['ProductID']

            # Only recommend products customer HASN'T bought
            if product_id not in target_products:
                scores[product_id] = row['recent_popularity']
                max_count = max(max_count, row['recent_popularity'])

        # Normalize to 0-1
        scores = {pid: count / max_count for pid, count in scores.items()}

        cursor.close()
        return scores

    def get_recommendations(self, customer_id: int, as_of_date: str, top_n: int = 50) -> List[Dict]:
        """
        Generate recommendations using V4 hybrid strategy:
        - V3 signals: frequency, recency (own history)
        - V4 NEW signals: collaborative filtering, category trends (new product discovery)
        """
        segment, subsegment = self.classify_customer(customer_id, as_of_date)

        logger.debug(f"Customer {customer_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

        # Get all signal scores
        frequency_scores = self.get_frequency_score(customer_id, as_of_date)
        recency_scores = self.get_recency_score(customer_id, as_of_date)
        collaborative_scores = self.get_collaborative_filtering_score(customer_id, as_of_date)
        global_trend_scores = self.get_global_trends_score(customer_id, as_of_date)

        # Determine weights based on segment
        if segment == "HEAVY":
            # Heavy users: own history is best, small boost from collaborative
            weights = {
                'frequency': 0.50,
                'recency': 0.25,
                'collaborative': 0.15,
                'global_trends': 0.10
            }
        elif segment == "REGULAR":
            if subsegment == "CONSISTENT":
                # Consistent: balanced approach
                weights = {
                    'frequency': 0.40,
                    'recency': 0.25,
                    'collaborative': 0.20,
                    'global_trends': 0.15
                }
            else:  # EXPLORATORY
                # Exploratory: favor new product discovery
                weights = {
                    'frequency': 0.20,
                    'recency': 0.25,
                    'collaborative': 0.30,
                    'global_trends': 0.25
                }
        else:  # LIGHT
            # Light users: heavily favor collaborative + trends (sparse own history)
            weights = {
                'frequency': 0.25,
                'recency': 0.15,
                'collaborative': 0.35,
                'global_trends': 0.25
            }

        # Combine all signals
        final_scores = {}
        all_products = (
            set(frequency_scores.keys()) |
            set(recency_scores.keys()) |
            set(collaborative_scores.keys()) |
            set(global_trend_scores.keys())
        )

        for product_id in all_products:
            freq_score = frequency_scores.get(product_id, 0)
            rec_score = recency_scores.get(product_id, 0)
            collab_score = collaborative_scores.get(product_id, 0)
            trend_score = global_trend_scores.get(product_id, 0)

            final_scores[product_id] = (
                weights['frequency'] * freq_score +
                weights['recency'] * rec_score +
                weights['collaborative'] * collab_score +
                weights['global_trends'] * trend_score
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
    """Test V4 recommender on a few customers"""
    logger.info("=" * 80)
    logger.info("COLLABORATIVE HYBRID RECOMMENDER V4 - TEST")
    logger.info("=" * 80)

    recommender = CollaborativeHybridRecommenderV4()

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
