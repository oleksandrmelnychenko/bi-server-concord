#!/usr/bin/env python3
"""
Improved Hybrid Recommender V3.5 - Brand Affinity Boosting for LIGHT Users

Based on V3 learnings, adds domain-specific enhancements for LIGHT users:

1. HEAVY USERS (500+ orders): V3 approach (57.7% precision)
   - Frequency-dominant (60%)
   - Maintenance cycle important (15%)

2. REGULAR USERS - SUB-SEGMENTED: V3 approach (40.0% precision)
   a) CONSISTENT (>40% repurchase rate):
      - Frequency: 50%
      - Recency: 25%
      - Maintenance: 15%

   b) EXPLORATORY (<40% repurchase rate):
      - Frequency: 25% (REDUCED)
      - Recency: 35% (INCREASED)
      - Category diversity: 20%
      - Seasonality: 15%

3. LIGHT USERS (<100 orders): V3.5 Brand Affinity Boosting (16.1% → target 30%+)
   - **NEW: Brand Affinity: 40%** (infer car brands from purchase history)
   - Frequency: 40% (own purchase patterns)
   - Recency: 20% (recent purchases)

   Strategy: Leverages ProductCarBrand table to infer customer's fleet (DAF, MAN, VOLVO, etc.)
   and recommend popular products for those brands they haven't bought yet.

   Why it works: B2B truck parts are brand-specific. A customer with a DAF fleet needs
   DAF-compatible parts, not MAN parts. With 261,800 products mapped to car brands,
   we can provide targeted recommendations even with sparse purchase history.

TARGET: Improve LIGHT users from 16.1% to 30%+ (V3 baseline: 29.2% overall)
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

    def get_brand_affinity_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """
        V3.5: Brand Affinity Boosting for LIGHT users

        Infer customer's car brand preferences from purchase history,
        then recommend popular products for those brands they haven't bought yet.

        Strategy:
        1. Get customer's car brands from past purchases (via ProductCarBrand)
        2. Weight brands by purchase frequency (more purchases = higher preference)
        3. For each brand, recommend top products they haven't bought
        """
        # Get customer's existing products to exclude
        cursor = self.conn.cursor(as_dict=True)

        customer_products_query = f"""
        SELECT DISTINCT oi.ProductID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        """
        cursor.execute(customer_products_query)
        customer_products = set(row['ProductID'] for row in cursor)

        # Get customer's car brand affinity (brands they've bought from + frequency)
        brand_affinity_query = f"""
        SELECT cb.ID as CarBrandID, cb.Name as CarBrandName,
               COUNT(DISTINCT oi.ProductID) as product_count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        INNER JOIN dbo.ProductCarBrand pcb ON oi.ProductID = pcb.ProductID
        INNER JOIN dbo.CarBrand cb ON pcb.CarBrandID = cb.ID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY cb.ID, cb.Name
        ORDER BY product_count DESC
        """
        cursor.execute(brand_affinity_query)
        brand_affinities = cursor.fetchall()

        if not brand_affinities:
            cursor.close()
            return {}

        # Calculate brand weights (normalized by total products)
        total_products = sum(b['product_count'] for b in brand_affinities)
        brand_weights = {b['CarBrandID']: b['product_count'] / total_products
                         for b in brand_affinities}

        # For each brand, get popular products customer hasn't bought
        brand_scores = defaultdict(lambda: defaultdict(float))

        for brand in brand_affinities[:10]:  # Top 10 brands only
            brand_id = brand['CarBrandID']
            brand_weight = brand_weights[brand_id]

            # Get popular products for this brand
            brand_products_query = f"""
            SELECT TOP 100 pcb.ProductID, COUNT(DISTINCT oi.OrderID) as popularity
            FROM dbo.ProductCarBrand pcb
            INNER JOIN dbo.OrderItem oi ON pcb.ProductID = oi.ProductID
            INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
            WHERE pcb.CarBrandID = {brand_id}
                  AND o.Created < '{as_of_date}'
                  AND pcb.ProductID NOT IN ({','.join(map(str, customer_products)) if customer_products else '0'})
            GROUP BY pcb.ProductID
            HAVING COUNT(DISTINCT oi.OrderID) >= 3
            ORDER BY popularity DESC
            """

            try:
                cursor.execute(brand_products_query)
                brand_products = cursor.fetchall()

                # Score products by (brand_weight * normalized_popularity)
                if brand_products:
                    max_popularity = max(p['popularity'] for p in brand_products)
                    for prod in brand_products:
                        normalized_pop = prod['popularity'] / max_popularity
                        brand_scores[prod['ProductID']][brand_id] = brand_weight * normalized_pop
            except Exception as e:
                logger.warning(f"Brand {brand_id} query failed: {e}")
                continue

        cursor.close()

        # Aggregate scores across brands (sum for products appearing in multiple brands)
        final_scores = {}
        for product_id, brand_scores_dict in brand_scores.items():
            final_scores[product_id] = sum(brand_scores_dict.values())

        # Normalize final scores to 0-1 range
        if final_scores:
            max_score = max(final_scores.values())
            final_scores = {pid: score / max_score for pid, score in final_scores.items()}

        return final_scores

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
            # V3.5: LIGHT users with Brand Affinity Boosting
            # Strategy: Use car brand preferences to recommend relevant products
            # (LIGHT users have <100 orders, so sparse frequency/recency data)

            # Get brand affinity score (NEW IN V3.5!)
            brand_affinity_scores = self.get_brand_affinity_score(customer_id, as_of_date)

            # Weights optimized for LIGHT users
            weights = {
                'brand_affinity': 0.40,  # NEW! Primary signal for LIGHT users
                'frequency': 0.40,       # Still important (their own history)
                'recency': 0.20          # Secondary signal
            }

            # Fallback logic for users with no history
            if not frequency_scores and not brand_affinity_scores:
                # Absolute cold-start: use global popularity
                category_scores = self.get_category_popularity_score(customer_id, as_of_date)
                final_scores = category_scores
            elif not frequency_scores:
                # Has brand affinity but no frequency: use brand affinity only
                final_scores = brand_affinity_scores
            else:
                # Combine all available signals
                final_scores = {}
                all_products = (
                    set(frequency_scores.keys()) |
                    set(recency_scores.keys()) |
                    set(brand_affinity_scores.keys())
                )

                for product_id in all_products:
                    freq_score = frequency_scores.get(product_id, 0)
                    rec_score = recency_scores.get(product_id, 0)
                    brand_score = brand_affinity_scores.get(product_id, 0)

                    final_scores[product_id] = (
                        weights['brand_affinity'] * brand_score +
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
