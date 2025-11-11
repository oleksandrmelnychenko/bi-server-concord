#!/usr/bin/env python3

import os
import json
import pymssql
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
    'as_dict': True
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedHybridRecommender:

    def __init__(self, conn=None):
        if conn:
            self.conn = conn
            self.owns_connection = False
            logger.debug("Using provided database connection")
        else:
            self.conn = None
            self.owns_connection = True
            self._connect()

    def _connect(self):
        try:
            self.conn = pymssql.connect(**DB_CONFIG)
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def classify_customer(self, customer_id: int, as_of_date: str) -> Tuple[str, str]:

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

        if orders_before >= 500:
            segment = "HEAVY"
            subsegment = None
        elif orders_before >= 100:
            segment = "REGULAR"

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

        scores = {pid: count / max_count for pid, count in scores.items()}

        cursor.close()
        return scores

    def get_recency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
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

            score = 2.718 ** (-days_ago / 90)
            scores[row['ProductID']] = score

        cursor.close()
        return scores

    def get_category_popularity_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:

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

        scores = {pid: count / max_count for pid, count in scores.items()}

        cursor.close()
        return scores

    def get_recommendations(self, customer_id: int, as_of_date: str, top_n: int = 50) -> List[Dict]:
        segment, subsegment = self.classify_customer(customer_id, as_of_date)

        logger.debug(f"Customer {customer_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

        frequency_scores = self.get_frequency_score(customer_id, as_of_date)
        recency_scores = self.get_recency_score(customer_id, as_of_date)

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
            else:
                weights = {
                    'frequency': 0.25,
                    'recency': 0.50,
                    'other': 0.25
                }
        else:

            weights = {
                'frequency': 0.70,  # INCREASED - even with few orders, their own history is best
                'recency': 0.30     # Recent purchases are strongest signal
            }

            if not frequency_scores:
                category_scores = self.get_category_popularity_score(customer_id, as_of_date)
                final_scores = category_scores
            else:

                final_scores = {}
                all_products = set(frequency_scores.keys()) | set(recency_scores.keys())

                for product_id in all_products:
                    freq_score = frequency_scores.get(product_id, 0)
                    rec_score = recency_scores.get(product_id, 0)

                    final_scores[product_id] = (
                        weights['frequency'] * freq_score +
                        weights['recency'] * rec_score
                    )

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

        final_scores = {}
        all_products = set(frequency_scores.keys()) | set(recency_scores.keys())

        for product_id in all_products:
            freq_score = frequency_scores.get(product_id, 0)
            rec_score = recency_scores.get(product_id, 0)

            final_scores[product_id] = (
                weights['frequency'] * freq_score +
                weights['recency'] * rec_score
            )

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
        if self.conn:
            self.conn.close()
            if self.owns_connection:
                logger.info("✓ Database connection closed")
            else:
                logger.debug("✓ Connection returned to pool")

def main():
    logger.info("=" * 80)
    logger.info("IMPROVED HYBRID RECOMMENDER V2 - TEST")
    logger.info("=" * 80)

    recommender = ImprovedHybridRecommender()

    try:

        test_customers = [
            410190,
            411726,
            411317,
            410839
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
