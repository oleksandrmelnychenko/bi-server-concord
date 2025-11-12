#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v34 import ImprovedHybridRecommenderV34
from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeightGridSearch:
    """
    Simple grid search for optimal feature weights.
    Tests different weight combinations and measures precision@20.
    """

    def __init__(self, conn, as_of_date='2024-06-01', num_customers=50):
        self.conn = conn
        self.as_of_date = as_of_date
        self.num_customers = num_customers
        self.test_customers = None

    def get_test_customers(self) -> List[int]:
        """Get customers for validation"""
        if self.test_customers is not None:
            return self.test_customers

        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        WITH CustomerActivity AS (
            SELECT DISTINCT c.ID as CustomerID
            FROM dbo.Client c
            INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            WHERE o.Created < '{self.as_of_date}'
                  AND c.IsActive = 1
                  AND c.IsBlocked = 0
                  AND c.Deleted = 0
        ),
        CustomerFuturePurchases AS (
            SELECT DISTINCT c.ID as CustomerID
            FROM dbo.Client c
            INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            WHERE o.Created >= '{self.as_of_date}'
                  AND o.Created < DATEADD(day, 30, '{self.as_of_date}')
        )
        SELECT TOP {self.num_customers} ca.CustomerID
        FROM CustomerActivity ca
        INNER JOIN CustomerFuturePurchases cf ON ca.CustomerID = cf.CustomerID
        ORDER BY ca.CustomerID
        """

        cursor.execute(query)
        self.test_customers = [row['CustomerID'] for row in cursor]
        cursor.close()

        return self.test_customers

    def get_future_purchases_by_agreement(self, customer_id: int) -> Dict[int, set]:
        """Get products purchased by each agreement in future period"""
        cursor = self.conn.cursor(as_dict=True)

        future_date = (datetime.strptime(self.as_of_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

        query = f"""
        SELECT DISTINCT
            ca.ID as AgreementID,
            oi.ProductID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created >= '{self.as_of_date}'
              AND o.Created < '{future_date}'
              AND oi.ProductID IS NOT NULL
        """

        cursor.execute(query)

        purchases_by_agreement = defaultdict(set)
        for row in cursor:
            purchases_by_agreement[row['AgreementID']].add(row['ProductID'])

        cursor.close()
        return dict(purchases_by_agreement)

    def evaluate_weights(self, weights: Dict[str, float]) -> float:
        """
        Evaluate a set of weights on the test set.
        Returns precision@20.
        """
        recommender = ImprovedHybridRecommenderV34(conn=self.conn, use_cache=False, custom_weights=weights)

        test_customers = self.get_test_customers()

        total_hits = 0
        total_recs = 0

        for customer_id in test_customers:
            try:
                # Get recommendations
                recommendations = recommender.get_recommendations(
                    customer_id=customer_id,
                    as_of_date=self.as_of_date,
                    top_n=20,
                    include_discovery=False
                )

                if not recommendations:
                    continue

                # Get future purchases
                future_purchases = self.get_future_purchases_by_agreement(customer_id)

                if not future_purchases:
                    continue

                # Calculate hits
                for rec in recommendations:
                    product_id = rec['product_id']
                    agreement_id = rec.get('agreement_id')

                    if agreement_id and agreement_id in future_purchases:
                        if product_id in future_purchases[agreement_id]:
                            total_hits += 1

                total_recs += len(recommendations)

            except Exception as e:
                logger.error(f"Error evaluating customer {customer_id}: {e}")
                continue

        precision = total_hits / total_recs if total_recs > 0 else 0.0
        return precision

    def run_grid_search(self):
        """
        Perform grid search over weight combinations.
        """
        logger.info("="*80)
        logger.info("GRID SEARCH - FEATURE WEIGHT OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"As of date: {self.as_of_date}")
        logger.info(f"Test customers: {self.num_customers}")
        logger.info("="*80)

        # Define search grid - weights must sum to 1.0
        weight_combinations = [
            # Baseline V3.3
            ('V3.3 Baseline', {'frequency': 0.35, 'recency': 0.20, 'co_purchase': 0.30, 'cycle': 0.15}),

            # Boost co_purchase (B2B hypothesis)
            ('High Co-Purchase 1', {'frequency': 0.25, 'recency': 0.15, 'co_purchase': 0.45, 'cycle': 0.15}),
            ('High Co-Purchase 2', {'frequency': 0.20, 'recency': 0.15, 'co_purchase': 0.50, 'cycle': 0.15}),
            ('High Co-Purchase 3', {'frequency': 0.20, 'recency': 0.10, 'co_purchase': 0.55, 'cycle': 0.15}),

            # Boost cycle detection
            ('High Cycle 1', {'frequency': 0.25, 'recency': 0.15, 'co_purchase': 0.35, 'cycle': 0.25}),
            ('High Cycle 2', {'frequency': 0.20, 'recency': 0.15, 'co_purchase': 0.35, 'cycle': 0.30}),

            # Boost traditional signals
            ('High Frequency', {'frequency': 0.50, 'recency': 0.25, 'co_purchase': 0.15, 'cycle': 0.10}),
            ('High Recency', {'frequency': 0.25, 'recency': 0.45, 'co_purchase': 0.20, 'cycle': 0.10}),

            # Balanced
            ('Equal Weights', {'frequency': 0.25, 'recency': 0.25, 'co_purchase': 0.25, 'cycle': 0.25}),

            # Hybrid approaches
            ('Hybrid 1', {'frequency': 0.30, 'recency': 0.20, 'co_purchase': 0.35, 'cycle': 0.15}),
            ('Hybrid 2', {'frequency': 0.30, 'recency': 0.15, 'co_purchase': 0.40, 'cycle': 0.15}),
            ('Hybrid 3', {'frequency': 0.30, 'recency': 0.15, 'co_purchase': 0.35, 'cycle': 0.20}),
        ]

        logger.info(f"Testing {len(weight_combinations)} weight combinations...")
        logger.info("")

        results = []

        for i, (name, weights) in enumerate(weight_combinations, 1):
            logger.info(f"Trial {i}/{len(weight_combinations)}: {name}")
            logger.info(f"  Weights: {weights}")

            precision = self.evaluate_weights(weights)

            results.append({
                'name': name,
                'weights': weights,
                'precision': precision,
                'trial': i
            })

            logger.info(f"  → Precision@20: {precision*100:.2f}%")
            logger.info("")

        # Sort by precision
        results.sort(key=lambda x: x['precision'], reverse=True)

        logger.info("="*80)
        logger.info("RESULTS - RANKED BY PRECISION")
        logger.info("="*80)

        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['name']}: {result['precision']*100:.2f}%")
            logger.info(f"   Weights: {result['weights']}")
            if i == 1:
                logger.info("   ⭐ BEST")
            logger.info("")

        # Show improvement over baseline
        baseline_precision = [r['precision'] for r in results if r['name'] == 'V3.3 Baseline'][0]
        best_precision = results[0]['precision']
        improvement = ((best_precision - baseline_precision) / baseline_precision * 100) if baseline_precision > 0 else 0

        logger.info("="*80)
        logger.info(f"Baseline (V3.3): {baseline_precision*100:.2f}%")
        logger.info(f"Best: {best_precision*100:.2f}%")
        logger.info(f"Improvement: +{improvement:.1f}% relative")
        logger.info("="*80)

        return results


def main():
    conn = get_connection()
    try:
        searcher = WeightGridSearch(conn, as_of_date='2024-06-01', num_customers=50)
        results = searcher.run_grid_search()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
