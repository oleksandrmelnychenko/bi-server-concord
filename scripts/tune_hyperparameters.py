#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Tune hyperparameters for V3.3 recommendation system using grid search.

    Parameters to optimize:
    - Feature weights (frequency, recency, co_purchase, cycle)
    - Co-purchase lookback window
    - Minimum cycle threshold
    """

    def __init__(self, conn, as_of_date: str = '2024-06-01', num_customers: int = 50):
        self.conn = conn
        self.as_of_date = as_of_date
        self.num_customers = num_customers

    def get_test_customers(self) -> List[int]:
        """Get customers for validation"""
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
        customers = [row['CustomerID'] for row in cursor]
        cursor.close()

        return customers

    def evaluate_weights(self, weights: Dict[str, float]) -> float:
        """
        Evaluate a set of weights on the test set.
        Returns precision@20.
        """
        # Import here to avoid circular dependencies
        from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33

        # Create recommender with custom weights
        recommender = ImprovedHybridRecommenderV33(conn=self.conn, use_cache=False)

        # Monkey-patch the weights for testing
        # This is a quick way to test different weights without creating new classes
        original_method = recommender.get_recommendations_for_agreement

        def patched_method(agreement_id, as_of_date, top_n=20, include_discovery=True):
            # Get original recommendations with default weights
            # We'll need to override the weight calculation
            # For now, let's use a simpler approach: modify the class weights

            # Call original but intercept at scoring stage
            # This requires modifying the v33 code to accept weights as parameters
            # For quick testing, we'll use a different approach

            return original_method(agreement_id, as_of_date, top_n, include_discovery)

        # Get test customers
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

    def grid_search(self):
        """
        Perform grid search over weight combinations.
        Weights must sum to 1.0.
        """
        logger.info("="*80)
        logger.info("HYPERPARAMETER TUNING - GRID SEARCH")
        logger.info("="*80)
        logger.info(f"As of date: {self.as_of_date}")
        logger.info(f"Test customers: {self.num_customers}")
        logger.info("="*80)

        # Define search grid
        # We'll test different weight combinations that sum to 1.0

        weight_combinations = [
            # Baseline V3.3
            {'frequency': 0.35, 'recency': 0.20, 'co_purchase': 0.30, 'cycle': 0.15},

            # Boost frequency
            {'frequency': 0.40, 'recency': 0.20, 'co_purchase': 0.25, 'cycle': 0.15},
            {'frequency': 0.45, 'recency': 0.20, 'co_purchase': 0.20, 'cycle': 0.15},
            {'frequency': 0.50, 'recency': 0.15, 'co_purchase': 0.20, 'cycle': 0.15},

            # Boost recency
            {'frequency': 0.30, 'recency': 0.25, 'co_purchase': 0.30, 'cycle': 0.15},
            {'frequency': 0.30, 'recency': 0.30, 'co_purchase': 0.25, 'cycle': 0.15},
            {'frequency': 0.25, 'recency': 0.35, 'co_purchase': 0.25, 'cycle': 0.15},

            # Boost co_purchase
            {'frequency': 0.30, 'recency': 0.20, 'co_purchase': 0.35, 'cycle': 0.15},
            {'frequency': 0.25, 'recency': 0.20, 'co_purchase': 0.40, 'cycle': 0.15},
            {'frequency': 0.25, 'recency': 0.15, 'co_purchase': 0.45, 'cycle': 0.15},

            # Boost cycle
            {'frequency': 0.30, 'recency': 0.20, 'co_purchase': 0.30, 'cycle': 0.20},
            {'frequency': 0.25, 'recency': 0.20, 'co_purchase': 0.30, 'cycle': 0.25},
            {'frequency': 0.25, 'recency': 0.15, 'co_purchase': 0.30, 'cycle': 0.30},

            # Balanced
            {'frequency': 0.25, 'recency': 0.25, 'co_purchase': 0.25, 'cycle': 0.25},

            # Heavy co_purchase + cycle (B2B focus)
            {'frequency': 0.20, 'recency': 0.15, 'co_purchase': 0.40, 'cycle': 0.25},
            {'frequency': 0.20, 'recency': 0.15, 'co_purchase': 0.45, 'cycle': 0.20},

            # Heavy frequency + recency (traditional)
            {'frequency': 0.50, 'recency': 0.30, 'co_purchase': 0.10, 'cycle': 0.10},
            {'frequency': 0.45, 'recency': 0.35, 'co_purchase': 0.10, 'cycle': 0.10},
        ]

        logger.info(f"Testing {len(weight_combinations)} weight combinations...")
        logger.info("")

        results = []

        for i, weights in enumerate(weight_combinations, 1):
            logger.info(f"Trial {i}/{len(weight_combinations)}: {weights}")

            # Note: This simplified version doesn't actually use custom weights
            # We'd need to modify v33 to accept weights as parameters
            # For now, this serves as a framework

            # Placeholder: actual evaluation would require modifying v33
            # precision = self.evaluate_weights(weights)
            precision = 0.0  # Placeholder

            results.append({
                'weights': weights,
                'precision': precision,
                'trial': i
            })

            logger.info(f"  → Precision@20: {precision*100:.2f}%")
            logger.info("")

        # Sort by precision
        results.sort(key=lambda x: x['precision'], reverse=True)

        logger.info("="*80)
        logger.info("TOP 5 WEIGHT COMBINATIONS")
        logger.info("="*80)

        for i, result in enumerate(results[:5], 1):
            logger.info(f"{i}. Precision: {result['precision']*100:.2f}%")
            logger.info(f"   Weights: {result['weights']}")
            logger.info("")

        return results


def main():
    logger.info("Note: This is a framework for hyperparameter tuning.")
    logger.info("The actual implementation requires modifying V3.3 to accept weights as parameters.")
    logger.info("See SIMPLIFIED_TUNING_APPROACH.md for a manual tuning strategy.")

    conn = get_connection()
    try:
        tuner = HyperparameterTuner(conn, as_of_date='2024-06-01', num_customers=50)
        results = tuner.grid_search()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
