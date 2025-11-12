#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33
from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PrecisionValidator:
    """
    Validates recommendation precision using temporal holdout:
    - Training period: Orders up to as_of_date
    - Test period: Orders in the 30 days after as_of_date
    """

    def __init__(self, conn):
        self.conn = conn
        self.recommender = ImprovedHybridRecommenderV33(conn=conn, use_cache=False)

    def get_test_customers(self, as_of_date: str, limit: int = 100) -> List[int]:
        """
        Get customers who:
        1. Had orders before as_of_date (so we can make recommendations)
        2. Had orders in the 30 days after as_of_date (so we can validate)
        """
        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        WITH CustomerActivity AS (
            SELECT DISTINCT c.ID as CustomerID
            FROM dbo.Client c
            INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            WHERE o.Created < '{as_of_date}'
                  AND c.IsActive = 1
                  AND c.IsBlocked = 0
                  AND c.Deleted = 0
        ),
        CustomerFuturePurchases AS (
            SELECT DISTINCT c.ID as CustomerID
            FROM dbo.Client c
            INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            WHERE o.Created >= '{as_of_date}'
                  AND o.Created < DATEADD(day, 30, '{as_of_date}')
        )
        SELECT TOP {limit} ca.CustomerID
        FROM CustomerActivity ca
        INNER JOIN CustomerFuturePurchases cf ON ca.CustomerID = cf.CustomerID
        ORDER BY ca.CustomerID
        """

        cursor.execute(query)
        customers = [row['CustomerID'] for row in cursor]
        cursor.close()

        logger.info(f"Found {len(customers)} test customers")
        return customers

    def get_future_purchases_by_agreement(
        self,
        customer_id: int,
        as_of_date: str,
        days_ahead: int = 30
    ) -> Dict[int, set]:
        """
        Get products purchased by each agreement in the future test period.
        Returns: {agreement_id: set(product_ids)}
        """
        cursor = self.conn.cursor(as_dict=True)

        future_date = (datetime.strptime(as_of_date, '%Y-%m-%d') + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        query = f"""
        SELECT DISTINCT
            ca.ID as AgreementID,
            oi.ProductID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created >= '{as_of_date}'
              AND o.Created < '{future_date}'
              AND oi.ProductID IS NOT NULL
        """

        cursor.execute(query)

        purchases_by_agreement = defaultdict(set)
        for row in cursor:
            purchases_by_agreement[row['AgreementID']].add(row['ProductID'])

        cursor.close()
        return dict(purchases_by_agreement)

    def validate_customer(
        self,
        customer_id: int,
        as_of_date: str,
        top_n: int = 20
    ) -> Dict:
        """
        Validate recommendations for a single customer.
        Returns metrics for this customer.
        """
        try:
            # Generate recommendations based on data up to as_of_date
            recommendations = self.recommender.get_recommendations(
                customer_id=customer_id,
                as_of_date=as_of_date,
                top_n=top_n,
                include_discovery=True
            )

            if not recommendations:
                return {
                    'customer_id': customer_id,
                    'status': 'no_recommendations',
                    'precision': 0.0,
                    'hits': 0,
                    'total_recs': 0
                }

            # Get future purchases by agreement
            future_purchases = self.get_future_purchases_by_agreement(
                customer_id=customer_id,
                as_of_date=as_of_date,
                days_ahead=30
            )

            if not future_purchases:
                return {
                    'customer_id': customer_id,
                    'status': 'no_future_purchases',
                    'precision': 0.0,
                    'hits': 0,
                    'total_recs': len(recommendations)
                }

            # Calculate hits: recommendations that match the agreement's future purchases
            hits = 0
            for rec in recommendations:
                product_id = rec['product_id']
                agreement_id = rec.get('agreement_id')

                if agreement_id and agreement_id in future_purchases:
                    if product_id in future_purchases[agreement_id]:
                        hits += 1

            precision = hits / len(recommendations) if recommendations else 0.0

            return {
                'customer_id': customer_id,
                'status': 'success',
                'precision': precision,
                'hits': hits,
                'total_recs': len(recommendations),
                'num_agreements': len(future_purchases),
                'total_future_products': sum(len(prods) for prods in future_purchases.values())
            }

        except Exception as e:
            logger.error(f"Error validating customer {customer_id}: {e}")
            return {
                'customer_id': customer_id,
                'status': 'error',
                'error': str(e),
                'precision': 0.0,
                'hits': 0,
                'total_recs': 0
            }

    def run_validation(
        self,
        as_of_date: str,
        num_customers: int = 100,
        top_n: int = 20
    ) -> Dict:
        """
        Run validation on a sample of customers.
        """
        logger.info("="*80)
        logger.info("RECOMMENDATION PRECISION VALIDATION")
        logger.info("="*80)
        logger.info(f"As of date: {as_of_date}")
        logger.info(f"Test period: 30 days after as_of_date")
        logger.info(f"Number of test customers: {num_customers}")
        logger.info(f"Recommendations per customer: {top_n}")
        logger.info("="*80)

        # Get test customers
        test_customers = self.get_test_customers(as_of_date, limit=num_customers)

        if not test_customers:
            logger.error("No test customers found!")
            return {}

        # Validate each customer
        results = []
        for i, customer_id in enumerate(test_customers, 1):
            result = self.validate_customer(customer_id, as_of_date, top_n)
            results.append(result)

            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(test_customers)} customers validated")

        # Calculate aggregate metrics
        successful_results = [r for r in results if r['status'] == 'success']

        if not successful_results:
            logger.error("No successful validations!")
            return {}

        total_precision = sum(r['precision'] for r in successful_results)
        avg_precision = total_precision / len(successful_results)

        total_hits = sum(r['hits'] for r in successful_results)
        total_recs = sum(r['total_recs'] for r in successful_results)
        overall_precision = total_hits / total_recs if total_recs > 0 else 0.0

        # Status breakdown
        status_counts = defaultdict(int)
        for r in results:
            status_counts[r['status']] += 1

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("VALIDATION RESULTS")
        logger.info("="*80)
        logger.info(f"Total customers tested: {len(test_customers)}")
        logger.info(f"Successful validations: {len(successful_results)}")
        logger.info(f"No recommendations: {status_counts.get('no_recommendations', 0)}")
        logger.info(f"No future purchases: {status_counts.get('no_future_purchases', 0)}")
        logger.info(f"Errors: {status_counts.get('error', 0)}")
        logger.info("-"*80)
        logger.info(f"Average Precision@{top_n}: {avg_precision*100:.2f}%")
        logger.info(f"Overall Precision@{top_n}: {overall_precision*100:.2f}%")
        logger.info(f"Total hits: {total_hits}/{total_recs}")
        logger.info("="*80)

        # Top 10 and bottom 10 customers by precision
        sorted_results = sorted(successful_results, key=lambda x: x['precision'], reverse=True)

        logger.info("\nTop 10 customers by precision:")
        for r in sorted_results[:10]:
            logger.info(
                f"  Customer {r['customer_id']}: {r['precision']*100:.1f}% "
                f"({r['hits']}/{r['total_recs']} hits)"
            )

        logger.info("\nBottom 10 customers by precision:")
        for r in sorted_results[-10:]:
            logger.info(
                f"  Customer {r['customer_id']}: {r['precision']*100:.1f}% "
                f"({r['hits']}/{r['total_recs']} hits)"
            )

        return {
            'as_of_date': as_of_date,
            'num_customers': len(test_customers),
            'successful_validations': len(successful_results),
            'avg_precision': avg_precision,
            'overall_precision': overall_precision,
            'total_hits': total_hits,
            'total_recommendations': total_recs,
            'status_counts': dict(status_counts),
            'results': results
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate recommendation precision using temporal holdout'
    )
    parser.add_argument(
        '--as-of-date',
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        help='Date to use as cutoff (default: 30 days ago)'
    )
    parser.add_argument(
        '--customers',
        type=int,
        default=100,
        help='Number of customers to test (default: 100)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of recommendations per customer (default: 20)'
    )

    args = parser.parse_args()

    conn = get_connection()
    try:
        validator = PrecisionValidator(conn)
        validator.run_validation(
            as_of_date=args.as_of_date,
            num_customers=args.customers,
            top_n=args.top_n
        )
    finally:
        conn.close()


if __name__ == '__main__':
    main()
