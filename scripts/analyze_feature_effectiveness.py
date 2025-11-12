#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33
from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEffectivenessAnalyzer:
    """
    Analyzes whether V3.3 features (co-purchase, cycle detection) are providing meaningful signals.

    Key Questions:
    1. What % of products get non-zero co-purchase scores?
    2. What % of products have detectable cycles?
    3. What are the score distributions?
    """

    def __init__(self, conn):
        self.conn = conn

    def get_test_customers(self, as_of_date: str, limit: int = 10) -> List[int]:
        """Get a small sample of test customers"""
        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        SELECT DISTINCT TOP {limit} c.ID as CustomerID
        FROM dbo.Client c
        INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE o.Created < '{as_of_date}'
              AND c.IsActive = 1
              AND c.IsBlocked = 0
              AND c.Deleted = 0
        ORDER BY c.ID
        """

        cursor.execute(query)
        customers = [row['CustomerID'] for row in cursor]
        cursor.close()

        return customers

    def analyze_agreement_features(self, agreement_id: int, as_of_date: str) -> Dict:
        """Analyze feature scores for a single agreement"""

        recommender = ImprovedHybridRecommenderV33(conn=self.conn, use_cache=False)

        # Get agreement's product candidates
        cursor = self.conn.cursor(as_dict=True)

        # Get products this agreement has purchased
        query = f"""
        SELECT DISTINCT oi.ProductID
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        """

        cursor.execute(query)
        purchased_products = [row['ProductID'] for row in cursor]
        cursor.close()

        if not purchased_products:
            return None

        # Get feature scores
        co_purchase_scores = recommender.get_co_purchase_scores(
            agreement_id=agreement_id,
            product_ids=purchased_products,
            as_of_date=as_of_date
        )

        cycle_scores = recommender.get_cycle_scores(
            agreement_id=agreement_id,
            product_ids=purchased_products,
            as_of_date=as_of_date
        )

        return {
            'agreement_id': agreement_id,
            'total_products': len(purchased_products),
            'co_purchase_coverage': len([s for s in co_purchase_scores.values() if s > 0]),
            'cycle_coverage': len([s for s in cycle_scores.values() if s > 0]),
            'co_purchase_scores': list(co_purchase_scores.values()),
            'cycle_scores': list(cycle_scores.values()),
            'co_purchase_pct': len([s for s in co_purchase_scores.values() if s > 0]) / len(purchased_products) * 100 if purchased_products else 0,
            'cycle_pct': len([s for s in cycle_scores.values() if s > 0]) / len(purchased_products) * 100 if purchased_products else 0,
        }

    def analyze_customer_agreements(self, customer_id: int, as_of_date: str) -> List[Dict]:
        """Analyze all agreements for a customer"""

        cursor = self.conn.cursor(as_dict=True)

        # Get customer's agreements
        query = f"""
        SELECT DISTINCT ca.ID as AgreementID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
        """

        cursor.execute(query)
        agreements = [row['AgreementID'] for row in cursor]
        cursor.close()

        results = []
        for agreement_id in agreements:
            result = self.analyze_agreement_features(agreement_id, as_of_date)
            if result:
                results.append(result)

        return results

    def run_analysis(self, as_of_date: str = '2024-06-01', num_customers: int = 10):
        """Run comprehensive feature effectiveness analysis"""

        logger.info("="*80)
        logger.info("FEATURE EFFECTIVENESS ANALYSIS")
        logger.info("="*80)
        logger.info(f"As of date: {as_of_date}")
        logger.info(f"Analyzing {num_customers} customers")
        logger.info("="*80)

        test_customers = self.get_test_customers(as_of_date, num_customers)

        all_agreement_results = []

        for i, customer_id in enumerate(test_customers, 1):
            logger.info(f"\nCustomer {i}/{len(test_customers)}: {customer_id}")

            agreement_results = self.analyze_customer_agreements(customer_id, as_of_date)
            all_agreement_results.extend(agreement_results)

            for result in agreement_results:
                logger.info(f"  Agreement {result['agreement_id']}:")
                logger.info(f"    Total products: {result['total_products']}")
                logger.info(f"    Co-purchase coverage: {result['co_purchase_coverage']} ({result['co_purchase_pct']:.1f}%)")
                logger.info(f"    Cycle coverage: {result['cycle_coverage']} ({result['cycle_pct']:.1f}%)")

        # Aggregate statistics
        logger.info("\n" + "="*80)
        logger.info("AGGREGATE STATISTICS")
        logger.info("="*80)

        total_agreements = len(all_agreement_results)
        total_products = sum(r['total_products'] for r in all_agreement_results)
        total_co_purchase_covered = sum(r['co_purchase_coverage'] for r in all_agreement_results)
        total_cycle_covered = sum(r['cycle_coverage'] for r in all_agreement_results)

        logger.info(f"Total agreements analyzed: {total_agreements}")
        logger.info(f"Total products: {total_products}")
        logger.info(f"\nCo-Purchase Feature:")
        logger.info(f"  Products with score > 0: {total_co_purchase_covered}/{total_products} ({total_co_purchase_covered/total_products*100:.1f}%)")
        logger.info(f"\nCycle Detection Feature:")
        logger.info(f"  Products with score > 0: {total_cycle_covered}/{total_products} ({total_cycle_covered/total_products*100:.1f}%)")

        # Score distributions
        all_co_purchase_scores = []
        all_cycle_scores = []

        for result in all_agreement_results:
            all_co_purchase_scores.extend(result['co_purchase_scores'])
            all_cycle_scores.extend(result['cycle_scores'])

        # Histogram
        logger.info("\n" + "="*80)
        logger.info("SCORE DISTRIBUTIONS")
        logger.info("="*80)

        logger.info("\nCo-Purchase Scores:")
        self.print_histogram(all_co_purchase_scores)

        logger.info("\nCycle Scores:")
        self.print_histogram(all_cycle_scores)

        # Interpretation
        logger.info("\n" + "="*80)
        logger.info("INTERPRETATION")
        logger.info("="*80)

        co_purchase_pct = total_co_purchase_covered/total_products*100 if total_products > 0 else 0
        cycle_pct = total_cycle_covered/total_products*100 if total_products > 0 else 0

        if co_purchase_pct < 20:
            logger.info("⚠️  CO-PURCHASE FEATURE IS SPARSE (<20% coverage)")
            logger.info("   → Feature has limited impact on most recommendations")
        elif co_purchase_pct < 50:
            logger.info("✓ Co-purchase feature has moderate coverage (20-50%)")
        else:
            logger.info("✓✓ Co-purchase feature has good coverage (>50%)")

        if cycle_pct < 10:
            logger.info("⚠️  CYCLE DETECTION FEATURE IS VERY SPARSE (<10% coverage)")
            logger.info("   → Feature has minimal impact")
        elif cycle_pct < 30:
            logger.info("✓ Cycle detection has moderate coverage (10-30%)")
        else:
            logger.info("✓✓ Cycle detection has good coverage (>30%)")

        logger.info("="*80)

        return {
            'total_agreements': total_agreements,
            'total_products': total_products,
            'co_purchase_coverage_pct': co_purchase_pct,
            'cycle_coverage_pct': cycle_pct,
            'all_agreement_results': all_agreement_results
        }

    def print_histogram(self, scores: List[float], bins: int = 10):
        """Print a simple histogram of scores"""

        if not scores:
            logger.info("  No scores available")
            return

        # Count zeros separately
        zero_count = sum(1 for s in scores if s == 0)
        non_zero_scores = [s for s in scores if s > 0]

        logger.info(f"  Zeros: {zero_count}/{len(scores)} ({zero_count/len(scores)*100:.1f}%)")

        if non_zero_scores:
            min_score = min(non_zero_scores)
            max_score = max(non_zero_scores)
            avg_score = sum(non_zero_scores) / len(non_zero_scores)

            logger.info(f"  Non-zero range: {min_score:.3f} - {max_score:.3f}")
            logger.info(f"  Non-zero average: {avg_score:.3f}")


def main():
    conn = get_connection()
    try:
        analyzer = FeatureEffectivenessAnalyzer(conn)
        analyzer.run_analysis(as_of_date='2024-06-01', num_customers=10)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
