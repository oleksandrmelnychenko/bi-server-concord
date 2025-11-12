#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33
from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationDiagnostic:
    """
    Deep diagnostic analysis to understand:
    1. WHY features are sparse (root cause analysis)
    2. Manual inspection of recommendations vs actual purchases
    3. Comparison with simpler baselines
    """

    def __init__(self, conn):
        self.conn = conn

    def analyze_purchase_patterns(self, agreement_id: int, as_of_date: str) -> Dict:
        """
        Analyze why co-purchase might be sparse for an agreement.

        Questions:
        - How many orders does this agreement have?
        - How many products per order (avg)?
        - Are products typically bought alone or in groups?
        """
        cursor = self.conn.cursor(as_dict=True)

        # Get order statistics
        query = f"""
        SELECT
            o.ID as OrderID,
            COUNT(DISTINCT oi.ProductID) as ProductCount
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY o.ID
        """

        cursor.execute(query)
        orders = list(cursor)
        cursor.close()

        if not orders:
            return None

        product_counts = [o['ProductCount'] for o in orders]

        return {
            'agreement_id': agreement_id,
            'total_orders': len(orders),
            'avg_products_per_order': sum(product_counts) / len(product_counts),
            'single_product_orders': len([c for c in product_counts if c == 1]),
            'multi_product_orders': len([c for c in product_counts if c > 1]),
            'max_products_in_order': max(product_counts),
            'single_product_pct': len([c for c in product_counts if c == 1]) / len(product_counts) * 100
        }

    def manual_inspection(self, customer_id: int, as_of_date: str, top_n: int = 10) -> Dict:
        """
        Manual inspection: What did V3.3 recommend vs what was actually purchased?
        """
        recommender = ImprovedHybridRecommenderV33(conn=self.conn, use_cache=False)

        # Get recommendations
        recommendations = recommender.get_recommendations(
            customer_id=customer_id,
            as_of_date=as_of_date,
            top_n=top_n,
            include_discovery=False
        )

        if not recommendations:
            return None

        # Get actual purchases in next 30 days
        cursor = self.conn.cursor(as_dict=True)

        future_date = (datetime.strptime(as_of_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

        query = f"""
        SELECT DISTINCT
            ca.ID as AgreementID,
            oi.ProductID,
            p.Name as ProductName,
            oi.PricePerItem,
            o.Created
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        LEFT JOIN dbo.Product p ON oi.ProductID = p.ID
        WHERE ca.ClientID = {customer_id}
              AND o.Created >= '{as_of_date}'
              AND o.Created < '{future_date}'
              AND oi.ProductID IS NOT NULL
        ORDER BY o.Created
        """

        cursor.execute(query)
        actual_purchases = list(cursor)
        cursor.close()

        # Map recommendations by agreement
        recs_by_agreement = defaultdict(list)
        for rec in recommendations:
            agreement_id = rec.get('agreement_id')
            if agreement_id:
                recs_by_agreement[agreement_id].append(rec)

        # Compare recommendations vs actual
        results = {
            'customer_id': customer_id,
            'total_recommendations': len(recommendations),
            'total_actual_purchases': len(actual_purchases),
            'agreements': []
        }

        # Group actual purchases by agreement
        actual_by_agreement = defaultdict(list)
        for purchase in actual_purchases:
            actual_by_agreement[purchase['AgreementID']].append(purchase)

        # Analyze each agreement
        for agreement_id in set(list(recs_by_agreement.keys()) + list(actual_by_agreement.keys())):
            recs = recs_by_agreement.get(agreement_id, [])
            actual = actual_by_agreement.get(agreement_id, [])

            rec_product_ids = set(r['product_id'] for r in recs)
            actual_product_ids = set(a['ProductID'] for a in actual)

            hits = rec_product_ids.intersection(actual_product_ids)

            agreement_result = {
                'agreement_id': agreement_id,
                'num_recommendations': len(recs),
                'num_actual_purchases': len(actual),
                'hits': len(hits),
                'precision': len(hits) / len(recs) if recs else 0,
                'recommended_products': [
                    {
                        'product_id': r['product_id'],
                        'product_name': r.get('product_name', 'N/A'),
                        'score': r.get('score', 0),
                        'purchased': r['product_id'] in actual_product_ids
                    }
                    for r in recs[:5]  # Top 5 only
                ],
                'missed_products': [
                    {
                        'product_id': a['ProductID'],
                        'product_name': a.get('ProductName', 'N/A'),
                        'price': float(a['PricePerItem']) if a.get('PricePerItem') else None,
                        'order_date': str(a['Created'])
                    }
                    for a in actual if a['ProductID'] not in rec_product_ids
                ][:5]  # Top 5 misses only
            }

            results['agreements'].append(agreement_result)

        return results

    def baseline_comparison(self, customer_id: int, as_of_date: str) -> Dict:
        """
        Compare V3.3 against simpler baselines:
        - Frequency-only (most purchased products)
        - Recency-only (most recently purchased)
        - Random baseline
        """
        cursor = self.conn.cursor(as_dict=True)

        # Get customer agreements
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

        if not agreements:
            return None

        results = {
            'customer_id': customer_id,
            'baselines': []
        }

        # For each agreement, compute frequency and recency baselines
        for agreement_id in agreements:
            # Frequency baseline: Top 20 most purchased products
            freq_baseline = self._get_frequency_baseline(agreement_id, as_of_date, top_n=20)

            # Recency baseline: Top 20 most recently purchased products
            recency_baseline = self._get_recency_baseline(agreement_id, as_of_date, top_n=20)

            results['baselines'].append({
                'agreement_id': agreement_id,
                'frequency_baseline': freq_baseline,
                'recency_baseline': recency_baseline
            })

        return results

    def _get_frequency_baseline(self, agreement_id: int, as_of_date: str, top_n: int = 20) -> List[int]:
        """Get top N most frequently purchased products"""
        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        SELECT TOP {top_n}
            oi.ProductID,
            COUNT(DISTINCT o.ID) as OrderCount
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        ORDER BY OrderCount DESC, oi.ProductID DESC
        """

        cursor.execute(query)
        products = [row['ProductID'] for row in cursor]
        cursor.close()

        return products

    def _get_recency_baseline(self, agreement_id: int, as_of_date: str, top_n: int = 20) -> List[int]:
        """Get top N most recently purchased products"""
        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        SELECT TOP {top_n}
            oi.ProductID,
            MAX(o.Created) as LastPurchase
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        ORDER BY LastPurchase DESC
        """

        cursor.execute(query)
        products = [row['ProductID'] for row in cursor]
        cursor.close()

        return products

    def run_diagnostic(self, num_customers: int = 5):
        """Run comprehensive diagnostic on sample customers"""
        logger.info("="*80)
        logger.info("DEEP DIAGNOSTIC ANALYSIS")
        logger.info("="*80)

        as_of_date = '2024-06-01'

        # Get test customers
        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        SELECT DISTINCT TOP {num_customers} c.ID as CustomerID
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

        # Part 1: Purchase pattern analysis
        logger.info("\n" + "="*80)
        logger.info("PART 1: WHY IS CO-PURCHASE SPARSE?")
        logger.info("="*80)

        all_patterns = []
        for customer_id in customers:
            cursor = self.conn.cursor(as_dict=True)

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

            for agreement_id in agreements:
                pattern = self.analyze_purchase_patterns(agreement_id, as_of_date)
                if pattern:
                    all_patterns.append(pattern)

        # Aggregate statistics
        if all_patterns:
            avg_products_per_order = sum(p['avg_products_per_order'] for p in all_patterns) / len(all_patterns)
            avg_single_pct = sum(p['single_product_pct'] for p in all_patterns) / len(all_patterns)

            logger.info(f"\nAnalyzed {len(all_patterns)} agreements:")
            logger.info(f"  Average products per order: {avg_products_per_order:.2f}")
            logger.info(f"  Single-product orders: {avg_single_pct:.1f}%")

            if avg_products_per_order < 2:
                logger.info("\n⚠️  ROOT CAUSE: Orders typically contain 1-2 products")
                logger.info("   → Co-purchase requires multiple products in same order")
                logger.info("   → Low products-per-order = sparse co-purchase signal")
            else:
                logger.info("\n✓ Orders contain multiple products on average")
                logger.info("  → Co-purchase sparsity likely due to other factors")

        # Part 2: Manual inspection
        logger.info("\n" + "="*80)
        logger.info("PART 2: MANUAL INSPECTION - RECOMMENDATIONS VS REALITY")
        logger.info("="*80)

        for i, customer_id in enumerate(customers[:3], 1):  # Top 3 only for detailed inspection
            logger.info(f"\nCustomer {i}: {customer_id}")

            inspection = self.manual_inspection(customer_id, as_of_date, top_n=10)

            if inspection:
                logger.info(f"  Total recommendations: {inspection['total_recommendations']}")
                logger.info(f"  Total actual purchases: {inspection['total_actual_purchases']}")

                for agreement in inspection['agreements'][:2]:  # Top 2 agreements
                    logger.info(f"\n  Agreement {agreement['agreement_id']}:")
                    logger.info(f"    Precision: {agreement['precision']*100:.1f}%")
                    logger.info(f"    Hits: {agreement['hits']}/{agreement['num_recommendations']}")

                    if agreement['recommended_products']:
                        logger.info(f"\n    Top Recommendations:")
                        for rec in agreement['recommended_products'][:3]:
                            status = "✓ HIT" if rec['purchased'] else "✗ MISS"
                            logger.info(f"      {status} - Product {rec['product_id']} (score: {rec['score']:.3f})")

                    if agreement['missed_products']:
                        logger.info(f"\n    Missed Opportunities:")
                        for miss in agreement['missed_products'][:3]:
                            logger.info(f"      • Product {miss['product_id']} - {miss['product_name']}")

        logger.info("\n" + "="*80)


def main():
    conn = get_connection()
    try:
        diagnostic = RecommendationDiagnostic(conn)
        diagnostic.run_diagnostic(num_customers=5)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
