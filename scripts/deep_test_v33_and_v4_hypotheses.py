#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33
from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepTestingFramework:
    """
    Comprehensive testing framework to:
    1. Analyze V3.3 recommendation quality in depth
    2. Understand what products are being missed
    3. Test hypotheses for V4 features (brand, analogue, category)
    4. Quantify expected impact of each feature
    """

    def __init__(self):
        self.conn = get_connection()
        self.recommender = ImprovedHybridRecommenderV33(conn=self.conn, use_cache=False)

    def test_recommendation_quality_deep(self, agreement_id: int, as_of_date: str, test_days: int = 30):
        """
        Deep analysis of recommendation quality for a single agreement.

        Returns detailed breakdown of:
        - What was recommended
        - What was actually purchased
        - What was missed and why
        - Brand patterns
        - Analogue opportunities
        - Category patterns
        """
        logger.info(f"="*80)
        logger.info(f"DEEP QUALITY ANALYSIS: Agreement {agreement_id}")
        logger.info(f"="*80)

        # Get V3.3 recommendations
        recommendations = self.recommender.get_recommendations_for_agreement(
            agreement_id=agreement_id,
            as_of_date=as_of_date,
            top_n=20
        )

        recommended_products = [r['product_id'] for r in recommendations]

        # Get actual purchases in test period
        test_start = datetime.strptime(as_of_date, '%Y-%m-%d')
        test_end = test_start + timedelta(days=test_days)

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(f"""
        SELECT DISTINCT oi.ProductID, p.Name, p.VendorCode
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        LEFT JOIN dbo.Product p ON oi.ProductID = p.ID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created >= '{test_start.strftime('%Y-%m-%d')}'
              AND o.Created < '{test_end.strftime('%Y-%m-%d')}'
        """)
        actual_purchases = list(cursor)
        actual_product_ids = [p['ProductID'] for p in actual_purchases]

        # Calculate hits and misses
        hits = set(recommended_products) & set(actual_product_ids)
        misses = set(actual_product_ids) - set(recommended_products)

        precision = len(hits) / len(recommended_products) if recommended_products else 0
        recall = len(hits) / len(actual_product_ids) if actual_product_ids else 0

        logger.info(f"\nBaseline Performance:")
        logger.info(f"  Recommended: {len(recommended_products)} products")
        logger.info(f"  Actually purchased: {len(actual_product_ids)} products")
        logger.info(f"  Hits: {len(hits)} products")
        logger.info(f"  Misses: {len(misses)} products")
        logger.info(f"  Precision@20: {precision*100:.1f}%")
        logger.info(f"  Recall@20: {recall*100:.1f}%")

        # Analyze missed products
        logger.info(f"\n" + "="*80)
        logger.info(f"ANALYZING MISSED PRODUCTS ({len(misses)} products)")
        logger.info(f"="*80)

        analysis_results = {
            'precision': precision,
            'recall': recall,
            'hits': len(hits),
            'misses': len(misses),
            'brand_opportunities': 0,
            'analogue_opportunities': 0,
            'category_opportunities': 0,
        }

        if misses:
            # Test Brand Hypothesis
            brand_analysis = self._test_brand_hypothesis(agreement_id, list(misses), as_of_date)
            analysis_results['brand_opportunities'] = brand_analysis['recoverable']

            # Test Analogue Hypothesis
            analogue_analysis = self._test_analogue_hypothesis(agreement_id, list(misses), as_of_date)
            analysis_results['analogue_opportunities'] = analogue_analysis['recoverable']

            # Test Category Hypothesis
            category_analysis = self._test_category_hypothesis(agreement_id, list(misses), as_of_date)
            analysis_results['category_opportunities'] = category_analysis['recoverable']

        cursor.close()
        return analysis_results

    def _test_brand_hypothesis(self, agreement_id: int, missed_products: List[int], as_of_date: str):
        """
        Test: Would brand-based scoring have helped?

        Logic:
        - Get brands customer has historically purchased
        - Check if missed products belong to those brands
        - If yes, brand-based scoring would have boosted them
        """
        logger.info(f"\n--- Testing Brand Hypothesis ---")

        cursor = self.conn.cursor(as_dict=True)

        # Get customer's historical brand preferences
        cursor.execute(f"""
        SELECT cb.Name as Brand, COUNT(*) as PurchaseCount
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        INNER JOIN dbo.ProductCarBrand pcb ON oi.ProductID = pcb.ProductID
        INNER JOIN dbo.CarBrand cb ON pcb.CarBrandID = cb.ID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND pcb.Deleted = 0
        GROUP BY cb.Name
        ORDER BY COUNT(*) DESC
        """)
        historical_brands = {row['Brand']: row['PurchaseCount'] for row in cursor}

        if not historical_brands:
            logger.info("  No brand data available for this customer")
            cursor.close()
            return {'recoverable': 0, 'total': len(missed_products)}

        logger.info(f"  Customer's top brands: {list(historical_brands.keys())[:5]}")

        # Check how many missed products belong to customer's preferred brands
        missed_str = ','.join(map(str, missed_products))
        cursor.execute(f"""
        SELECT
            pcb.ProductID,
            cb.Name as Brand,
            p.Name as ProductName
        FROM dbo.ProductCarBrand pcb
        INNER JOIN dbo.CarBrand cb ON pcb.CarBrandID = cb.ID
        LEFT JOIN dbo.Product p ON pcb.ProductID = p.ID
        WHERE pcb.ProductID IN ({missed_str})
              AND pcb.Deleted = 0
              AND cb.Name IN ({','.join([f"'{b}'" for b in historical_brands.keys()])})
        """)
        recoverable = list(cursor)

        recovery_rate = len(recoverable) / len(missed_products) * 100 if missed_products else 0

        logger.info(f"  Missed products from customer's preferred brands: {len(recoverable)}/{len(missed_products)} ({recovery_rate:.1f}%)")

        if recoverable:
            logger.info(f"  Sample missed products that match customer's brands:")
            for item in recoverable[:3]:
                logger.info(f"    - {item['Brand']:15s}: {item['ProductName'][:60]}")

        cursor.close()
        return {'recoverable': len(recoverable), 'total': len(missed_products), 'recovery_rate': recovery_rate}

    def _test_analogue_hypothesis(self, agreement_id: int, missed_products: List[int], as_of_date: str):
        """
        Test: Would analogue-based scoring have helped?

        Logic:
        - Get products customer has purchased
        - Check if missed products are analogues of purchased products
        - If yes, analogue-based scoring would have recommended them
        """
        logger.info(f"\n--- Testing Analogue Hypothesis ---")

        cursor = self.conn.cursor(as_dict=True)

        # Get customer's historical purchases
        cursor.execute(f"""
        SELECT DISTINCT oi.ProductID
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
        """)
        purchased_products = [row['ProductID'] for row in cursor]

        if not purchased_products:
            logger.info("  No purchase history available")
            cursor.close()
            return {'recoverable': 0, 'total': len(missed_products)}

        # Check if missed products are analogues of previously purchased products
        missed_str = ','.join(map(str, missed_products))
        purchased_str = ','.join(map(str, purchased_products))

        cursor.execute(f"""
        SELECT
            pa.AnalogueProductID as MissedProduct,
            pa.BaseProductID as PurchasedProduct,
            p1.Name as MissedName,
            p2.Name as PurchasedName
        FROM dbo.ProductAnalogue pa
        LEFT JOIN dbo.Product p1 ON pa.AnalogueProductID = p1.ID
        LEFT JOIN dbo.Product p2 ON pa.BaseProductID = p2.ID
        WHERE pa.AnalogueProductID IN ({missed_str})
              AND pa.BaseProductID IN ({purchased_str})
              AND pa.Deleted = 0
        """)
        recoverable = list(cursor)

        recovery_rate = len(recoverable) / len(missed_products) * 100 if missed_products else 0

        logger.info(f"  Missed products that are analogues of purchased products: {len(recoverable)}/{len(missed_products)} ({recovery_rate:.1f}%)")

        if recoverable:
            logger.info(f"  Sample analogue opportunities:")
            for item in recoverable[:3]:
                logger.info(f"    - Purchased: {item['PurchasedName'][:40]}")
                logger.info(f"      Missed analogue: {item['MissedName'][:40]}")

        cursor.close()
        return {'recoverable': len(recoverable), 'total': len(missed_products), 'recovery_rate': recovery_rate}

    def _test_category_hypothesis(self, agreement_id: int, missed_products: List[int], as_of_date: str):
        """
        Test: Would category-based scoring have helped?

        Logic:
        - Extract categories from product names using keywords
        - Check if missed products belong to categories customer frequently buys
        - If yes, category-based scoring would have boosted them
        """
        logger.info(f"\n--- Testing Category Hypothesis ---")

        # Auto parts category keywords
        keywords = {
            'фільтр': 'Filters', 'фильтр': 'Filters',
            'масло': 'Oils', 'олія': 'Oils',
            'тормоз': 'Brakes', 'гальм': 'Brakes',
            'диск': 'Disks',
            'колодк': 'Brake Pads',
            'амортизатор': 'Suspension', 'амортизац': 'Suspension',
            'подшипник': 'Bearings', 'підшипник': 'Bearings',
            'шланг': 'Hoses',
            'насос': 'Pumps',
            'клапан': 'Valves',
            'прокладк': 'Gaskets',
            'компрессор': 'Compressors', 'компресор': 'Compressors',
            'радиатор': 'Radiators', 'радіатор': 'Radiators',
            'ремень': 'Belts', 'ремінь': 'Belts',
            'свеч': 'Spark Plugs', 'свіч': 'Spark Plugs',
        }

        cursor = self.conn.cursor(as_dict=True)

        # Get customer's historical purchases with names
        cursor.execute(f"""
        SELECT DISTINCT oi.ProductID, p.Name
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        LEFT JOIN dbo.Product p ON oi.ProductID = p.ID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND p.Name IS NOT NULL
        """)
        purchased = list(cursor)

        # Categorize purchased products
        purchased_categories = Counter()
        for item in purchased:
            name_lower = item['Name'].lower() if item['Name'] else ''
            for keyword, category in keywords.items():
                if keyword in name_lower:
                    purchased_categories[category] += 1
                    break

        if not purchased_categories:
            logger.info("  Unable to categorize purchased products")
            cursor.close()
            return {'recoverable': 0, 'total': len(missed_products)}

        logger.info(f"  Customer's top categories: {purchased_categories.most_common(5)}")

        # Get missed products with names
        missed_str = ','.join(map(str, missed_products))
        cursor.execute(f"""
        SELECT ID as ProductID, Name
        FROM dbo.Product
        WHERE ID IN ({missed_str})
              AND Name IS NOT NULL
        """)
        missed = list(cursor)

        # Check how many missed products belong to customer's preferred categories
        recoverable = 0
        sample_recoverable = []

        for item in missed:
            name_lower = item['Name'].lower() if item['Name'] else ''
            for keyword, category in keywords.items():
                if keyword in name_lower:
                    if category in purchased_categories:
                        recoverable += 1
                        if len(sample_recoverable) < 3:
                            sample_recoverable.append((category, item['Name']))
                    break

        recovery_rate = recoverable / len(missed_products) * 100 if missed_products else 0

        logger.info(f"  Missed products from customer's preferred categories: {recoverable}/{len(missed_products)} ({recovery_rate:.1f}%)")

        if sample_recoverable:
            logger.info(f"  Sample category opportunities:")
            for category, name in sample_recoverable:
                logger.info(f"    - {category:15s}: {name[:60]}")

        cursor.close()
        return {'recoverable': recoverable, 'total': len(missed_products), 'recovery_rate': recovery_rate}

    def run_comprehensive_test(self, num_customers: int = 10, as_of_date: str = '2024-06-01'):
        """
        Run comprehensive test across multiple customers to quantify V4 potential.
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE DEEP TESTING FRAMEWORK")
        logger.info("="*80)
        logger.info(f"Testing {num_customers} customers")
        logger.info(f"As-of date: {as_of_date}")
        logger.info(f"Test window: 30 days after as-of date")
        logger.info("="*80)

        # Get sample customers (agreements)
        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(f"""
        SELECT TOP {num_customers}
            ca.ID as AgreementID,
            ca.ClientID,
            COUNT(DISTINCT o.ID) as OrderCount
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE o.Created >= DATEADD(month, -12, '{as_of_date}')
              AND o.Created < '{as_of_date}'
        GROUP BY ca.ID, ca.ClientID
        HAVING COUNT(DISTINCT o.ID) >= 50
        ORDER BY COUNT(DISTINCT o.ID) DESC
        """)
        customers = list(cursor)
        cursor.close()

        logger.info(f"\nTesting {len(customers)} agreements...")

        all_results = []
        total_brand_opportunities = 0
        total_analogue_opportunities = 0
        total_category_opportunities = 0
        total_misses = 0
        total_hits = 0
        total_actual = 0

        for idx, customer in enumerate(customers, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Test {idx}/{len(customers)}: Agreement {customer['AgreementID']}")
            logger.info(f"{'='*80}")

            try:
                results = self.test_recommendation_quality_deep(
                    agreement_id=customer['AgreementID'],
                    as_of_date=as_of_date,
                    test_days=30
                )

                all_results.append(results)
                total_brand_opportunities += results['brand_opportunities']
                total_analogue_opportunities += results['analogue_opportunities']
                total_category_opportunities += results['category_opportunities']
                total_misses += results['misses']
                total_hits += results['hits']

            except Exception as e:
                logger.error(f"Error testing agreement {customer['AgreementID']}: {e}")

        # Aggregate results
        logger.info(f"\n" + "="*80)
        logger.info("AGGREGATE RESULTS")
        logger.info("="*80)

        avg_precision = sum(r['precision'] for r in all_results) / len(all_results) if all_results else 0

        logger.info(f"\nBaseline V3.3 Performance:")
        logger.info(f"  Average Precision@20: {avg_precision*100:.1f}%")
        logger.info(f"  Total hits: {total_hits}")
        logger.info(f"  Total misses: {total_misses}")

        logger.info(f"\nV4 Feature Opportunities (among {total_misses} missed products):")

        brand_recovery = (total_brand_opportunities / total_misses * 100) if total_misses > 0 else 0
        logger.info(f"  Brand-based: {total_brand_opportunities} products ({brand_recovery:.1f}% of misses)")

        analogue_recovery = (total_analogue_opportunities / total_misses * 100) if total_misses > 0 else 0
        logger.info(f"  Analogue-based: {total_analogue_opportunities} products ({analogue_recovery:.1f}% of misses)")

        category_recovery = (total_category_opportunities / total_misses * 100) if total_misses > 0 else 0
        logger.info(f"  Category-based: {total_category_opportunities} products ({category_recovery:.1f}% of misses)")

        # Estimate V4 precision improvement
        # Assuming we can recover 50% of opportunities (conservative)
        conservative_recovery = (total_brand_opportunities + total_analogue_opportunities + total_category_opportunities) * 0.5
        potential_new_hits = total_hits + conservative_recovery
        total_recommended = len(all_results) * 20  # 20 recs per customer

        estimated_v4_precision = (potential_new_hits / total_recommended) if total_recommended > 0 else 0
        improvement = ((estimated_v4_precision - avg_precision) / avg_precision * 100) if avg_precision > 0 else 0

        logger.info(f"\nEstimated V4 Impact (conservative 50% recovery rate):")
        logger.info(f"  Current V3.3 Precision: {avg_precision*100:.1f}%")
        logger.info(f"  Estimated V4 Precision: {estimated_v4_precision*100:.1f}%")
        logger.info(f"  Improvement: +{improvement:.1f}% relative improvement")

        logger.info(f"\n" + "="*80)
        logger.info("CONCLUSION")
        logger.info("="*80)
        logger.info(f"V4 features (brand + analogue + category) can potentially recover")
        logger.info(f"{brand_recovery + analogue_recovery + category_recovery:.1f}% of currently missed products.")
        logger.info(f"This translates to an estimated {improvement:.1f}% improvement in precision.")

        return {
            'baseline_precision': avg_precision,
            'estimated_v4_precision': estimated_v4_precision,
            'improvement_pct': improvement,
            'brand_recovery': brand_recovery,
            'analogue_recovery': analogue_recovery,
            'category_recovery': category_recovery,
        }

    def close(self):
        self.conn.close()


def main():
    tester = DeepTestingFramework()

    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test(
            num_customers=15,
            as_of_date='2024-10-01'
        )

        logger.info(f"\n" + "="*80)
        logger.info("FINAL RESULTS")
        logger.info("="*80)
        logger.info(f"Baseline V3.3: {results['baseline_precision']*100:.2f}%")
        logger.info(f"Estimated V4: {results['estimated_v4_precision']*100:.2f}%")
        logger.info(f"Expected improvement: +{results['improvement_pct']:.1f}%")
        logger.info("="*80)

    finally:
        tester.close()


if __name__ == '__main__':
    main()
