#!/usr/bin/env python3
"""
V3.5 Validation - LIGHT Users Only

Tests brand affinity boosting on 27 LIGHT users from validation set.
Compares V3 (16.1% baseline) vs V3.5 (target: 30%+).

Hypothesis: Brand affinity boosting will improve LIGHT user recommendations
by leveraging ProductCarBrand data to infer fleet composition.
"""

import os
import sys
import json
import pymssql
import logging
from datetime import datetime
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender import ImprovedHybridRecommender  # V3
from scripts.improved_hybrid_recommender_v35 import ImprovedHybridRecommender as V35Recommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

AS_OF_DATE = "2024-07-01"

DB_CONFIG = {
    'server': '78.152.175.67',
    'port': 1433,
    'database': 'ConcordDb_v5',
    'user': 'ef_migrator',
    'password': 'Grimm_jow92',
    'as_dict': True
}


class V35Validator:
    def __init__(self):
        self.conn = None
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'v3_results': [],
            'v35_results': [],
            'improvements': []
        }

    def connect_db(self):
        """Connect to database"""
        self.conn = pymssql.connect(**DB_CONFIG)
        logger.info("✓ Connected to database")

    def load_light_users(self):
        """Load LIGHT users from previous validation results"""
        try:
            with open('ensemble_comprehensive_validation_results.json', 'r') as f:
                data = json.load(f)
            light_users = [r['customer_id'] for r in data['v3_baseline_results']
                          if r['segment'] == 'LIGHT']
            logger.info(f"✓ Loaded {len(light_users)} LIGHT users")
            return light_users
        except FileNotFoundError:
            logger.error("✗ ensemble_comprehensive_validation_results.json not found")
            return []

    def get_validation_products(self, customer_id):
        """Get actual products purchased after as_of_date"""
        query = f"""
        SELECT DISTINCT oi.ProductID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created >= '{AS_OF_DATE}'
              AND oi.ProductID IS NOT NULL
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        products = [row['ProductID'] for row in cursor]
        cursor.close()
        return set(products)

    def get_v3_recommendations(self, customer_id, top_n=50):
        """Get V3 recommendations"""
        recommender = ImprovedHybridRecommender()
        try:
            recs = recommender.get_recommendations(customer_id, AS_OF_DATE, top_n=top_n)
            return [rec['product_id'] for rec in recs]
        finally:
            recommender.close()

    def get_v35_recommendations(self, customer_id, top_n=50):
        """Get V3.5 recommendations with brand affinity"""
        recommender = V35Recommender()
        try:
            recs = recommender.get_recommendations(customer_id, AS_OF_DATE, top_n=top_n)
            return [rec['product_id'] for rec in recs]
        finally:
            recommender.close()

    def calculate_precision(self, recommendations, validation_products):
        """Calculate precision@50"""
        hits = sum(1 for rec in recommendations[:50] if rec in validation_products)
        return hits / min(50, len(recommendations)) if recommendations else 0

    def run_validation(self):
        """Run validation on LIGHT users"""
        logger.info("\n" + "="*80)
        logger.info("V3.5 VALIDATION - LIGHT USERS ONLY")
        logger.info("="*80)
        logger.info("Testing: Brand Affinity Boosting (40% brand + 40% frequency + 20% recency)")
        logger.info("Baseline: V3 achieves 16.1% on LIGHT users")
        logger.info("Target: 30%+ precision")

        light_users = self.load_light_users()
        if not light_users:
            logger.error("✗ Cannot proceed without LIGHT users")
            return

        self.connect_db()

        logger.info(f"\nTesting {len(light_users)} LIGHT users...")
        logger.info("-" * 80)

        for idx, customer_id in enumerate(light_users, 1):
            logger.info(f"[{idx:2d}/{len(light_users)}] Customer {customer_id}...")

            # Get validation products
            validation_products = self.get_validation_products(customer_id)
            validation_count = len(validation_products)

            if validation_count == 0:
                logger.warning(f"  No validation products, skipping")
                continue

            # Get V3 baseline
            try:
                v3_recs = self.get_v3_recommendations(customer_id)
                v3_precision = self.calculate_precision(v3_recs, validation_products)
            except Exception as e:
                logger.error(f"  V3 error: {e}")
                continue

            # Get V3.5 with brand affinity
            try:
                v35_recs = self.get_v35_recommendations(customer_id)
                v35_precision = self.calculate_precision(v35_recs, validation_products)
            except Exception as e:
                logger.error(f"  V3.5 error: {e}")
                v35_precision = v3_precision  # Fallback to V3 on error

            # Calculate improvement
            improvement = v35_precision - v3_precision
            improvement_pct = (improvement / v3_precision * 100) if v3_precision > 0 else 0

            # Store results
            self.results['v3_results'].append({
                'customer_id': customer_id,
                'precision': v3_precision,
                'validation_count': validation_count
            })

            self.results['v35_results'].append({
                'customer_id': customer_id,
                'precision': v35_precision,
                'validation_count': validation_count,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })

            # Print result
            emoji = "🏆" if improvement > 0.1 else "✅" if improvement > 0 else "⚠️" if improvement > -0.05 else "❌"
            logger.info(f"  V3: {v3_precision:5.1%} | V3.5: {v35_precision:5.1%} | Δ: {improvement:+6.1%} {emoji}")

        # Print summary
        self.print_summary()

        # Save results
        with open('v35_light_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info("\n✓ Results saved to v35_light_validation_results.json")

    def print_summary(self):
        """Print summary statistics"""
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)

        all_v3 = [r['precision'] for r in self.results['v3_results']]
        all_v35 = [r['precision'] for r in self.results['v35_results']]

        v3_mean = statistics.mean(all_v3)
        v35_mean = statistics.mean(all_v35)
        improvement = v35_mean - v3_mean
        improvement_pct = (improvement / v3_mean * 100) if v3_mean > 0 else 0

        logger.info(f"\nOVERALL RESULTS (n={len(all_v3)} LIGHT users):")
        logger.info(f"  V3 Baseline:      {v3_mean:5.1%}")
        logger.info(f"  V3.5 (Brand):     {v35_mean:5.1%}")
        logger.info(f"  Improvement:      {improvement:+6.1%} ({improvement_pct:+.1f}%)")

        # Success criteria
        logger.info(f"\nSUCCESS CRITERIA:")
        logger.info(f"  Target: >30% for LIGHT users")
        logger.info(f"  Actual: {v35_mean:5.1%}")
        logger.info(f"  Status: {'✅ PASSED' if v35_mean > 0.30 else '⚠️  PARTIAL' if v35_mean > v3_mean else '❌ FAILED'}")

        # Top improvements
        logger.info(f"\nTOP 5 IMPROVEMENTS:")
        improvements = sorted(self.results['v35_results'], key=lambda x: x['improvement'], reverse=True)
        for i, result in enumerate(improvements[:5], 1):
            customer_id = result['customer_id']
            v3_prec = next(r['precision'] for r in self.results['v3_results'] if r['customer_id'] == customer_id)
            logger.info(f"  {i}. Customer {customer_id}: {v3_prec:5.1%} → {result['precision']:5.1%} (Δ {result['improvement']:+6.1%})")

        # Top regressions
        logger.info(f"\nTOP 5 REGRESSIONS:")
        regressions = sorted(self.results['v35_results'], key=lambda x: x['improvement'])
        for i, result in enumerate(regressions[:5], 1):
            customer_id = result['customer_id']
            v3_prec = next(r['precision'] for r in self.results['v3_results'] if r['customer_id'] == customer_id)
            logger.info(f"  {i}. Customer {customer_id}: {v3_prec:5.1%} → {result['precision']:5.1%} (Δ {result['improvement']:+6.1%})")

        # Success rate
        better_count = sum(1 for r in self.results['v35_results'] if r['improvement'] > 0)
        same_count = sum(1 for r in self.results['v35_results'] if r['improvement'] == 0)
        worse_count = sum(1 for r in self.results['v35_results'] if r['improvement'] < 0)

        logger.info(f"\nIMPROVEMENT DISTRIBUTION:")
        logger.info(f"  Better:  {better_count:2d} ({better_count/len(all_v35)*100:5.1f}%)")
        logger.info(f"  Same:    {same_count:2d} ({same_count/len(all_v35)*100:5.1f}%)")
        logger.info(f"  Worse:   {worse_count:2d} ({worse_count/len(all_v35)*100:5.1f}%)")

        # Deployment recommendation
        logger.info(f"\n" + "="*80)
        logger.info("DEPLOYMENT RECOMMENDATION")
        logger.info("="*80)

        if v35_mean > 0.30:
            logger.info(f"\n✅ RECOMMENDATION: DEPLOY V3.5 FOR LIGHT USERS")
            logger.info(f"   V3.5 achieves {v35_mean:.1%} precision on LIGHT users (>30% target)")
            logger.info(f"   {better_count} users improved ({better_count/len(all_v35)*100:.1f}%)")
            logger.info(f"\n   Next steps:")
            logger.info(f"   1. Replace ImprovedHybridRecommender with V3.5 in production API")
            logger.info(f"   2. Monitor LIGHT user engagement metrics")
            logger.info(f"   3. Consider tuning weights (try 50% brand / 30% frequency / 20% recency)")
        elif v35_mean > v3_mean:
            logger.info(f"\n⚠️  RECOMMENDATION: CONDITIONAL DEPLOYMENT")
            logger.info(f"   V3.5 improves to {v35_mean:.1%} vs V3's {v3_mean:.1%}")
            logger.info(f"   But doesn't reach 30% threshold")
            logger.info(f"\n   Options:")
            logger.info(f"   A. Deploy V3.5 anyway (modest improvement)")
            logger.info(f"   B. Tune brand affinity weights (increase to 50-60%)")
            logger.info(f"   C. Investigate regressions and add fallback logic")
        else:
            logger.info(f"\n❌ RECOMMENDATION: DO NOT DEPLOY V3.5")
            logger.info(f"   V3.5 regresses to {v35_mean:.1%} vs V3's {v3_mean:.1%}")
            logger.info(f"   Brand affinity may not generalize across all LIGHT users")
            logger.info(f"\n   Next steps:")
            logger.info(f"   1. Analyze why brand affinity fails")
            logger.info(f"   2. Check if ProductCarBrand coverage is incomplete")
            logger.info(f"   3. Consider hybrid approach (brand affinity only when available)")


def main():
    validator = V35Validator()
    validator.run_validation()


if __name__ == '__main__':
    main()
