#!/usr/bin/env python3
"""
V4 Validation Script - Direct Comparison with V3

Tests both V3 and V4 on the same 50 validation customers to measure improvements.
Directly instantiates recommenders instead of using API for faster testing.
"""

import json
import pymssql
import logging
from datetime import datetime
from collections import defaultdict
import statistics

# Import both recommenders
import sys
sys.path.append('/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts')
from improved_hybrid_recommender import ImprovedHybridRecommender  # V3
from collaborative_hybrid_recommender_v4 import CollaborativeHybridRecommenderV4  # V4

AS_OF_DATE = "2024-07-01"

# Database config
DB_CONFIG = {
    'server': '78.152.175.67',
    'port': 1433,
    'database': 'ConcordDb_v5',
    'user': 'ef_migrator',
    'password': 'Grimm_jow92',
    'as_dict': True
}

logging.basicConfig(
    level=logging.WARNING,  # Only show warnings/errors (suppress info logs)
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class V4Validator:
    def __init__(self):
        self.conn = None
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'v3_results': [],
            'v4_results': []
        }

    def connect_db(self):
        """Connect to database"""
        self.conn = pymssql.connect(**DB_CONFIG)
        print("✓ Connected to database")

    def load_validation_customers(self):
        """Load 50 validation customers from previous test"""
        try:
            with open('validation_50_customers_results.json', 'r') as f:
                data = json.load(f)
            customers = [r['customer_id'] for r in data['results']]
            print(f"✓ Loaded {len(customers)} validation customers")
            return customers
        except FileNotFoundError:
            print("✗ validation_50_customers_results.json not found")
            # Try loading from comprehensive_api_test_results.json
            try:
                with open('comprehensive_api_test_results.json', 'r') as f:
                    data = json.load(f)
                customers = [r['customer_id'] for r in data['phases']['phase1']['results']]
                print(f"✓ Loaded {len(customers)} validation customers from comprehensive test")
                return customers
            except:
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

    def count_validation_products(self, customer_id):
        """Get count of validation products"""
        query = f"""
        SELECT COUNT(DISTINCT oi.ProductID) as count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created >= '{AS_OF_DATE}'
              AND oi.ProductID IS NOT NULL
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        cursor.close()
        return row['count'] if row else 0

    def calculate_precision(self, recommendations, validation_products):
        """Calculate precision@50"""
        hits = sum(1 for rec in recommendations[:50] if rec['product_id'] in validation_products)
        return hits / min(50, len(recommendations)) if recommendations else 0

    def run_validation(self):
        """Run validation comparing V3 vs V4"""
        print("\n" + "="*80)
        print("V4 VALIDATION - Direct Comparison with V3")
        print("="*80)

        customers = self.load_validation_customers()
        if not customers:
            print("✗ Cannot proceed without validation customers")
            return

        self.connect_db()

        # Initialize recommenders with shared connection
        print("\n✓ Initializing V3 recommender...")
        recommender_v3 = ImprovedHybridRecommender()

        print("✓ Initializing V4 recommender...")
        recommender_v4 = CollaborativeHybridRecommenderV4()

        # Results by segment
        v3_by_segment = defaultdict(list)
        v4_by_segment = defaultdict(list)

        print(f"\nTesting {len(customers)} customers...")
        print("-" * 80)

        for idx, customer_id in enumerate(customers, 1):
            print(f"[{idx:2d}/{len(customers)}] Customer {customer_id}...", end=' ', flush=True)

            # Get validation products
            validation_products = self.get_validation_products(customer_id)
            validation_count = len(validation_products)

            # Get V3 recommendations
            try:
                v3_recs = recommender_v3.get_recommendations(customer_id, AS_OF_DATE, top_n=50)
                v3_precision = self.calculate_precision(v3_recs, validation_products)
                v3_segment = v3_recs[0]['segment'] if v3_recs else 'UNKNOWN'
            except Exception as e:
                print(f"\n✗ V3 error for customer {customer_id}: {e}")
                v3_precision = 0
                v3_segment = 'ERROR'

            # Get V4 recommendations
            try:
                v4_recs = recommender_v4.get_recommendations(customer_id, AS_OF_DATE, top_n=50)
                v4_precision = self.calculate_precision(v4_recs, validation_products)
                v4_segment = v4_recs[0]['segment'] if v4_recs else 'UNKNOWN'
            except Exception as e:
                print(f"\n✗ V4 error for customer {customer_id}: {e}")
                v4_precision = 0
                v4_segment = 'ERROR'

            # Calculate improvement
            improvement = v4_precision - v3_precision
            improvement_pct = (improvement / v3_precision * 100) if v3_precision > 0 else 0

            # Extract base segment
            base_segment = v3_segment.split('_')[0] if v3_segment else 'UNKNOWN'

            # Store results
            self.results['v3_results'].append({
                'customer_id': customer_id,
                'precision': v3_precision,
                'segment': v3_segment,
                'validation_count': validation_count
            })

            self.results['v4_results'].append({
                'customer_id': customer_id,
                'precision': v4_precision,
                'segment': v4_segment,
                'validation_count': validation_count,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })

            # Track by segment
            v3_by_segment[base_segment].append(v3_precision)
            v4_by_segment[base_segment].append(v4_precision)

            # Print result
            print(f"V3: {v3_precision:5.1%} | V4: {v4_precision:5.1%} | Δ: {improvement:+6.1%} ({base_segment})")

        # Close recommenders
        recommender_v3.close()
        recommender_v4.close()

        # Calculate and print summary statistics
        self.print_summary(v3_by_segment, v4_by_segment)

        # Save results
        with open('v4_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\n✓ Results saved to v4_validation_results.json")

    def print_summary(self, v3_by_segment, v4_by_segment):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        # Overall statistics
        all_v3 = [r['precision'] for r in self.results['v3_results']]
        all_v4 = [r['precision'] for r in self.results['v4_results']]

        v3_mean = statistics.mean(all_v3)
        v4_mean = statistics.mean(all_v4)
        improvement = v4_mean - v3_mean
        improvement_pct = (improvement / v3_mean * 100) if v3_mean > 0 else 0

        print(f"\nOVERALL RESULTS (n={len(all_v3)} customers):")
        print(f"  V3 Precision:      {v3_mean:5.1%}")
        print(f"  V4 Precision:      {v4_mean:5.1%}")
        print(f"  Improvement:       {improvement:+6.1%} ({improvement_pct:+.1f}%)")

        # Segment-specific statistics
        print(f"\nSEGMENT-SPECIFIC RESULTS:")
        print(f"{'Segment':<12} {'V3 Precision':>14} {'V4 Precision':>14} {'Improvement':>14} {'Count':>8}")
        print("-" * 80)

        for segment in sorted(v3_by_segment.keys()):
            if segment in v4_by_segment:
                v3_seg_mean = statistics.mean(v3_by_segment[segment])
                v4_seg_mean = statistics.mean(v4_by_segment[segment])
                seg_improvement = v4_seg_mean - v3_seg_mean
                seg_improvement_pct = (seg_improvement / v3_seg_mean * 100) if v3_seg_mean > 0 else 0
                count = len(v3_by_segment[segment])

                print(f"{segment:<12} {v3_seg_mean:13.1%} {v4_seg_mean:13.1%} {seg_improvement:+13.1%} ({seg_improvement_pct:+5.1f}%)  n={count:2d}")

        # Top improvements
        print(f"\nTOP 5 IMPROVEMENTS:")
        improvements = sorted(self.results['v4_results'], key=lambda x: x['improvement'], reverse=True)
        for i, result in enumerate(improvements[:5], 1):
            customer_id = result['customer_id']
            v3_prec = next(r['precision'] for r in self.results['v3_results'] if r['customer_id'] == customer_id)
            print(f"  {i}. Customer {customer_id}: {v3_prec:5.1%} → {result['precision']:5.1%} (Δ {result['improvement']:+6.1%})")

        # Top regressions
        print(f"\nTOP 5 REGRESSIONS:")
        regressions = sorted(self.results['v4_results'], key=lambda x: x['improvement'])
        for i, result in enumerate(regressions[:5], 1):
            customer_id = result['customer_id']
            v3_prec = next(r['precision'] for r in self.results['v3_results'] if r['customer_id'] == customer_id)
            print(f"  {i}. Customer {customer_id}: {v3_prec:5.1%} → {result['precision']:5.1%} (Δ {result['improvement']:+6.1%})")

        # Success rate
        better_count = sum(1 for r in self.results['v4_results'] if r['improvement'] > 0)
        same_count = sum(1 for r in self.results['v4_results'] if r['improvement'] == 0)
        worse_count = sum(1 for r in self.results['v4_results'] if r['improvement'] < 0)

        print(f"\nIMPROVEMENT DISTRIBUTION:")
        print(f"  Better:  {better_count:2d} ({better_count/len(all_v4)*100:5.1f}%)")
        print(f"  Same:    {same_count:2d} ({same_count/len(all_v4)*100:5.1f}%)")
        print(f"  Worse:   {worse_count:2d} ({worse_count/len(all_v4)*100:5.1f}%)")


def main():
    validator = V4Validator()
    validator.run_validation()


if __name__ == '__main__':
    main()
