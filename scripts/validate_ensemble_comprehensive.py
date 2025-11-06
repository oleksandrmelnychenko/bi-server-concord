#!/usr/bin/env python3
"""
Comprehensive Ensemble Validation - V3 Hybrid + GNN (70/30)

Tests the ensemble approach (70% V3 Hybrid + 30% GNN) on all 50 validation customers
to determine if the 75.2% result on 5 HEAVY customers generalizes across all segments.

Compares ensemble performance against V3 baseline (27-40% precision).
"""

import os
import sys
import logging
import json
import pymssql
import torch
from datetime import datetime
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender import ImprovedHybridRecommender  # V3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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


class EnsembleValidator:
    def __init__(self):
        self.conn = None
        self.gnn_model = None
        self.gnn_metadata = None
        self.edge_index_dict = None
        self.device = torch.device('cpu')

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'ensemble_config': {'hybrid_weight': 0.7, 'gnn_weight': 0.3},
            'v3_baseline_results': [],
            'ensemble_results': [],
            'improvements': []
        }

    def connect_db(self):
        """Connect to database"""
        self.conn = pymssql.connect(**DB_CONFIG)
        logger.info("✓ Connected to database")

    def load_gnn_model(self):
        """Load trained GNN model"""
        try:
            from scripts.validate_gnn_recommender import load_trained_model
            from scripts.build_gnn_recommender import load_graph_data

            logger.info("\n📦 Loading GNN model...")
            self.gnn_model, self.gnn_metadata = load_trained_model(device=self.device)
            self.edge_index_dict, _ = load_graph_data()
            logger.info("✓ GNN model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load GNN model: {e}")
            return False

    def load_validation_customers(self):
        """Load 50 validation customers from V3/V4 tests"""
        try:
            with open('validation_50_customers_results.json', 'r') as f:
                data = json.load(f)
            customers = [r['customer_id'] for r in data['results']]
            logger.info(f"✓ Loaded {len(customers)} validation customers")
            return customers
        except FileNotFoundError:
            logger.error("✗ validation_50_customers_results.json not found")
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

    def get_v3_recommendations(self, customer_id, top_n=35):
        """Get V3 hybrid recommendations"""
        recommender = ImprovedHybridRecommender()
        try:
            recs = recommender.get_recommendations(customer_id, AS_OF_DATE, top_n=top_n)
            return [rec['product_id'] for rec in recs]
        finally:
            recommender.close()

    def get_gnn_recommendations(self, customer_id, top_n=15):
        """Get GNN recommendations"""
        try:
            from scripts.validate_gnn_recommender import get_recommendations_gnn, get_test_data

            train_products, _ = get_test_data(customer_id)
            gnn_recs = get_recommendations_gnn(
                self.gnn_model, self.edge_index_dict, customer_id,
                self.gnn_metadata, train_products, top_n=top_n, device=self.device
            )
            return [int(r) for r in gnn_recs]
        except Exception as e:
            logger.warning(f"    GNN failed for customer {customer_id}: {e}")
            return []

    def merge_recommendations(self, hybrid_recs, gnn_recs, target=50):
        """Merge hybrid and GNN recommendations, removing duplicates"""
        seen = set()
        final_recs = []

        # Add hybrid recs first (70%)
        for prod_id in hybrid_recs:
            if prod_id not in seen:
                final_recs.append(prod_id)
                seen.add(prod_id)
                if len(final_recs) >= target:
                    break

        # Add GNN recs (30%)
        for prod_id in gnn_recs:
            if prod_id not in seen:
                final_recs.append(prod_id)
                seen.add(prod_id)
                if len(final_recs) >= target:
                    break

        return final_recs

    def calculate_precision(self, recommendations, validation_products):
        """Calculate precision@50"""
        hits = sum(1 for rec in recommendations[:50] if rec in validation_products)
        return hits / min(50, len(recommendations)) if recommendations else 0

    def run_validation(self):
        """Run comprehensive validation on all 50 customers"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE ENSEMBLE VALIDATION - V3 + GNN (70/30)")
        logger.info("="*80)

        customers = self.load_validation_customers()
        if not customers:
            logger.error("✗ Cannot proceed without validation customers")
            return

        self.connect_db()

        if not self.load_gnn_model():
            logger.error("✗ Cannot proceed without GNN model")
            return

        # Results by segment
        v3_by_segment = defaultdict(list)
        ensemble_by_segment = defaultdict(list)
        segment_mapping = {}

        logger.info(f"\nTesting {len(customers)} customers...")
        logger.info("-" * 80)

        for idx, customer_id in enumerate(customers, 1):
            logger.info(f"[{idx:2d}/{len(customers)}] Customer {customer_id}...")

            # Get validation products
            validation_products = self.get_validation_products(customer_id)
            validation_count = len(validation_products)

            if validation_count == 0:
                logger.warning(f"  No validation products, skipping")
                continue

            # Get V3 recommendations (for baseline)
            try:
                v3_recs_full = self.get_v3_recommendations(customer_id, top_n=50)
                v3_precision = self.calculate_precision(v3_recs_full, validation_products)

                # Get segment from V3
                v3_recommender = ImprovedHybridRecommender()
                segment, subsegment = v3_recommender.classify_customer(customer_id, AS_OF_DATE)
                v3_recommender.close()

                base_segment = segment  # HEAVY, REGULAR, or LIGHT
                segment_mapping[customer_id] = base_segment
            except Exception as e:
                logger.error(f"  V3 error: {e}")
                continue

            # Get ensemble recommendations (70/30 split)
            try:
                hybrid_recs = self.get_v3_recommendations(customer_id, top_n=35)  # 70%
                gnn_recs = self.get_gnn_recommendations(customer_id, top_n=15)     # 30%
                ensemble_recs = self.merge_recommendations(hybrid_recs, gnn_recs, target=50)
                ensemble_precision = self.calculate_precision(ensemble_recs, validation_products)
            except Exception as e:
                logger.error(f"  Ensemble error: {e}")
                continue

            # Calculate improvement
            improvement = ensemble_precision - v3_precision
            improvement_pct = (improvement / v3_precision * 100) if v3_precision > 0 else 0

            # Store results
            self.results['v3_baseline_results'].append({
                'customer_id': customer_id,
                'precision': v3_precision,
                'segment': base_segment,
                'validation_count': validation_count
            })

            self.results['ensemble_results'].append({
                'customer_id': customer_id,
                'precision': ensemble_precision,
                'segment': base_segment,
                'validation_count': validation_count,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })

            # Track by segment
            v3_by_segment[base_segment].append(v3_precision)
            ensemble_by_segment[base_segment].append(ensemble_precision)

            # Print result
            emoji = "🏆" if improvement > 0.1 else "✅" if improvement > 0 else "⚠️" if improvement > -0.05 else "❌"
            logger.info(f"  V3: {v3_precision:5.1%} | Ensemble: {ensemble_precision:5.1%} | Δ: {improvement:+6.1%} ({base_segment}) {emoji}")

        # Calculate and print summary statistics
        self.print_summary(v3_by_segment, ensemble_by_segment)

        # Save results
        with open('ensemble_comprehensive_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info("\n✓ Results saved to ensemble_comprehensive_validation_results.json")

    def print_summary(self, v3_by_segment, ensemble_by_segment):
        """Print comprehensive summary statistics"""
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)

        # Overall statistics
        all_v3 = [r['precision'] for r in self.results['v3_baseline_results']]
        all_ensemble = [r['precision'] for r in self.results['ensemble_results']]

        v3_mean = statistics.mean(all_v3)
        ensemble_mean = statistics.mean(all_ensemble)
        improvement = ensemble_mean - v3_mean
        improvement_pct = (improvement / v3_mean * 100) if v3_mean > 0 else 0

        logger.info(f"\nOVERALL RESULTS (n={len(all_v3)} customers):")
        logger.info(f"  V3 Baseline:      {v3_mean:5.1%}")
        logger.info(f"  Ensemble (70/30): {ensemble_mean:5.1%}")
        logger.info(f"  Improvement:      {improvement:+6.1%} ({improvement_pct:+.1f}%)")

        # Success criteria
        logger.info(f"\nSUCCESS CRITERIA:")
        logger.info(f"  Target: >50% overall")
        logger.info(f"  Actual: {ensemble_mean:5.1%}")
        logger.info(f"  Status: {'✅ PASSED' if ensemble_mean > 0.50 else '❌ FAILED'}")

        # Segment-specific statistics
        logger.info(f"\nSEGMENT-SPECIFIC RESULTS:")
        logger.info(f"{'Segment':<12} {'V3 Baseline':>14} {'Ensemble':>14} {'Improvement':>14} {'Count':>8}")
        logger.info("-" * 80)

        for segment in sorted(v3_by_segment.keys()):
            if segment in ensemble_by_segment:
                v3_seg_mean = statistics.mean(v3_by_segment[segment])
                ensemble_seg_mean = statistics.mean(ensemble_by_segment[segment])
                seg_improvement = ensemble_seg_mean - v3_seg_mean
                seg_improvement_pct = (seg_improvement / v3_seg_mean * 100) if v3_seg_mean > 0 else 0
                count = len(v3_by_segment[segment])

                logger.info(f"{segment:<12} {v3_seg_mean:13.1%} {ensemble_seg_mean:13.1%} {seg_improvement:+13.1%} ({seg_improvement_pct:+5.1f}%)  n={count:2d}")

        # Top improvements
        logger.info(f"\nTOP 5 IMPROVEMENTS:")
        improvements = sorted(self.results['ensemble_results'], key=lambda x: x['improvement'], reverse=True)
        for i, result in enumerate(improvements[:5], 1):
            customer_id = result['customer_id']
            v3_prec = next(r['precision'] for r in self.results['v3_baseline_results'] if r['customer_id'] == customer_id)
            logger.info(f"  {i}. Customer {customer_id}: {v3_prec:5.1%} → {result['precision']:5.1%} (Δ {result['improvement']:+6.1%})")

        # Top regressions
        logger.info(f"\nTOP 5 REGRESSIONS:")
        regressions = sorted(self.results['ensemble_results'], key=lambda x: x['improvement'])
        for i, result in enumerate(regressions[:5], 1):
            customer_id = result['customer_id']
            v3_prec = next(r['precision'] for r in self.results['v3_baseline_results'] if r['customer_id'] == customer_id)
            logger.info(f"  {i}. Customer {customer_id}: {v3_prec:5.1%} → {result['precision']:5.1%} (Δ {result['improvement']:+6.1%})")

        # Success rate
        better_count = sum(1 for r in self.results['ensemble_results'] if r['improvement'] > 0)
        same_count = sum(1 for r in self.results['ensemble_results'] if r['improvement'] == 0)
        worse_count = sum(1 for r in self.results['ensemble_results'] if r['improvement'] < 0)

        logger.info(f"\nIMPROVEMENT DISTRIBUTION:")
        logger.info(f"  Better:  {better_count:2d} ({better_count/len(all_ensemble)*100:5.1f}%)")
        logger.info(f"  Same:    {same_count:2d} ({same_count/len(all_ensemble)*100:5.1f}%)")
        logger.info(f"  Worse:   {worse_count:2d} ({worse_count/len(all_ensemble)*100:5.1f}%)")

        # Final recommendation
        logger.info(f"\n" + "="*80)
        logger.info("DEPLOYMENT RECOMMENDATION")
        logger.info("="*80)

        if ensemble_mean > 0.50:
            logger.info(f"\n✅ RECOMMENDATION: DEPLOY ENSEMBLE TO PRODUCTION")
            logger.info(f"   Ensemble achieves {ensemble_mean:.1%} overall precision")
            logger.info(f"   {better_count} customers improved ({better_count/len(all_ensemble)*100:.1f}%)")
            logger.info(f"\n   Next steps:")
            logger.info(f"   1. Create ensemble recommender class for production")
            logger.info(f"   2. Update API to use 70% V3 + 30% GNN")
            logger.info(f"   3. Deploy and monitor real-world performance")
        elif ensemble_mean > v3_mean:
            logger.info(f"\n⚠️  RECOMMENDATION: CONDITIONAL DEPLOYMENT")
            logger.info(f"   Ensemble improves to {ensemble_mean:.1%} vs V3's {v3_mean:.1%}")
            logger.info(f"   But doesn't reach 50% threshold")
            logger.info(f"\n   Options:")
            logger.info(f"   A. Deploy ensemble (modest improvement)")
            logger.info(f"   B. Optimize ensemble weights (try 60/40, 80/20)")
            logger.info(f"   C. Deploy segment-specific (ensemble for HEAVY only)")
        else:
            logger.info(f"\n❌ RECOMMENDATION: DO NOT DEPLOY ENSEMBLE")
            logger.info(f"   Ensemble regresses to {ensemble_mean:.1%} vs V3's {v3_mean:.1%}")
            logger.info(f"   {worse_count} customers performed worse ({worse_count/len(all_ensemble)*100:.1f}%)")
            logger.info(f"\n   Next steps:")
            logger.info(f"   1. Keep V3 as production recommender")
            logger.info(f"   2. Investigate why GNN adds noise")
            logger.info(f"   3. Consider alternative ML approaches")


def main():
    validator = EnsembleValidator()
    validator.run_validation()


if __name__ == '__main__':
    main()
