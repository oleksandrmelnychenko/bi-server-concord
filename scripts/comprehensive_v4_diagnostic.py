#!/usr/bin/env python3
"""
Comprehensive V4 Diagnostic Script
Tests 100+ customers, analyzes scoring behavior, identifies root causes
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33
from scripts.improved_hybrid_recommender_v4 import ImprovedHybridRecommenderV4
from api.db_pool import get_connection

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class V4DiagnosticFramework:
    
    def __init__(self):
        self.results = []
        self.feature_stats = {
            'brand_matches': [],
            'analogue_matches': [],
            'category_matches': []
        }
    
    def get_test_agreements(self, target_count=150):
        """Find agreements with good activity for testing"""
        conn = get_connection()
        cursor = conn.cursor(as_dict=True)
        
        # Get agreements with at least 5 orders in recent history
        query = f"""
        SELECT TOP {target_count} ca.ID as AgreementID,
               COUNT(DISTINCT o.ID) as order_count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE o.Created >= '2024-08-01'
              AND o.Created < '2024-11-01'
        GROUP BY ca.ID
        HAVING COUNT(DISTINCT o.ID) >= 5
        ORDER BY COUNT(DISTINCT o.ID) DESC
        """
        
        cursor.execute(query)
        agreements = [(row['AgreementID'], row['order_count']) for row in cursor]
        cursor.close()
        conn.close()
        
        return agreements
    
    def analyze_single_agreement(self, agreement_id, as_of_date='2024-10-01', test_days=30):
        """Deep analysis of one agreement"""
        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
        test_start = as_of_datetime
        test_end = as_of_datetime + timedelta(days=test_days)
        
        # Get actual purchases
        conn = get_connection()
        try:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(f"""
            SELECT DISTINCT oi.ProductID
            FROM dbo.[Order] o
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.ClientAgreementID = {agreement_id}
                  AND o.Created >= '{test_start}'
                  AND o.Created < '{test_end}'
            """)
            actual_products = {row['ProductID'] for row in cursor}
            cursor.close()
        finally:
            conn.close()
        
        if not actual_products:
            return None
        
        # Get V3.3 recommendations WITH SCORES
        conn_v3 = get_connection()
        try:
            rec_v3 = ImprovedHybridRecommenderV33(conn=conn_v3, use_cache=False)
            recs_v3 = rec_v3.get_recommendations_for_agreement(
                agreement_id=agreement_id,
                as_of_date=as_of_date,
                top_n=20,
                include_discovery=False
            )
            v3_products = {r['product_id']: r['score'] for r in recs_v3}
            v3_hits = len(set(v3_products.keys()) & actual_products)
            v3_precision = v3_hits / 20 if recs_v3 else 0
        finally:
            conn_v3.close()
        
        # Get V4 recommendations WITH SCORES  
        conn_v4 = get_connection()
        try:
            rec_v4 = ImprovedHybridRecommenderV4(conn=conn_v4, use_cache=False)
            recs_v4 = rec_v4.get_recommendations_for_agreement(
                agreement_id=agreement_id,
                as_of_date=as_of_date,
                top_n=20,
                include_discovery=False
            )
            v4_products = {r['product_id']: r['score'] for r in recs_v4}
            v4_hits = len(set(v4_products.keys()) & actual_products)
            v4_precision = v4_hits / 20 if recs_v4 else 0
            
            # Analyze feature coverage for V4 products
            if recs_v4:
                product_ids = [r['product_id'] for r in recs_v4]
                
                # Check brand matches
                brand_scores = rec_v4.get_brand_scores(agreement_id, product_ids, as_of_date)
                brand_matches = sum(1 for score in brand_scores.values() if score > 0)
                
                # Check analogue matches
                analogue_scores = rec_v4.get_analogue_scores(agreement_id, product_ids, as_of_date)
                analogue_matches = sum(1 for score in analogue_scores.values() if score > 0)
                
                # Check category matches
                category_scores = rec_v4.get_category_scores(agreement_id, product_ids, as_of_date)
                category_matches = sum(1 for score in category_scores.values() if score > 0)
                
                self.feature_stats['brand_matches'].append(brand_matches / 20)
                self.feature_stats['analogue_matches'].append(analogue_matches / 20)
                self.feature_stats['category_matches'].append(category_matches / 20)
        finally:
            conn_v4.close()
        
        return {
            'agreement_id': agreement_id,
            'actual_count': len(actual_products),
            'v3_precision': v3_precision,
            'v3_hits': v3_hits,
            'v3_avg_score': sum(v3_products.values()) / len(v3_products) if v3_products else 0,
            'v4_precision': v4_precision,
            'v4_hits': v4_hits,
            'v4_avg_score': sum(v4_products.values()) / len(v4_products) if v4_products else 0,
            'improvement': v4_precision - v3_precision,
            'brand_coverage': brand_matches / 20 if recs_v4 else 0,
            'analogue_coverage': analogue_matches / 20 if recs_v4 else 0,
            'category_coverage': category_matches / 20 if recs_v4 else 0
        }
    
    def run_comprehensive_test(self, target_agreements=100):
        """Run test on many agreements"""
        print("=" * 80)
        print("COMPREHENSIVE V4 DIAGNOSTIC TEST")
        print("=" * 80)
        print(f"\nTarget: {target_agreements} agreements")
        print("Finding test agreements...")
        
        agreements = self.get_test_agreements(target_count=target_agreements + 50)
        print(f"Found {len(agreements)} candidate agreements")
        print(f"\nTesting with as-of-date: 2024-10-01, test window: 30 days\n")
        
        tested = 0
        for idx, (agreement_id, order_count) in enumerate(agreements, 1):
            if tested >= target_agreements:
                break
            
            print(f"  [{tested+1}/{target_agreements}] Agreement {agreement_id} (orders={order_count})...", end=" ", flush=True)
            
            try:
                result = self.analyze_single_agreement(agreement_id)
                if result:
                    self.results.append(result)
                    tested += 1
                    print(f"V3: {result['v3_precision']*100:.0f}%, V4: {result['v4_precision']*100:.0f}% ({result['improvement']*100:+.0f}%)")
                else:
                    print("Skip (no test purchases)")
            except Exception as e:
                print(f"Error: {str(e)[:50]}")
        
        print(f"\n✓ Successfully tested {len(self.results)} agreements\n")
        self.print_analysis()
    
    def print_analysis(self):
        """Print comprehensive analysis"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        # Overall metrics
        avg_v3 = sum(r['v3_precision'] for r in self.results) / len(self.results)
        avg_v4 = sum(r['v4_precision'] for r in self.results) / len(self.results)
        improvement = avg_v4 - avg_v3
        
        print(f"\nOverall Performance ({len(self.results)} agreements):")
        print(f"  V3.3 Average Precision@20: {avg_v3*100:.2f}%")
        print(f"  V4 Average Precision@20:   {avg_v4*100:.2f}%")
        print(f"  Absolute Improvement:       {improvement*100:+.2f}%")
        if avg_v3 > 0:
            print(f"  Relative Improvement:       {improvement/avg_v3*100:+.1f}%")
        
        # Win/loss/tie
        wins = sum(1 for r in self.results if r['improvement'] > 0)
        losses = sum(1 for r in self.results if r['improvement'] < 0)
        ties = sum(1 for r in self.results if r['improvement'] == 0)
        
        print(f"\nHead-to-Head:")
        print(f"  V4 Wins:   {wins}/{len(self.results)} ({wins/len(self.results)*100:.1f}%)")
        print(f"  V4 Losses: {losses}/{len(self.results)} ({losses/len(self.results)*100:.1f}%)")
        print(f"  Ties:      {ties}/{len(self.results)} ({ties/len(self.results)*100:.1f}%)")
        
        # Feature coverage analysis
        print(f"\n" + "=" * 80)
        print("FEATURE COVERAGE ANALYSIS")
        print("=" * 80)
        
        avg_brand = sum(self.feature_stats['brand_matches']) / len(self.feature_stats['brand_matches']) * 100
        avg_analogue = sum(self.feature_stats['analogue_matches']) / len(self.feature_stats['analogue_matches']) * 100
        avg_category = sum(self.feature_stats['category_matches']) / len(self.feature_stats['category_matches']) * 100
        
        print(f"\nAverage Feature Match Rates (among top-20 recommendations):")
        print(f"  Brand matches:    {avg_brand:.1f}%")
        print(f"  Analogue matches: {avg_analogue:.1f}%")
        print(f"  Category matches: {avg_category:.1f}%")
        
        # Score analysis
        print(f"\n" + "=" * 80)
        print("SCORE DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        avg_v3_score = sum(r['v3_avg_score'] for r in self.results) / len(self.results)
        avg_v4_score = sum(r['v4_avg_score'] for r in self.results) / len(self.results)
        
        print(f"\nAverage Recommendation Scores:")
        print(f"  V3.3 avg score: {avg_v3_score:.4f}")
        print(f"  V4 avg score:   {avg_v4_score:.4f}")
        print(f"  Score change:   {avg_v4_score - avg_v3_score:+.4f} ({(avg_v4_score - avg_v3_score)/avg_v3_score*100:+.1f}%)")
        
        # Top improvements and regressions
        print(f"\n" + "=" * 80)
        print("BEST AND WORST CASES")
        print("=" * 80)
        
        top_improvements = sorted(self.results, key=lambda x: x['improvement'], reverse=True)[:5]
        print(f"\nTop 5 Improvements:")
        for r in top_improvements:
            print(f"  Agreement {r['agreement_id']}: {r['v3_precision']*100:.0f}% → {r['v4_precision']*100:.0f}% ({r['improvement']*100:+.0f}%) " +
                  f"[Brand:{r['brand_coverage']*100:.0f}%, Analogue:{r['analogue_coverage']*100:.0f}%, Cat:{r['category_coverage']*100:.0f}%]")
        
        top_regressions = sorted(self.results, key=lambda x: x['improvement'])[:5]
        if top_regressions[0]['improvement'] < 0:
            print(f"\nTop 5 Regressions:")
            for r in top_regressions:
                print(f"  Agreement {r['agreement_id']}: {r['v3_precision']*100:.0f}% → {r['v4_precision']*100:.0f}% ({r['improvement']*100:.0f}%) " +
                      f"[Brand:{r['brand_coverage']*100:.0f}%, Analogue:{r['analogue_coverage']*100:.0f}%, Cat:{r['category_coverage']*100:.0f}%]")
        
        # Diagnosis
        print(f"\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)
        
        if avg_v4 > avg_v3:
            print(f"\n✅ V4 IS BETTER by {improvement*100:.2f}%")
            if avg_v4 >= 0.40:
                print(f"🎯 TARGET ACHIEVED: {avg_v4*100:.1f}% precision (target: 40-50%)")
            else:
                print(f"📊 Progress toward target: {avg_v4*100:.1f}% / 40-50%")
        else:
            print(f"\n⚠️  V4 IS WORSE by {abs(improvement)*100:.2f}%")
            print(f"\nLikely root causes:")
            print(f"  1. Weight balance issue: V4 features may not be providing enough signal")
            print(f"  2. Feature coverage too low: Only {avg_brand:.0f}% brand, {avg_analogue:.0f}% analogue, {avg_category:.0f}% category")
            print(f"  3. Binary scoring (0 or 1) vs continuous V3.3 scores may be too coarse")
            
            if avg_v4_score < avg_v3_score:
                print(f"  4. V4 scores are LOWER on average ({avg_v4_score:.3f} vs {avg_v3_score:.3f})")
                print(f"     → Indicates weight reduction was too aggressive")
        
        print("=" * 80)

if __name__ == '__main__':
    framework = V4DiagnosticFramework()
    framework.run_comprehensive_test(target_agreements=100)
