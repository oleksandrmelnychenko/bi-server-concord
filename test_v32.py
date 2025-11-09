#!/usr/bin/env python3
"""
Test V3.2 Implementation

Validates:
1. Heavy users get discovery (no longer skipped)
2. Strict 20+5 mix for all customers
3. Product diversity (max 3 per group)
"""

import sys
from datetime import datetime
from scripts.improved_hybrid_recommender_v32 import ImprovedHybridRecommenderV32
from api.db_pool import get_connection

def test_v32():
    """Test V3.2 on sample customers"""

    # Test customers (different segments)
    test_customers = [
        (410169, "HEAVY"),
        (410175, "LIGHT"),
        (410176, "REGULAR_EXPLORATORY"),
        (410180, "HEAVY")
    ]

    conn = get_connection()
    recommender = ImprovedHybridRecommenderV32(conn=conn, use_cache=False)

    as_of_date = '2024-07-01'

    print("=" * 80)
    print("V3.2 QUALITY IMPROVEMENTS TEST")
    print("=" * 80)
    print()

    for customer_id, expected_segment in test_customers:
        print(f"\n{'=' * 80}")
        print(f"Testing Customer {customer_id} (Expected: {expected_segment})")
        print(f"{'=' * 80}")

        # Generate recommendations
        recs = recommender.get_recommendations(
            customer_id=customer_id,
            as_of_date=as_of_date,
            top_n=25,
            repurchase_count=20,
            discovery_count=5,
            include_discovery=True
        )

        # Count by source
        repurchase_count = sum(1 for r in recs if r['source'] == 'repurchase')
        discovery_count = sum(1 for r in recs if r['source'] == 'discovery')

        print(f"\n✓ Total recommendations: {len(recs)}")
        print(f"  - Repurchase: {repurchase_count}")
        print(f"  - Discovery: {discovery_count}")

        # Test 1: Heavy users should get discovery
        if "HEAVY" in expected_segment:
            if discovery_count > 0:
                print(f"\n✅ Test 1 PASSED: Heavy user got {discovery_count} discovery products")
            else:
                print(f"\n❌ Test 1 FAILED: Heavy user got 0 discovery products")

        # Test 2: Strict 20+5 mix (or close to it)
        if repurchase_count == 20 and discovery_count == 5:
            print(f"✅ Test 2 PASSED: Exact 20+5 mix")
        elif len(recs) == 25 and repurchase_count >= 15 and discovery_count >= 3:
            print(f"⚠️  Test 2 PARTIAL: Got {repurchase_count}+{discovery_count} (target: 20+5)")
        else:
            print(f"❌ Test 2 FAILED: Got {repurchase_count}+{discovery_count} (expected: 20+5)")

        # Test 3: Product diversity - get product groups
        product_ids = [r['product_id'] for r in recs]
        groups = recommender.get_product_groups(product_ids)

        # Count products per group
        from collections import Counter
        group_counts = Counter(groups.values())
        max_per_group = max(group_counts.values()) if group_counts else 0

        print(f"\n✓ Product diversity:")
        print(f"  - {len(groups)}/{len(recs)} products have groups")
        print(f"  - {len(group_counts)} unique groups")
        print(f"  - Max products per group: {max_per_group}")

        if max_per_group <= 3:
            print(f"✅ Test 3 PASSED: Max {max_per_group} products per group (limit: 3)")
        else:
            print(f"❌ Test 3 FAILED: Max {max_per_group} products per group (limit: 3)")

        # Show top 10 recommendations
        print(f"\nTop 10 recommendations:")
        print(f"{'Rank':<6} {'Product ID':<12} {'Score':<8} {'Source':<12}")
        print("-" * 50)
        for r in recs[:10]:
            print(f"{r['rank']:<6} {r['product_id']:<12} {r['score']:<8.4f} {r['source']:<12}")

    recommender.close()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    test_v32()
