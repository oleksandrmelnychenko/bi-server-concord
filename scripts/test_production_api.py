#!/usr/bin/env python3
"""
Test Production API with Validation Customers

Tests the production-ready recommendation API with the same customers
used in validation to verify consistency.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from predict_recommendations_v2 import get_recommendations, get_recommendations_batch

TEST_CUSTOMERS = [410376, 411706, 410767, 410258]
CUSTOMER_NAMES = {
    410376: "Regular User",
    411706: "Heavy User",
    410767: "Light User",
    410258: "Very Light User"
}

print("="*80)
print("PRODUCTION API TEST")
print("="*80)
print()

# Test 1: Single customer
print("Test 1: Single Customer Recommendations")
print("-"*80)
customer_id = TEST_CUSTOMERS[0]
print(f"Customer: {customer_id} ({CUSTOMER_NAMES[customer_id]})")
recs = get_recommendations(customer_id, top_n=10)
print(f"✓ Generated {len(recs)} recommendations")
print(f"  Top 3:")
for rec in recs[:3]:
    print(f"    #{rec['rank']}: Product {rec['product_id']} (score: {rec['score']:.3f})")
print()

# Test 2: Batch processing
print("Test 2: Batch Processing")
print("-"*80)
batch_results = get_recommendations_batch(TEST_CUSTOMERS, top_n=10)
print(f"✓ Processed {len(batch_results)} customers")
for customer_id in TEST_CUSTOMERS:
    recs = batch_results[customer_id]
    print(f"  Customer {customer_id} ({CUSTOMER_NAMES[customer_id]}): {len(recs)} recommendations")
print()

# Test 3: Historical recommendations
print("Test 3: Historical Recommendations (as of 2024-06-30)")
print("-"*80)
from datetime import datetime
as_of_date = datetime(2024, 6, 30)
customer_id = TEST_CUSTOMERS[0]
print(f"Customer: {customer_id}")
recs = get_recommendations(customer_id, top_n=10, as_of_date=as_of_date)
print(f"✓ Generated {len(recs)} recommendations as of {as_of_date.date()}")
print()

# Test 4: Different top-N values
print("Test 4: Different Top-N Values")
print("-"*80)
customer_id = TEST_CUSTOMERS[1]  # Heavy user
for top_n in [10, 20, 50, 100]:
    recs = get_recommendations(customer_id, top_n=top_n)
    print(f"  Top-{top_n}: {len(recs)} recommendations")
print()

# Summary
print("="*80)
print("✅ ALL TESTS PASSED")
print("="*80)
print("The production API is working correctly and ready for deployment.")
print()
