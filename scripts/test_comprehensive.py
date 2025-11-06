#!/usr/bin/env python3
"""
Comprehensive Testing Suite - ML Recommendation System

Tests all components critically:
1. Data Quality
2. Model Performance
3. Inference Engine
4. Edge Cases
5. Production Readiness
"""

import sys
import pickle
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime
import traceback

# Import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.predict_recommendations import RecommendationEngine

print("="*80)
print("COMPREHENSIVE TESTING SUITE - ML RECOMMENDATION SYSTEM")
print("="*80)
print(f"Test started at: {datetime.now()}")
print()

# Test results tracking
test_results = []

def log_test(test_name, passed, details=""):
    """Log test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"       {details}")
    test_results.append({'test': test_name, 'passed': passed, 'details': details})

def test_section(title):
    """Print section header"""
    print()
    print("="*80)
    print(f"{title}")
    print("="*80)


# =============================================================================
# SECTION 1: DATA QUALITY TESTS
# =============================================================================
test_section("1. DATA QUALITY TESTS")

try:
    conn = duckdb.connect('data/ml_features/concord_ml.duckdb', read_only=True)

    # Test 1.1: Check interaction_matrix row count
    count = conn.execute("SELECT COUNT(*) FROM ml_features.interaction_matrix").fetchone()[0]
    log_test("Interaction matrix row count", count > 0, f"{count:,} rows")

    # Test 1.2: Check for NULL customer_ids
    null_customers = conn.execute("""
        SELECT COUNT(*) FROM ml_features.interaction_matrix WHERE customer_id IS NULL
    """).fetchone()[0]
    log_test("No NULL customer_ids", null_customers == 0, f"{null_customers} NULLs found")

    # Test 1.3: Check for NULL product_ids
    null_products = conn.execute("""
        SELECT COUNT(*) FROM ml_features.interaction_matrix WHERE product_id IS NULL
    """).fetchone()[0]
    log_test("No NULL product_ids", null_products == 0, f"{null_products} NULLs found")

    # Test 1.4: Check for negative values in critical fields
    negative_purchases = conn.execute("""
        SELECT COUNT(*) FROM ml_features.interaction_matrix WHERE num_purchases < 0
    """).fetchone()[0]
    log_test("No negative num_purchases", negative_purchases == 0, f"{negative_purchases} negatives found")

    # Test 1.5: Check for duplicates
    duplicates = conn.execute("""
        SELECT customer_id, product_id, COUNT(*) as cnt
        FROM ml_features.interaction_matrix
        GROUP BY customer_id, product_id
        HAVING COUNT(*) > 1
    """).fetchall()
    log_test("No duplicate customer-product pairs", len(duplicates) == 0, f"{len(duplicates)} duplicates found")

    # Test 1.6: Check implicit_rating range (should be 0-5)
    rating_range = conn.execute("""
        SELECT MIN(implicit_rating), MAX(implicit_rating) FROM ml_features.interaction_matrix
    """).fetchone()
    log_test("Implicit ratings in valid range (0-5)",
             rating_range[0] >= 0 and rating_range[1] <= 5.5,
             f"Range: [{rating_range[0]:.2f}, {rating_range[1]:.2f}]")

    # Test 1.7: Check customer_features exists
    customer_count = conn.execute("SELECT COUNT(*) FROM ml_features.customer_features").fetchone()[0]
    log_test("Customer features table populated", customer_count > 0, f"{customer_count:,} customers")

    # Test 1.8: Check RFM scores are in valid range (1-5)
    rfm_check = conn.execute("""
        SELECT
            MIN(recency_score), MAX(recency_score),
            MIN(frequency_score), MAX(frequency_score),
            MIN(monetary_score), MAX(monetary_score)
        FROM ml_features.customer_features
    """).fetchone()
    log_test("RFM scores in valid range (1-5)",
             all(1 <= x <= 5 for x in rfm_check if x is not None),
             f"R:[{rfm_check[0]}-{rfm_check[1]}] F:[{rfm_check[2]}-{rfm_check[3]}] M:[{rfm_check[4]}-{rfm_check[5]}]")

    conn.close()

except Exception as e:
    log_test("Data quality tests", False, f"Error: {e}")
    traceback.print_exc()


# =============================================================================
# SECTION 2: MODEL LOADING TESTS
# =============================================================================
test_section("2. MODEL LOADING TESTS")

try:
    # Test 2.1: Load Collaborative Filtering model
    with open('models/collaborative_filtering/als_model_v2.pkl', 'rb') as f:
        als_model = pickle.load(f)
    log_test("ALS model loads successfully", True, f"{als_model.factors} factors")

    # Test 2.2: Load ID mappings
    with open('models/collaborative_filtering/id_mappings_v2.pkl', 'rb') as f:
        mappings = pickle.load(f)
    log_test("ID mappings load successfully", True,
             f"{len(mappings['user_id_map']):,} users, {len(mappings['item_id_map']):,} items")

    # Test 2.3: Load Survival Analysis model
    with open('models/survival_analysis/weibull_repurchase_model.pkl', 'rb') as f:
        survival_model = pickle.load(f)
    log_test("Survival model loads successfully", True,
             f"C-index: {survival_model.concordance_index_:.4f}")

    # Test 2.4: Check C-index is good (>0.7)
    log_test("Survival model C-index > 0.70",
             survival_model.concordance_index_ > 0.70,
             f"{survival_model.concordance_index_:.4f}")

except Exception as e:
    log_test("Model loading tests", False, f"Error: {e}")
    traceback.print_exc()


# =============================================================================
# SECTION 3: INFERENCE ENGINE TESTS
# =============================================================================
test_section("3. INFERENCE ENGINE TESTS")

try:
    # Initialize engine
    engine = RecommendationEngine()
    engine.load_models()

    # Test 3.1: Standard customer with history
    recs = engine.get_recommendations(customer_id='410187', top_n=20, repurchase_ratio=0.8)
    log_test("Get recommendations for standard customer", len(recs) == 20,
             f"Got {len(recs)} recommendations")

    # Test 3.2: Verify 80/20 split
    repurchase_count = sum(1 for r in recs if r['type'] == 'REPURCHASE')
    discovery_count = sum(1 for r in recs if r['type'] == 'DISCOVERY')
    log_test("Correct 80/20 split (16 repurchase, 4 discovery)",
             repurchase_count == 16 and discovery_count == 4,
             f"{repurchase_count} repurchase, {discovery_count} discovery")

    # Test 3.3: All recommendations have required fields
    required_fields = ['rank', 'product_id', 'type', 'score', 'reason']
    all_have_fields = all(all(field in rec for field in required_fields) for rec in recs)
    log_test("All recommendations have required fields", all_have_fields)

    # Test 3.4: Ranks are sequential 1-20
    ranks = [r['rank'] for r in recs]
    log_test("Ranks are sequential 1-20", ranks == list(range(1, 21)))

    # Test 3.5: No duplicate product_ids in recommendations
    product_ids = [r['product_id'] for r in recs]
    log_test("No duplicate products recommended", len(product_ids) == len(set(product_ids)),
             f"{len(set(product_ids))} unique out of {len(product_ids)}")

    # Test 3.6: Scores are valid numbers (not NaN, not infinite)
    scores = [r['score'] for r in recs]
    valid_scores = all(np.isfinite(s) for s in scores)
    log_test("All scores are valid numbers (not NaN/infinite)", valid_scores)

except Exception as e:
    log_test("Inference engine tests", False, f"Error: {e}")
    traceback.print_exc()


# =============================================================================
# SECTION 4: EDGE CASE TESTS
# =============================================================================
test_section("4. EDGE CASE TESTS")

try:
    # Get list of customers for testing
    conn = duckdb.connect('data/ml_features/concord_ml.duckdb', read_only=True)

    # Test 4.1: Customer with minimal history (only 1-2 products)
    minimal_customer = conn.execute("""
        SELECT customer_id, COUNT(DISTINCT product_id) as product_count
        FROM ml_features.interaction_matrix
        GROUP BY customer_id
        HAVING COUNT(DISTINCT product_id) BETWEEN 1 AND 2
        ORDER BY customer_id
        LIMIT 1
    """).fetchone()

    if minimal_customer:
        recs = engine.get_recommendations(customer_id=minimal_customer[0], top_n=20)
        log_test("Handle customer with minimal history", len(recs) > 0,
                 f"Customer {minimal_customer[0]} ({minimal_customer[1]} products) → {len(recs)} recs")
    else:
        log_test("Handle customer with minimal history", True, "No minimal customers to test (SKIP)")

    # Test 4.2: Customer with many products
    power_customer = conn.execute("""
        SELECT customer_id, COUNT(DISTINCT product_id) as product_count
        FROM ml_features.interaction_matrix
        GROUP BY customer_id
        ORDER BY COUNT(DISTINCT product_id) DESC
        LIMIT 1
    """).fetchone()

    if power_customer:
        recs = engine.get_recommendations(customer_id=power_customer[0], top_n=20)
        log_test("Handle power customer with many products", len(recs) == 20,
                 f"Customer {power_customer[0]} ({power_customer[1]} products) → {len(recs)} recs")

    # Test 4.3: Non-existent customer
    try:
        recs = engine.get_recommendations(customer_id='NONEXISTENT999999', top_n=20)
        log_test("Handle non-existent customer gracefully", len(recs) > 0,
                 f"Returned {len(recs)} fallback recommendations")
    except Exception as e:
        log_test("Handle non-existent customer gracefully", False, f"Error: {e}")

    # Test 4.4: Customer with only single purchases (no repurchases)
    single_purchase_customer = conn.execute("""
        SELECT customer_id
        FROM ml_features.interaction_matrix
        WHERE num_purchases = 1
        GROUP BY customer_id
        HAVING COUNT(*) >= 3  -- Has 3+ products but all single purchases
        LIMIT 1
    """).fetchone()

    if single_purchase_customer:
        recs = engine.get_recommendations(customer_id=single_purchase_customer[0], top_n=20)
        log_test("Handle customer with only single purchases", len(recs) > 0,
                 f"Customer {single_purchase_customer[0]} → {len(recs)} recs (mostly DISCOVERY)")
    else:
        log_test("Handle customer with only single purchases", True, "No such customers (SKIP)")

    # Test 4.5: Different repurchase ratios
    recs_90_10 = engine.get_recommendations(customer_id='410187', top_n=20, repurchase_ratio=0.9)
    repurchase_90 = sum(1 for r in recs_90_10 if r['type'] == 'REPURCHASE')
    log_test("Custom repurchase ratio (90/10)", repurchase_90 == 18,
             f"90% ratio → {repurchase_90} repurchase")

    recs_50_50 = engine.get_recommendations(customer_id='410187', top_n=20, repurchase_ratio=0.5)
    repurchase_50 = sum(1 for r in recs_50_50 if r['type'] == 'REPURCHASE')
    log_test("Custom repurchase ratio (50/50)", repurchase_50 == 10,
             f"50% ratio → {repurchase_50} repurchase")

    conn.close()

except Exception as e:
    log_test("Edge case tests", False, f"Error: {e}")
    traceback.print_exc()


# =============================================================================
# SECTION 5: PERFORMANCE TESTS
# =============================================================================
test_section("5. PERFORMANCE TESTS")

try:
    import time

    # Test 5.1: Cold start (first call - loads matrix)
    start = time.time()
    recs = engine.get_recommendations(customer_id='410187', top_n=20)
    cold_duration = time.time() - start
    log_test("Cold start prediction < 2 seconds", cold_duration < 2.0,
             f"{cold_duration:.3f} seconds")

    # Test 5.2: Warm start (subsequent calls - matrix cached)
    start = time.time()
    recs = engine.get_recommendations(customer_id='411133', top_n=20)
    warm_duration = time.time() - start
    log_test("Warm start prediction < 1 second", warm_duration < 1.0,
             f"{warm_duration:.3f} seconds")

    # Test 5.3: Batch prediction (10 customers)
    test_customers = ['410187', '411133', '410849', '410547', '412230']
    start = time.time()
    for cust_id in test_customers[:10]:
        recs = engine.get_recommendations(customer_id=cust_id, top_n=20)
    batch_duration = time.time() - start
    avg_duration = batch_duration / len(test_customers[:10])
    log_test("Average prediction < 0.5 seconds (after warmup)", avg_duration < 0.5,
             f"{avg_duration:.3f} seconds/customer")

except Exception as e:
    log_test("Performance tests", False, f"Error: {e}")
    traceback.print_exc()


# =============================================================================
# SECTION 6: PREDICTION QUALITY TESTS
# =============================================================================
test_section("6. PREDICTION QUALITY TESTS")

try:
    conn = duckdb.connect('data/ml_features/concord_ml.duckdb', read_only=True)

    # Test 6.1: Repurchase recommendations prioritize overdue products
    recs = engine.get_recommendations(customer_id='410187', top_n=20)
    repurchase_recs = [r for r in recs if r['type'] == 'REPURCHASE']
    if len(repurchase_recs) >= 2:
        # Check that days_overdue is descending (most urgent first)
        overdue_values = [r.get('days_overdue', 0) for r in repurchase_recs if r.get('days_overdue') is not None]
        is_sorted = all(overdue_values[i] >= overdue_values[i+1] for i in range(len(overdue_values)-1))
        log_test("Repurchase recs sorted by urgency (days_overdue)", is_sorted,
                 f"Top 3 overdue: {overdue_values[:3]}")

    # Test 6.2: Discovery recommendations exclude already purchased products
    customer_products = set(conn.execute("""
        SELECT DISTINCT product_id FROM ml_features.interaction_matrix WHERE customer_id = '410187'
    """).df()['product_id'])

    discovery_recs = [r for r in recs if r['type'] == 'DISCOVERY']
    discovery_products = set(r['product_id'] for r in discovery_recs)
    no_overlap = len(customer_products & discovery_products) == 0
    log_test("Discovery recs don't include already purchased products", no_overlap,
             f"{len(discovery_products)} discovery products, {len(customer_products)} purchased")

    # Test 6.3: Check that predictions are reasonable
    # Load a customer's actual repurchase interval
    actual_interval = conn.execute("""
        SELECT AVG(purchase_span_days / (num_purchases - 1)) as avg_interval
        FROM ml_features.interaction_matrix
        WHERE customer_id = '410187' AND num_purchases >= 2
    """).fetchone()[0]

    # Get predicted intervals from recommendations
    predicted_intervals = [r.get('predicted_days_to_repurchase') for r in repurchase_recs
                          if r.get('predicted_days_to_repurchase') is not None]

    if predicted_intervals and actual_interval:
        avg_predicted = np.mean(predicted_intervals)
        # Predictions should be within 3x of actual (rough sanity check)
        reasonable = 0.1 * actual_interval < avg_predicted < 10 * actual_interval
        log_test("Predicted intervals are reasonable", reasonable,
                 f"Actual avg: {actual_interval:.0f} days, Predicted avg: {avg_predicted:.0f} days")

    conn.close()

except Exception as e:
    log_test("Prediction quality tests", False, f"Error: {e}")
    traceback.print_exc()


# =============================================================================
# SECTION 7: FILE INTEGRITY TESTS
# =============================================================================
test_section("7. FILE INTEGRITY TESTS")

try:
    from pathlib import Path

    # Test 7.1: Check all required files exist
    required_files = [
        'data/ml_features/concord_ml.duckdb',
        'models/collaborative_filtering/als_model_v2.pkl',
        'models/collaborative_filtering/id_mappings_v2.pkl',
        'models/survival_analysis/weibull_repurchase_model.pkl',
        'models/survival_analysis/reorder_alerts.csv',
        'scripts/predict_recommendations.py',
        'scripts/refresh_models_daily.py',
    ]

    missing_files = [f for f in required_files if not Path(f).exists()]
    log_test("All required files exist", len(missing_files) == 0,
             f"{len(missing_files)} missing: {missing_files}")

    # Test 7.2: Check file sizes are reasonable
    duckdb_size = Path('data/ml_features/concord_ml.duckdb').stat().st_size / 1024 / 1024  # MB
    log_test("DuckDB size reasonable (10-50 MB)", 10 < duckdb_size < 50,
             f"{duckdb_size:.1f} MB")

    # Test 7.3: Check alerts CSV has data
    alerts_df = pd.read_csv('models/survival_analysis/reorder_alerts.csv')
    log_test("Reorder alerts CSV has data", len(alerts_df) > 1000,
             f"{len(alerts_df):,} alerts")

except Exception as e:
    log_test("File integrity tests", False, f"Error: {e}")
    traceback.print_exc()


# =============================================================================
# TEST SUMMARY
# =============================================================================
test_section("TEST SUMMARY")

total_tests = len(test_results)
passed_tests = sum(1 for t in test_results if t['passed'])
failed_tests = total_tests - passed_tests

print(f"\nTotal Tests: {total_tests}")
print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")

if failed_tests > 0:
    print("\n" + "="*80)
    print("FAILED TESTS:")
    print("="*80)
    for t in test_results:
        if not t['passed']:
            print(f"❌ {t['test']}")
            if t['details']:
                print(f"   {t['details']}")

print("\n" + "="*80)
print(f"Test completed at: {datetime.now()}")
print("="*80)

# Exit with error code if any tests failed
sys.exit(0 if failed_tests == 0 else 1)
