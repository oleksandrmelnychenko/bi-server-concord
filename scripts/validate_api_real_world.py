#!/usr/bin/env python3
"""
Real-World API Validation Test

Tests recommendation API on 10 diverse customers to validate
whether the 59.3% precision claim generalizes beyond the 13-customer test set.

Approach:
1. Select 10 customers (3 heavy, 5 regular, 2 light)
2. Query their purchase history from database
3. Split: training (before July 1, 2024) vs validation (after)
4. Call /recommend API
5. Calculate precision@50
6. Generate detailed report

Expected time: 1-1.5 hours
"""

import os
import sys
import json
import requests
import logging
import pandas as pd
import pymssql
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000/recommend"
AS_OF_DATE = datetime(2024, 7, 1)
TOP_N = 50

# Database connection
def connect_db():
    """Connect to MSSQL database"""
    return pymssql.connect(
        server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
        port=int(os.environ.get('MSSQL_PORT', '1433')),
        database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
        user=os.environ.get('MSSQL_USER', 'ef_migrator'),
        password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
    )


def select_test_customers(conn, n_heavy=3, n_regular=5, n_light=2) -> List[int]:
    """
    Select diverse test customers from database

    Criteria:
    - Heavy: 500+ purchases
    - Regular: 100-500 purchases
    - Light: 20-100 purchases
    - Must have purchases before AND after July 1, 2024
    """
    logger.info("Selecting test customers...")

    query = """
    SELECT
        ca.ClientID,
        COUNT(DISTINCT o.ID) as total_orders,
        COUNT(DISTINCT oi.ProductID) as unique_products,
        MIN(o.Created) as first_order,
        MAX(o.Created) as last_order,
        SUM(CASE WHEN o.Created < '2024-07-01' THEN 1 ELSE 0 END) as orders_before,
        SUM(CASE WHEN o.Created >= '2024-07-01' THEN 1 ELSE 0 END) as orders_after
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE oi.ProductID IS NOT NULL
    GROUP BY ca.ClientID
    HAVING
        SUM(CASE WHEN o.Created < '2024-07-01' THEN 1 ELSE 0 END) >= 10
        AND SUM(CASE WHEN o.Created >= '2024-07-01' THEN 1 ELSE 0 END) >= 5
    ORDER BY total_orders DESC
    """

    df = pd.read_sql(query, conn)

    # Segment customers
    df['segment'] = pd.cut(
        df['total_orders'],
        bins=[0, 100, 500, float('inf')],
        labels=['light', 'regular', 'heavy']
    )

    # Select customers
    selected = []

    # Heavy users (top performers)
    heavy_candidates = df[df['segment'] == 'heavy'].head(10)
    selected.extend(heavy_candidates['ClientID'].sample(n=min(n_heavy, len(heavy_candidates))).tolist())

    # Regular users (middle performers)
    regular_candidates = df[df['segment'] == 'regular'].head(20)
    selected.extend(regular_candidates['ClientID'].sample(n=min(n_regular, len(regular_candidates))).tolist())

    # Light users (challenging cases)
    light_candidates = df[df['segment'] == 'light'].head(10)
    selected.extend(light_candidates['ClientID'].sample(n=min(n_light, len(light_candidates))).tolist())

    logger.info(f"  Selected {len(selected)} customers:")
    logger.info(f"    Heavy: {n_heavy}, Regular: {n_regular}, Light: {n_light}")

    return selected


def get_customer_info(conn, customer_id: int) -> Dict:
    """Get customer segment and purchase statistics"""
    query = f"""
    SELECT
        COUNT(DISTINCT o.ID) as total_orders,
        COUNT(DISTINCT oi.ProductID) as unique_products,
        MIN(o.Created) as first_order,
        MAX(o.Created) as last_order
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND oi.ProductID IS NOT NULL
    """

    df = pd.read_sql(query, conn)

    if len(df) == 0:
        return None

    row = df.iloc[0]
    total_orders = row['total_orders']

    # Determine segment
    if total_orders >= 500:
        segment = 'HEAVY'
    elif total_orders >= 100:
        segment = 'REGULAR'
    else:
        segment = 'LIGHT'

    return {
        'customer_id': customer_id,
        'segment': segment,
        'total_orders': int(total_orders),
        'unique_products': int(row['unique_products']),
        'first_order': row['first_order'].strftime('%Y-%m-%d'),
        'last_order': row['last_order'].strftime('%Y-%m-%d')
    }


def get_validation_purchases(conn, customer_id: int) -> Tuple[List[int], List[str]]:
    """
    Get products purchased AFTER as_of_date (validation period)

    Returns: (product_ids, product_names)
    """
    query = f"""
    SELECT DISTINCT
        CAST(oi.ProductID AS INT) as product_id,
        p.Name as product_name
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    LEFT JOIN dbo.Product p ON oi.ProductID = p.ID
    WHERE ca.ClientID = {customer_id}
        AND o.Created >= '{AS_OF_DATE.strftime('%Y-%m-%d')}'
        AND oi.ProductID IS NOT NULL
    """

    df = pd.read_sql(query, conn)

    product_ids = df['product_id'].tolist()
    product_names = dict(zip(df['product_id'], df['product_name']))

    return product_ids, product_names


def get_product_names(conn, product_ids: List[int]) -> Dict[int, str]:
    """Get product names for a list of product IDs"""
    if not product_ids:
        return {}

    ids_str = ','.join(map(str, product_ids))
    query = f"""
    SELECT
        CAST(ID AS INT) as product_id,
        Name as product_name
    FROM dbo.Product
    WHERE ID IN ({ids_str})
    """

    df = pd.read_sql(query, conn)
    return dict(zip(df['product_id'], df['product_name']))


def call_recommend_api(customer_id: int, top_n: int = 50) -> Dict:
    """Call /recommend API endpoint"""
    payload = {
        "customer_id": customer_id,
        "top_n": top_n,
        "as_of_date": AS_OF_DATE.strftime('%Y-%m-%d'),
        "use_cache": False  # Force fresh recommendations
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"  API call failed: {e}")
        return None


def calculate_precision(recommendations: List[Dict], validation_products: List[int]) -> Tuple[float, int, List[Dict]]:
    """
    Calculate precision@50

    Returns: (precision, hits, detailed_recs)
    """
    validation_set = set(validation_products)

    detailed_recs = []
    hits = 0

    for rec in recommendations[:TOP_N]:
        product_id = int(rec['product_id'])
        is_hit = product_id in validation_set

        if is_hit:
            hits += 1

        detailed_recs.append({
            'rank': rec['rank'],
            'product_id': product_id,
            'score': rec['score'],
            'reason': rec['reason'],
            'hit': is_hit
        })

    precision = hits / len(recommendations[:TOP_N]) if recommendations else 0.0

    return precision, hits, detailed_recs


def test_customer(conn, customer_id: int) -> Dict:
    """Run full validation test for one customer"""
    logger.info(f"\nTesting Customer {customer_id}")
    logger.info("=" * 80)

    # Get customer info
    info = get_customer_info(conn, customer_id)
    if not info:
        logger.error(f"  No data for customer {customer_id}")
        return None

    logger.info(f"  Segment: {info['segment']}")
    logger.info(f"  Total Orders: {info['total_orders']}")
    logger.info(f"  Unique Products: {info['unique_products']}")
    logger.info(f"  Order History: {info['first_order']} to {info['last_order']}")

    # Get validation purchases
    validation_products, product_names = get_validation_purchases(conn, customer_id)
    logger.info(f"  Validation Products (after {AS_OF_DATE.strftime('%Y-%m-%d')}): {len(validation_products)}")

    if len(validation_products) == 0:
        logger.warning(f"  No validation purchases for customer {customer_id}, skipping")
        return None

    # Call API
    logger.info(f"  Calling API...")
    api_response = call_recommend_api(customer_id, TOP_N)

    if not api_response:
        logger.error(f"  API call failed")
        return None

    recommendations = api_response.get('recommendations', [])
    latency_ms = api_response.get('latency_ms', 0)

    logger.info(f"  API Latency: {latency_ms:.2f}ms")
    logger.info(f"  Recommendations Received: {len(recommendations)}")

    # Get product names for recommendations
    rec_product_ids = [int(r['product_id']) for r in recommendations[:TOP_N]]
    rec_product_names = get_product_names(conn, rec_product_ids)

    # Calculate precision
    precision, hits, detailed_recs = calculate_precision(recommendations, validation_products)

    logger.info(f"  Precision@50: {precision:.1%} ({hits}/{TOP_N} hits)")

    # Show top 10 recommendations
    logger.info(f"\n  Top 10 Recommendations:")
    for rec in detailed_recs[:10]:
        hit_status = "✅ PURCHASED" if rec['hit'] else "❌ Not purchased"
        product_name = rec_product_names.get(rec['product_id'], f"Product {rec['product_id']}")
        logger.info(f"    {rec['rank']:2d}. {product_name[:50]:50s} - {hit_status} (score: {rec['score']:.3f})")

    # Compile result
    result = {
        'customer_id': customer_id,
        'segment': info['segment'],
        'total_orders': info['total_orders'],
        'unique_products': info['unique_products'],
        'validation_products_count': len(validation_products),
        'precision': precision,
        'hits': hits,
        'total_recommendations': len(recommendations),
        'latency_ms': latency_ms,
        'top_recommendations': detailed_recs[:10],
        'product_names': rec_product_names
    }

    return result


def main():
    logger.info("="*80)
    logger.info("REAL-WORLD API VALIDATION TEST")
    logger.info("="*80)
    logger.info(f"\nGoal: Validate whether 59.3% precision generalizes beyond 13-customer test set")
    logger.info(f"Method: Test API on 10 diverse customers")
    logger.info(f"API: {API_URL}")
    logger.info(f"As-of-date: {AS_OF_DATE.strftime('%Y-%m-%d')}")
    logger.info(f"Top-N: {TOP_N}\n")

    # Connect to database
    conn = connect_db()

    try:
        # Select test customers
        test_customers = select_test_customers(conn, n_heavy=3, n_regular=5, n_light=2)

        if len(test_customers) < 10:
            logger.warning(f"Only found {len(test_customers)} customers (expected 10)")

        # Test each customer
        all_results = []

        for i, customer_id in enumerate(test_customers, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"CUSTOMER {i}/{len(test_customers)}")
            logger.info(f"{'='*80}")

            result = test_customer(conn, customer_id)

            if result:
                all_results.append(result)

        # Calculate aggregate statistics
        logger.info(f"\n{'='*80}")
        logger.info("AGGREGATE RESULTS")
        logger.info(f"{'='*80}\n")

        if not all_results:
            logger.error("No successful tests!")
            return

        precisions = [r['precision'] for r in all_results]
        avg_precision = sum(precisions) / len(precisions)
        std_precision = pd.Series(precisions).std()
        min_precision = min(precisions)
        max_precision = max(precisions)

        logger.info(f"Customers Tested: {len(all_results)}")
        logger.info(f"Average Precision@50: {avg_precision:.1%}")
        logger.info(f"Std Deviation: {std_precision:.1%}")
        logger.info(f"Min Precision: {min_precision:.1%}")
        logger.info(f"Max Precision: {max_precision:.1%}")
        logger.info(f"\nExpected (Phase E/I): 59.3%")
        logger.info(f"Actual: {avg_precision:.1%}")
        logger.info(f"Difference: {(avg_precision - 0.593) * 100:+.1f}pp")

        # Segment breakdown
        logger.info(f"\n{'='*80}")
        logger.info("SEGMENT BREAKDOWN")
        logger.info(f"{'='*80}\n")

        segments = defaultdict(list)
        for r in all_results:
            segments[r['segment']].append(r['precision'])

        for segment in ['HEAVY', 'REGULAR', 'LIGHT']:
            if segment in segments:
                seg_prec = segments[segment]
                avg = sum(seg_prec) / len(seg_prec)
                logger.info(f"{segment:8s}: {avg:.1%} (n={len(seg_prec)})")

        # Save results
        output_file = 'real_world_validation_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'meta': {
                    'test_date': datetime.now().isoformat(),
                    'api_url': API_URL,
                    'as_of_date': AS_OF_DATE.strftime('%Y-%m-%d'),
                    'top_n': TOP_N,
                    'customers_tested': len(all_results),
                    'expected_precision': 0.593,
                    'actual_precision': avg_precision,
                    'difference_pp': (avg_precision - 0.593) * 100
                },
                'aggregate_stats': {
                    'avg_precision': avg_precision,
                    'std_precision': std_precision,
                    'min_precision': min_precision,
                    'max_precision': max_precision
                },
                'segment_breakdown': {
                    segment: {
                        'avg_precision': sum(precs) / len(precs),
                        'count': len(precs)
                    }
                    for segment, precs in segments.items()
                },
                'customer_results': all_results
            }, f, indent=2, default=str)

        logger.info(f"\n💾 Results saved to: {output_file}")

        # Verdict
        logger.info(f"\n{'='*80}")
        logger.info("VERDICT")
        logger.info(f"{'='*80}\n")

        diff_pp = (avg_precision - 0.593) * 100

        if abs(diff_pp) <= 2:
            logger.info(f"✅ VALIDATION SUCCESS")
            logger.info(f"   {avg_precision:.1%} is within 2pp of expected 59.3%")
            logger.info(f"   The 59.3% precision claim is VALIDATED")
            logger.info(f"\n→ Next Steps: Expand to 100 customers, A/B test vs baseline")
        elif diff_pp > 2:
            logger.info(f"⚠️  BETTER THAN EXPECTED")
            logger.info(f"   {avg_precision:.1%} is {diff_pp:+.1f}pp higher than 59.3%")
            logger.info(f"   The 13-customer test set may have been pessimistic")
            logger.info(f"\n→ Next Steps: Validate on even more customers to confirm")
        else:
            logger.info(f"❌ VALIDATION FAILED")
            logger.info(f"   {avg_precision:.1%} is {diff_pp:.1f}pp lower than 59.3%")
            logger.info(f"   The 13-customer test set may have been cherry-picked")
            logger.info(f"\n→ Next Steps: Error analysis, investigate overfitting")

        return all_results

    finally:
        conn.close()


if __name__ == '__main__':
    results = main()
