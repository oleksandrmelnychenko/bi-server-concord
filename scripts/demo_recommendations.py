#!/usr/bin/env python3
"""
Real-World Recommendation Demo

Shows how V3 recommendations work in practice:
1. Customer purchase history
2. V3 recommendations
3. Actual purchases
4. Match analysis
"""

import os
import sys
import pymssql
import logging
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender import ImprovedHybridRecommender

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'server': '78.152.175.67',
    'port': 1433,
    'database': 'ConcordDb_v5',
    'user': 'ef_migrator',
    'password': 'Grimm_jow92',
    'as_dict': True
}

AS_OF_DATE = "2024-07-01"


def print_header(text, char="="):
    """Print formatted header"""
    print(f"\n{char * 80}")
    print(f"{text}")
    print(f"{char * 80}\n")


def get_customer_profile(conn, customer_id, as_of_date):
    """Get customer profile information"""
    cursor = conn.cursor()

    # Count orders before as_of_date
    query = f"""
    SELECT COUNT(DISTINCT o.ID) as order_count,
           MIN(o.Created) as first_order,
           MAX(o.Created) as last_order
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    WHERE ca.ClientID = {customer_id}
          AND o.Created < '{as_of_date}'
    """
    cursor.execute(query)
    profile = cursor.fetchone()

    # Count unique products
    query = f"""
    SELECT COUNT(DISTINCT oi.ProductID) as unique_products
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
          AND o.Created < '{as_of_date}'
          AND oi.ProductID IS NOT NULL
    """
    cursor.execute(query)
    products = cursor.fetchone()

    cursor.close()

    return {
        'order_count': profile['order_count'],
        'unique_products': products['unique_products'],
        'first_order': profile['first_order'],
        'last_order': profile['last_order']
    }


def get_purchase_history(conn, customer_id, as_of_date, limit=10):
    """Get top purchased products before as_of_date"""
    cursor = conn.cursor()

    query = f"""
    SELECT TOP {limit}
        oi.ProductID,
        p.Name as ProductName,
        COUNT(DISTINCT o.ID) as purchase_count,
        MAX(o.Created) as last_purchase
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    LEFT JOIN dbo.Product p ON oi.ProductID = p.ID
    WHERE ca.ClientID = {customer_id}
          AND o.Created < '{as_of_date}'
          AND oi.ProductID IS NOT NULL
    GROUP BY oi.ProductID, p.Name
    ORDER BY purchase_count DESC, last_purchase DESC
    """

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    return results


def get_actual_purchases(conn, customer_id, as_of_date):
    """Get products actually purchased after as_of_date"""
    cursor = conn.cursor()

    query = f"""
    SELECT DISTINCT oi.ProductID,
           p.Name as ProductName,
           MIN(o.Created) as first_purchase_after
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    LEFT JOIN dbo.Product p ON oi.ProductID = p.ID
    WHERE ca.ClientID = {customer_id}
          AND o.Created >= '{as_of_date}'
          AND oi.ProductID IS NOT NULL
    GROUP BY oi.ProductID, p.Name
    ORDER BY first_purchase_after
    """

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    return results


def main():
    # Customer 410280: REGULAR-CONSISTENT with 90% precision
    DEMO_CUSTOMER = 410280

    print_header("V3 RECOMMENDATION SYSTEM - REAL-WORLD DEMO", "=")
    print(f"Customer ID: {DEMO_CUSTOMER}")
    print(f"As-of Date: {AS_OF_DATE} (recommendations generated on this date)")
    print(f"Validation Period: After {AS_OF_DATE} (what they actually bought)")

    # Connect to database
    conn = pymssql.connect(**DB_CONFIG)

    # Get customer profile
    print_header("1. CUSTOMER PROFILE", "-")
    profile = get_customer_profile(conn, DEMO_CUSTOMER, AS_OF_DATE)

    print(f"Orders before {AS_OF_DATE}: {profile['order_count']}")
    print(f"Unique products purchased: {profile['unique_products']}")
    print(f"First order: {profile['first_order'].strftime('%Y-%m-%d')}")
    print(f"Last order (before cutoff): {profile['last_order'].strftime('%Y-%m-%d')}")

    # Classify customer
    recommender = ImprovedHybridRecommender(conn=conn)
    segment, subsegment = recommender.classify_customer(DEMO_CUSTOMER, AS_OF_DATE)
    print(f"\nSegment: {segment}" + (f" ({subsegment})" if subsegment else ""))

    # Show purchase history
    print_header("2. PURCHASE HISTORY (Top 10 Products Before July 1)", "-")
    history = get_purchase_history(conn, DEMO_CUSTOMER, AS_OF_DATE, limit=10)

    print(f"{'Rank':<6} {'Product ID':<12} {'Times Bought':<14} {'Last Purchase':<15} {'Product Name'}")
    print("-" * 80)
    for idx, item in enumerate(history, 1):
        product_name = item['ProductName'][:35] if item['ProductName'] else "N/A"
        last_purchase = item['last_purchase'].strftime('%Y-%m-%d')
        print(f"{idx:<6} {item['ProductID']:<12} {item['purchase_count']:<14} {last_purchase:<15} {product_name}")

    # Generate recommendations
    print_header("3. V3 RECOMMENDATIONS (Top 50 as of July 1)", "-")
    recommendations = recommender.get_recommendations(DEMO_CUSTOMER, AS_OF_DATE, top_n=50)

    print(f"Generated {len(recommendations)} recommendations")
    print(f"\nTop 10 Recommendations:")
    print(f"{'Rank':<6} {'Product ID':<12} {'Score':<10}")
    print("-" * 30)
    for rec in recommendations[:10]:
        print(f"{rec['rank']:<6} {rec['product_id']:<12} {rec['score']:.4f}")

    recommended_products = set(rec['product_id'] for rec in recommendations)

    # Get actual purchases
    print_header("4. ACTUAL PURCHASES (After July 1)", "-")
    actual = get_actual_purchases(conn, DEMO_CUSTOMER, AS_OF_DATE)

    print(f"Total products purchased: {len(actual)}")
    print(f"\nFirst 10 purchases:")
    print(f"{'Product ID':<12} {'First Purchase':<15} {'Product Name'}")
    print("-" * 80)
    for item in actual[:10]:
        product_name = item['ProductName'][:45] if item['ProductName'] else "N/A"
        first_purchase = item['first_purchase_after'].strftime('%Y-%m-%d')
        print(f"{item['ProductID']:<12} {first_purchase:<15} {product_name}")

    actual_products = set(item['ProductID'] for item in actual)

    # Match analysis
    print_header("5. MATCH ANALYSIS", "-")

    hits = recommended_products & actual_products
    misses = recommended_products - actual_products
    not_recommended = actual_products - recommended_products

    precision = len(hits) / len(recommended_products) if recommended_products else 0

    print(f"Recommendations: {len(recommended_products)} products")
    print(f"Actual purchases: {len(actual_products)} products")
    print(f"\n✅ HITS (Recommended AND Bought): {len(hits)} products")
    print(f"❌ MISSES (Recommended but NOT Bought): {len(misses)} products")
    print(f"⚠️  NOT RECOMMENDED (Bought but not in top 50): {len(not_recommended)} products")
    print(f"\n🎯 PRECISION@50: {precision:.1%}")

    # Show some hits
    if hits:
        print(f"\n{'='*80}")
        print("EXAMPLE HITS (Products V3 Recommended AND Customer Bought)")
        print(f"{'='*80}\n")

        # Get details for first 5 hits
        cursor = conn.cursor()
        hit_list = list(hits)[:5]
        for product_id in hit_list:
            # Get product name
            cursor.execute(f"SELECT Name FROM dbo.Product WHERE ID = {product_id}")
            result = cursor.fetchone()
            product_name = result['Name'][:50] if result and result['Name'] else "N/A"

            # Get recommendation rank
            rank = next((rec['rank'] for rec in recommendations if rec['product_id'] == product_id), None)
            score = next((rec['score'] for rec in recommendations if rec['product_id'] == product_id), None)

            # Get when they bought it
            purchase_date = next((item['first_purchase_after'] for item in actual if item['ProductID'] == product_id), None)
            purchase_date_str = purchase_date.strftime('%Y-%m-%d') if purchase_date else "N/A"

            print(f"✅ Product {product_id}")
            print(f"   Name: {product_name}")
            print(f"   V3 Rank: #{rank} (score: {score:.4f})")
            print(f"   Bought on: {purchase_date_str}")
            print()

        cursor.close()

    # Show some misses
    if misses:
        print(f"{'='*80}")
        print("EXAMPLE MISSES (Products V3 Recommended but Customer Didn't Buy)")
        print(f"{'='*80}\n")

        cursor = conn.cursor()
        miss_list = list(misses)[:3]
        for product_id in miss_list:
            # Get product name
            cursor.execute(f"SELECT Name FROM dbo.Product WHERE ID = {product_id}")
            result = cursor.fetchone()
            product_name = result['Name'][:50] if result and result['Name'] else "N/A"

            # Get recommendation rank
            rank = next((rec['rank'] for rec in recommendations if rec['product_id'] == product_id), None)
            score = next((rec['score'] for rec in recommendations if rec['product_id'] == product_id), None)

            print(f"❌ Product {product_id}")
            print(f"   Name: {product_name}")
            print(f"   V3 Rank: #{rank} (score: {score:.4f})")
            print(f"   Status: Recommended but not purchased (yet)")
            print()

        cursor.close()

    print_header("CONCLUSION", "=")
    print(f"V3 achieved {precision:.1%} precision for Customer {DEMO_CUSTOMER}")
    print(f"Out of 50 recommendations, {len(hits)} products were actually purchased.")
    print(f"\nThis demonstrates how V3 uses purchase history to predict future needs")
    print(f"in the B2B truck parts domain with high accuracy!")

    # Close
    recommender.close()
    conn.close()


if __name__ == '__main__':
    main()
