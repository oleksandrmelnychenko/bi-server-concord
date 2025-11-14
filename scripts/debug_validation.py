#!/usr/bin/env python3

import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33
from api.db_pool import get_connection

def debug_customer(customer_id: int, as_of_date: str):
    """Debug a single customer to understand the recommendation/validation logic"""

    conn = get_connection()
    cursor = conn.cursor(as_dict=True)

    print(f"\n{'='*80}")
    print(f"DEBUG CUSTOMER {customer_id}")
    print(f"{'='*80}")
    print(f"As of date: {as_of_date}")

    # Get recommendations
    recommender = ImprovedHybridRecommenderV33(conn=conn, use_cache=False)
    recommendations = recommender.get_recommendations(
        customer_id=customer_id,
        as_of_date=as_of_date,
        top_n=10,
        include_discovery=False
    )

    print(f"\nGenerated {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. Product {rec['product_id']} -> Agreement {rec.get('agreement_id')} (score: {rec.get('score', 0):.3f})")

    # Get future purchases by agreement
    future_date = (datetime.strptime(as_of_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

    query = f"""
    SELECT
        ca.ID as AgreementID,
        oi.ProductID,
        COUNT(*) as PurchaseCount
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
          AND o.Created >= '{as_of_date}'
          AND o.Created < '{future_date}'
          AND oi.ProductID IS NOT NULL
    GROUP BY ca.ID, oi.ProductID
    ORDER BY ca.ID, COUNT(*) DESC
    """

    cursor.execute(query)
    future_purchases = list(cursor)

    purchases_by_agreement = defaultdict(list)
    for row in future_purchases:
        purchases_by_agreement[row['AgreementID']].append({
            'product_id': row['ProductID'],
            'count': row['PurchaseCount']
        })

    print(f"\nFuture purchases (next 30 days) by agreement:")
    for agreement_id, products in purchases_by_agreement.items():
        print(f"  Agreement {agreement_id}:")
        for p in products[:5]:
            print(f"    - Product {p['product_id']} ({p['count']} orders)")

    # Check matches
    print(f"\nChecking matches:")
    hits = 0
    for rec in recommendations:
        product_id = rec['product_id']
        agreement_id = rec.get('agreement_id')

        if agreement_id and agreement_id in purchases_by_agreement:
            future_products = [p['product_id'] for p in purchases_by_agreement[agreement_id]]
            if product_id in future_products:
                print(f"  ✓ HIT: Product {product_id} for Agreement {agreement_id}")
                hits += 1
            else:
                print(f"  ✗ MISS: Product {product_id} for Agreement {agreement_id} (recommended but not purchased)")
        else:
            print(f"  ✗ MISS: Product {product_id} for Agreement {agreement_id} (agreement had no future purchases)")

    precision = hits / len(recommendations) if recommendations else 0
    print(f"\nPrecision: {hits}/{len(recommendations)} = {precision*100:.1f}%")
    print(f"{'='*80}\n")

    cursor.close()
    conn.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--customer-id', type=int, required=True)
    parser.add_argument('--as-of-date', type=str, default='2024-06-01')

    args = parser.parse_args()

    debug_customer(args.customer_id, args.as_of_date)
