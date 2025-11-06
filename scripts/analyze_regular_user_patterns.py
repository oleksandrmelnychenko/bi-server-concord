#!/usr/bin/env python3
"""
Phase 1.1: Analyze Regular User Failure Patterns

Compares successful regular users (90% precision) with failing ones (10%)
to identify what makes some predictable and others not.

Goal: Understand the root causes of regular user failures
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymssql
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Regular users from real-world validation
REGULAR_USERS = {
    'success': [
        {'id': 411726, 'precision': 0.90, 'label': 'SUCCESS'},
    ],
    'mediocre': [
        {'id': 410204, 'precision': 0.50, 'label': 'MEDIOCRE'},
    ],
    'failure': [
        {'id': 411317, 'precision': 0.10, 'label': 'FAILURE'},
        {'id': 414304, 'precision': 0.14, 'label': 'FAILURE'},
        {'id': 410962, 'precision': 0.16, 'label': 'FAILURE'},
    ]
}

AS_OF_DATE = datetime(2024, 7, 1)

def connect_db():
    """Connect to MSSQL database"""
    return pymssql.connect(
        server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
        port=int(os.environ.get('MSSQL_PORT', '1433')),
        database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
        user=os.environ.get('MSSQL_USER', 'ef_migrator'),
        password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
    )


def analyze_customer_basic_stats(conn, customer_id: int) -> Dict:
    """Get basic customer statistics"""
    query = f"""
    SELECT
        COUNT(DISTINCT o.ID) as total_orders,
        COUNT(DISTINCT oi.ProductID) as unique_products,
        COUNT(oi.ProductID) as total_items,
        MIN(o.Created) as first_order,
        MAX(o.Created) as last_order,
        SUM(CASE WHEN o.Created < '{AS_OF_DATE.strftime('%Y-%m-%d')}' THEN 1 ELSE 0 END) as orders_before,
        SUM(CASE WHEN o.Created >= '{AS_OF_DATE.strftime('%Y-%m-%d')}' THEN 1 ELSE 0 END) as orders_after,
        COUNT(DISTINCT CASE WHEN o.Created < '{AS_OF_DATE.strftime('%Y-%m-%d')}' THEN oi.ProductID END) as products_before,
        COUNT(DISTINCT CASE WHEN o.Created >= '{AS_OF_DATE.strftime('%Y-%m-%d')}' THEN oi.ProductID END) as products_after
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
    first_order = row['first_order']
    last_order = row['last_order']

    days_active = (last_order - first_order).days if first_order and last_order else 0

    return {
        'total_orders': int(row['total_orders']),
        'unique_products': int(row['unique_products']),
        'total_items': int(row['total_items']),
        'orders_before': int(row['orders_before']),
        'orders_after': int(row['orders_after']),
        'products_before': int(row['products_before']),
        'products_after': int(row['products_after']),
        'days_active': days_active,
        'first_order': first_order.strftime('%Y-%m-%d') if first_order else None,
        'last_order': last_order.strftime('%Y-%m-%d') if last_order else None,
        'avg_products_per_order': row['unique_products'] / row['total_orders'] if row['total_orders'] > 0 else 0,
        'product_diversity': row['unique_products'] / row['total_items'] if row['total_items'] > 0 else 0
    }


def analyze_purchase_frequency(conn, customer_id: int) -> Dict:
    """Analyze order frequency and patterns"""
    query = f"""
    SELECT
        o.Created as order_date
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    WHERE ca.ClientID = {customer_id}
        AND o.Created < '{AS_OF_DATE.strftime('%Y-%m-%d')}'
    ORDER BY o.Created
    """

    df = pd.read_sql(query, conn)

    if len(df) < 2:
        return {'avg_days_between_orders': None, 'std_days_between_orders': None, 'consistency_score': 0}

    # Calculate days between consecutive orders
    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df.sort_values('order_date')
    df['days_since_last'] = df['order_date'].diff().dt.days

    days_between = df['days_since_last'].dropna()

    if len(days_between) == 0:
        return {'avg_days_between_orders': None, 'std_days_between_orders': None, 'consistency_score': 0}

    avg_days = days_between.mean()
    std_days = days_between.std()

    # Consistency score: lower is better (more consistent)
    # CV (coefficient of variation) = std / mean
    consistency_score = 1 / (1 + (std_days / avg_days)) if avg_days > 0 else 0

    return {
        'avg_days_between_orders': avg_days,
        'std_days_between_orders': std_days,
        'consistency_score': consistency_score,  # 0-1, higher = more consistent
        'coefficient_of_variation': std_days / avg_days if avg_days > 0 else 0
    }


def analyze_product_repurchase_patterns(conn, customer_id: int) -> Dict:
    """Analyze how often products are repurchased"""
    query = f"""
    SELECT
        oi.ProductID,
        COUNT(DISTINCT o.ID) as purchase_count,
        MIN(o.Created) as first_purchase,
        MAX(o.Created) as last_purchase
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND o.Created < '{AS_OF_DATE.strftime('%Y-%m-%d')}'
        AND oi.ProductID IS NOT NULL
    GROUP BY oi.ProductID
    HAVING COUNT(DISTINCT o.ID) >= 2  -- Only repurchased products
    """

    df = pd.read_sql(query, conn)

    if len(df) == 0:
        return {
            'repurchased_products_count': 0,
            'repurchase_rate': 0,
            'avg_repurchases_per_product': 0,
            'frequent_products_count': 0
        }

    total_products_query = f"""
    SELECT COUNT(DISTINCT oi.ProductID) as total
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND o.Created < '{AS_OF_DATE.strftime('%Y-%m-%d')}'
        AND oi.ProductID IS NOT NULL
    """
    total_df = pd.read_sql(total_products_query, conn)
    total_products = int(total_df.iloc[0]['total'])

    return {
        'repurchased_products_count': len(df),
        'repurchase_rate': len(df) / total_products if total_products > 0 else 0,
        'avg_repurchases_per_product': df['purchase_count'].mean(),
        'frequent_products_count': len(df[df['purchase_count'] >= 5]),  # Bought 5+ times
        'max_repurchases': int(df['purchase_count'].max())
    }


def analyze_product_overlap(conn, customer_id: int) -> Dict:
    """Analyze overlap between training and validation products"""
    # Products before as_of_date
    before_query = f"""
    SELECT DISTINCT CAST(oi.ProductID AS INT) as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND o.Created < '{AS_OF_DATE.strftime('%Y-%m-%d')}'
        AND oi.ProductID IS NOT NULL
    """

    # Products after as_of_date
    after_query = f"""
    SELECT DISTINCT CAST(oi.ProductID AS INT) as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND o.Created >= '{AS_OF_DATE.strftime('%Y-%m-%d')}'
        AND oi.ProductID IS NOT NULL
    """

    before_df = pd.read_sql(before_query, conn)
    after_df = pd.read_sql(after_query, conn)

    products_before = set(before_df['product_id'].tolist())
    products_after = set(after_df['product_id'].tolist())

    overlap = products_before & products_after
    new_products = products_after - products_before

    return {
        'products_before_count': len(products_before),
        'products_after_count': len(products_after),
        'overlap_count': len(overlap),
        'overlap_rate': len(overlap) / len(products_after) if len(products_after) > 0 else 0,
        'new_products_count': len(new_products),
        'new_products_rate': len(new_products) / len(products_after) if len(products_after) > 0 else 0
    }


def analyze_customer(conn, customer_id: int, precision: float, label: str) -> Dict:
    """Comprehensive customer analysis"""
    logger.info(f"\nAnalyzing Customer {customer_id} ({label}, {precision:.0%} precision)")

    result = {
        'customer_id': customer_id,
        'precision': precision,
        'label': label
    }

    # Basic stats
    basic = analyze_customer_basic_stats(conn, customer_id)
    if not basic:
        logger.warning(f"  No data for customer {customer_id}")
        return None

    result.update(basic)
    logger.info(f"  Total Orders: {basic['total_orders']}, Unique Products: {basic['unique_products']}")

    # Frequency patterns
    frequency = analyze_purchase_frequency(conn, customer_id)
    result.update(frequency)
    if frequency['avg_days_between_orders']:
        logger.info(f"  Avg Days Between Orders: {frequency['avg_days_between_orders']:.1f} ± {frequency['std_days_between_orders']:.1f}")
        logger.info(f"  Consistency Score: {frequency['consistency_score']:.2f}")

    # Repurchase patterns
    repurchase = analyze_product_repurchase_patterns(conn, customer_id)
    result.update(repurchase)
    logger.info(f"  Repurchase Rate: {repurchase['repurchase_rate']:.1%}")
    logger.info(f"  Frequent Products (5+ purchases): {repurchase['frequent_products_count']}")

    # Product overlap
    overlap = analyze_product_overlap(conn, customer_id)
    result.update(overlap)
    logger.info(f"  Product Overlap (training→validation): {overlap['overlap_rate']:.1%}")
    logger.info(f"  New Products in Validation: {overlap['new_products_rate']:.1%}")

    return result


def compare_groups(success_results: List[Dict], failure_results: List[Dict]):
    """Compare success vs failure groups"""
    logger.info(f"\n{'='*80}")
    logger.info("GROUP COMPARISON: SUCCESS vs FAILURE")
    logger.info(f"{'='*80}\n")

    success_df = pd.DataFrame(success_results)
    failure_df = pd.DataFrame(failure_results)

    # Compare key metrics
    metrics = [
        'total_orders',
        'unique_products',
        'products_before',
        'products_after',
        'avg_products_per_order',
        'product_diversity',
        'avg_days_between_orders',
        'consistency_score',
        'repurchase_rate',
        'frequent_products_count',
        'overlap_rate',
        'new_products_rate'
    ]

    comparison = []

    for metric in metrics:
        if metric not in success_df.columns or metric not in failure_df.columns:
            continue

        success_mean = success_df[metric].mean()
        failure_mean = failure_df[metric].mean()

        diff = success_mean - failure_mean
        diff_pct = (diff / failure_mean * 100) if failure_mean != 0 else 0

        comparison.append({
            'metric': metric,
            'success_avg': success_mean,
            'failure_avg': failure_mean,
            'difference': diff,
            'difference_pct': diff_pct
        })

    comp_df = pd.DataFrame(comparison)

    logger.info("\nKey Metrics Comparison:")
    logger.info("=" * 100)
    logger.info(f"{'Metric':<30} {'Success':>12} {'Failure':>12} {'Diff':>12} {'Diff %':>12}")
    logger.info("=" * 100)

    for _, row in comp_df.iterrows():
        logger.info(f"{row['metric']:<30} {row['success_avg']:>12.2f} {row['failure_avg']:>12.2f} {row['difference']:>12.2f} {row['difference_pct']:>11.1f}%")

    # Identify key differentiators (>50% difference)
    key_diffs = comp_df[abs(comp_df['difference_pct']) > 50].sort_values('difference_pct', ascending=False)

    if len(key_diffs) > 0:
        logger.info(f"\n{'='*80}")
        logger.info("KEY DIFFERENTIATORS (>50% difference):")
        logger.info(f"{'='*80}\n")

        for _, row in key_diffs.iterrows():
            direction = "HIGHER" if row['difference'] > 0 else "LOWER"
            logger.info(f"✓ {row['metric']}: Success is {abs(row['difference_pct']):.0f}% {direction}")

    return comp_df


def main():
    logger.info("="*80)
    logger.info("PHASE 1.1: REGULAR USER FAILURE PATTERN ANALYSIS")
    logger.info("="*80)
    logger.info(f"\nGoal: Understand why some regular users get 90% precision")
    logger.info(f"      while others get 10% precision\n")

    conn = connect_db()

    all_results = []
    success_results = []
    failure_results = []
    mediocre_results = []

    try:
        # Analyze success cases
        logger.info(f"\n{'='*80}")
        logger.info("SUCCESS CASES (90% precision)")
        logger.info(f"{'='*80}")

        for customer in REGULAR_USERS['success']:
            result = analyze_customer(conn, customer['id'], customer['precision'], customer['label'])
            if result:
                all_results.append(result)
                success_results.append(result)

        # Analyze mediocre cases
        logger.info(f"\n{'='*80}")
        logger.info("MEDIOCRE CASES (50% precision)")
        logger.info(f"{'='*80}")

        for customer in REGULAR_USERS['mediocre']:
            result = analyze_customer(conn, customer['id'], customer['precision'], customer['label'])
            if result:
                all_results.append(result)
                mediocre_results.append(result)

        # Analyze failure cases
        logger.info(f"\n{'='*80}")
        logger.info("FAILURE CASES (10-16% precision)")
        logger.info(f"{'='*80}")

        for customer in REGULAR_USERS['failure']:
            result = analyze_customer(conn, customer['id'], customer['precision'], customer['label'])
            if result:
                all_results.append(result)
                failure_results.append(result)

        # Compare groups
        if success_results and failure_results:
            comparison_df = compare_groups(success_results, failure_results)

        # Save results
        output_file = 'regular_user_pattern_analysis.json'
        with open(output_file, 'w') as f:
            json.dump({
                'meta': {
                    'analysis_date': datetime.now().isoformat(),
                    'as_of_date': AS_OF_DATE.strftime('%Y-%m-%d'),
                    'customers_analyzed': len(all_results),
                    'success_count': len(success_results),
                    'failure_count': len(failure_results)
                },
                'customers': all_results,
                'comparison': comparison_df.to_dict('records') if 'comparison_df' in locals() else []
            }, f, indent=2, default=str)

        logger.info(f"\n💾 Results saved to: {output_file}")

        # Generate insights
        logger.info(f"\n{'='*80}")
        logger.info("KEY INSIGHTS")
        logger.info(f"{'='*80}\n")

        if success_results and failure_results:
            success_df = pd.DataFrame(success_results)
            failure_df = pd.DataFrame(failure_results)

            # Insight 1: Product overlap
            success_overlap = success_df['overlap_rate'].mean()
            failure_overlap = failure_df['overlap_rate'].mean()

            logger.info(f"1. PRODUCT OVERLAP:")
            logger.info(f"   Success: {success_overlap:.1%} of validation products were in training")
            logger.info(f"   Failure: {failure_overlap:.1%} of validation products were in training")
            logger.info(f"   → Implication: {'Success users buy familiar products' if success_overlap > failure_overlap else 'Success users try new products'}\n")

            # Insight 2: Repurchase behavior
            success_repurchase = success_df['repurchase_rate'].mean()
            failure_repurchase = failure_df['repurchase_rate'].mean()

            logger.info(f"2. REPURCHASE BEHAVIOR:")
            logger.info(f"   Success: {success_repurchase:.1%} of products are repurchased")
            logger.info(f"   Failure: {failure_repurchase:.1%} of products are repurchased")
            logger.info(f"   → Implication: {'Success users have predictable patterns' if success_repurchase > failure_repurchase else 'Failure users are more exploratory'}\n")

            # Insight 3: Purchase consistency
            success_consistency = success_df['consistency_score'].mean()
            failure_consistency = failure_df['consistency_score'].mean()

            logger.info(f"3. PURCHASE CONSISTENCY:")
            logger.info(f"   Success: {success_consistency:.2f} consistency score")
            logger.info(f"   Failure: {failure_consistency:.2f} consistency score")
            logger.info(f"   → Implication: {'Success users order regularly' if success_consistency > failure_consistency else 'Failure users order sporadically'}\n")

        logger.info(f"{'='*80}")
        logger.info("NEXT STEPS")
        logger.info(f"{'='*80}\n")
        logger.info("1. Based on findings, tune weights for regular users")
        logger.info("2. If overlap_rate is key: Increase frequency weight for high-overlap users")
        logger.info("3. If new_products_rate is high: Increase recency/seasonality weights")
        logger.info("4. Consider sub-segmenting regular users into 'consistent' vs 'sporadic'")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
