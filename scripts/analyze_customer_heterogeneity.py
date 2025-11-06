#!/usr/bin/env python3
"""
Phase F: Customer Heterogeneity Analysis & Sub-Segmentation

Objective: Understand why heavy users have 30pp performance variance

Key Insight from Phase E:
- Customer variance (84.2% → 54.6% = -30pp) >> optimization gains (+2.2pp)
- Some heavy users highly predictable (80-90%), others not (20-40%)
- Root cause: Customer heterogeneity within "heavy" segment

Approach:
1. Extract behavioral features for all heavy users
2. Cluster into sub-segments (k=3-4)
3. Correlate clusters with precision
4. Identify characteristics of predictable customers
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import pymssql
from datetime import datetime, timedelta
from scipy.stats import variation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def connect_mssql():
    """Connect to MSSQL database"""
    return pymssql.connect(
        server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
        port=int(os.environ.get('MSSQL_PORT', '1433')),
        database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
        user=os.environ.get('MSSQL_USER', 'ef_migrator'),
        password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
    )


def get_heavy_customers(conn):
    """Get all heavy user customers (500+ unique products)"""
    query = """
    SELECT
        ca.ClientID as customer_id,
        COUNT(DISTINCT oi.ProductID) as unique_products,
        COUNT(DISTINCT o.ID) as total_orders
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE o.Created < '2024-07-01'
        AND o.Created IS NOT NULL
        AND oi.ProductID IS NOT NULL
    GROUP BY ca.ClientID
    HAVING COUNT(DISTINCT oi.ProductID) >= 500
    ORDER BY unique_products DESC
    """
    df = pd.read_sql(query, conn)
    logger.info(f"Found {len(df)} heavy customers (500+ products)")
    return df


def extract_customer_features(customer_id: int, conn) -> dict:
    """
    Extract behavioral features for a customer

    Features:
    - Purchase regularity: CV of order intervals
    - Repeat purchase rate: % products bought 2+ times
    - Order frequency stability: CV of monthly order counts
    - Average order size: mean products per order
    - Order size variance: CV of order sizes
    - Product diversity: unique products / total orders
    """

    # Get order history
    order_query = f"""
    SELECT
        o.ID as order_id,
        o.Created as order_date,
        oi.ProductID as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {customer_id}
        AND o.Created < '2024-07-01'
        AND o.Created IS NOT NULL
        AND oi.ProductID IS NOT NULL
    ORDER BY o.Created
    """

    df = pd.read_sql(order_query, conn)

    if len(df) == 0:
        return None

    df['order_date'] = pd.to_datetime(df['order_date'])

    # 1. Purchase Regularity: CV of order intervals (days)
    order_dates = df.groupby('order_id')['order_date'].first().sort_values()
    if len(order_dates) > 1:
        intervals = order_dates.diff().dt.days.dropna()
        purchase_regularity = variation(intervals) if len(intervals) > 0 and intervals.mean() > 0 else 999
    else:
        purchase_regularity = 999

    # 2. Repeat Purchase Rate: % products bought 2+ times
    product_counts = df['product_id'].value_counts()
    repeat_rate = (product_counts >= 2).sum() / len(product_counts) if len(product_counts) > 0 else 0

    # 3. Order Frequency Stability: CV of monthly order counts
    df['year_month'] = df['order_date'].dt.to_period('M')
    monthly_orders = df.groupby('year_month')['order_id'].nunique()
    if len(monthly_orders) > 1 and monthly_orders.mean() > 0:
        frequency_stability = variation(monthly_orders)
    else:
        frequency_stability = 999

    # 4. Average Order Size: mean products per order
    order_sizes = df.groupby('order_id')['product_id'].nunique()
    avg_order_size = order_sizes.mean()

    # 5. Order Size Variance: CV of order sizes
    if len(order_sizes) > 1 and order_sizes.mean() > 0:
        order_size_variance = variation(order_sizes)
    else:
        order_size_variance = 999

    # 6. Total unique products
    unique_products = df['product_id'].nunique()

    # 7. Total orders
    total_orders = df['order_id'].nunique()

    # 8. Time span (days)
    time_span = (df['order_date'].max() - df['order_date'].min()).days

    # 9. Product diversity: unique products per order ratio
    product_diversity = unique_products / total_orders if total_orders > 0 else 0

    return {
        'customer_id': customer_id,
        'purchase_regularity': purchase_regularity,  # Lower = more regular
        'repeat_purchase_rate': repeat_rate,  # Higher = more repeats
        'frequency_stability': frequency_stability,  # Lower = more stable
        'avg_order_size': avg_order_size,
        'order_size_variance': order_size_variance,  # Lower = more consistent
        'product_diversity': product_diversity,  # Higher = more exploratory
        'unique_products': unique_products,
        'total_orders': total_orders,
        'time_span_days': time_span,
        'orders_per_month': total_orders / (time_span / 30) if time_span > 0 else 0
    }


def cluster_customers(features_df: pd.DataFrame, n_clusters=3):
    """Cluster customers using K-means"""

    # Select features for clustering (exclude identifiers and raw counts)
    cluster_features = [
        'purchase_regularity',
        'repeat_purchase_rate',
        'frequency_stability',
        'order_size_variance',
        'product_diversity',
        'orders_per_month'
    ]

    # Handle infinite values
    X = features_df[cluster_features].copy()
    X = X.replace([np.inf, -np.inf], 999)
    X = X.fillna(999)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(X_scaled)

    return features_df, kmeans, scaler


def main():
    logger.info("="*80)
    logger.info("PHASE F: CUSTOMER HETEROGENEITY ANALYSIS")
    logger.info("="*80)
    logger.info("\nObjective: Understand 30pp performance variance within heavy users")
    logger.info("Hypothesis: Different behavior patterns → different predictability\n")

    conn = connect_mssql()

    try:
        # Get heavy customers
        heavy_customers = get_heavy_customers(conn)

        logger.info(f"\nStep 1: Extracting behavioral features for {len(heavy_customers)} customers...")

        features_list = []
        for i, row in heavy_customers.iterrows():
            customer_id = row['customer_id']
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(heavy_customers)} customers")

            features = extract_customer_features(customer_id, conn)
            if features:
                features_list.append(features)

        features_df = pd.DataFrame(features_list)

        logger.info(f"\nExtracted features for {len(features_df)} customers")
        logger.info("\nFeature Summary:")
        logger.info(features_df.describe())

        # Cluster customers
        logger.info(f"\nStep 2: Clustering customers into sub-segments...")

        for n_clusters in [3, 4]:
            logger.info(f"\n--- Testing k={n_clusters} clusters ---")

            features_clustered, kmeans, scaler = cluster_customers(features_df.copy(), n_clusters)

            # Analyze clusters
            logger.info(f"\nCluster Distribution:")
            cluster_counts = features_clustered['cluster'].value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                pct = count / len(features_clustered) * 100
                logger.info(f"  Cluster {cluster_id}: {count} customers ({pct:.1f}%)")

            logger.info(f"\nCluster Characteristics:")
            for cluster_id in range(n_clusters):
                cluster_data = features_clustered[features_clustered['cluster'] == cluster_id]
                logger.info(f"\n  Cluster {cluster_id} (n={len(cluster_data)}):")
                logger.info(f"    Purchase Regularity (CV):  {cluster_data['purchase_regularity'].median():.2f}")
                logger.info(f"    Repeat Purchase Rate:      {cluster_data['repeat_purchase_rate'].mean():.2f}")
                logger.info(f"    Frequency Stability (CV):  {cluster_data['frequency_stability'].median():.2f}")
                logger.info(f"    Order Size Variance (CV):  {cluster_data['order_size_variance'].median():.2f}")
                logger.info(f"    Product Diversity:         {cluster_data['product_diversity'].mean():.2f}")
                logger.info(f"    Orders/Month:              {cluster_data['orders_per_month'].mean():.2f}")

        # Use k=3 for final analysis
        features_clustered, kmeans, scaler = cluster_customers(features_df.copy(), n_clusters=3)

        # Save results
        os.makedirs('results', exist_ok=True)
        features_clustered.to_csv('results/customer_clusters.csv', index=False)

        logger.info("\n" + "="*80)
        logger.info("CLUSTERING COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults saved to: results/customer_clusters.csv")
        logger.info(f"\nNext Step: Correlate clusters with precision to identify")
        logger.info(f"which customer types are predictable vs unpredictable")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
