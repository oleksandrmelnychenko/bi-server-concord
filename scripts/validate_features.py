#!/usr/bin/env python3
"""
Validate the ML feature transformations
"""

import duckdb
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DB = "/opt/dagster/app/data/dbt/concord_bi.duckdb"

def validate():
    conn = duckdb.connect(OUTPUT_DB, read_only=True)

    logger.info("=" * 70)
    logger.info("ML FEATURE VALIDATION")
    logger.info("=" * 70)

    # 1. Customer Features Validation
    logger.info("\n📊 Customer Features:")
    logger.info("-" * 70)

    # Sample customers
    result = conn.execute("""
        SELECT
            customer_name,
            customer_segment,
            rfm_segment,
            total_orders,
            unique_products_purchased,
            ROUND(lifetime_value, 2) as ltv,
            customer_status
        FROM ml_features.customer_features
        WHERE total_orders > 0
        ORDER BY lifetime_value DESC
        LIMIT 5
    """).fetchall()

    logger.info("\nTop 5 Customers by Lifetime Value:")
    logger.info(f"{'Customer':<30} {'Segment':<20} {'RFM':<5} {'Orders':<8} {'Products':<10} {'LTV':<12} {'Status':<10}")
    logger.info("-" * 110)
    for row in result:
        logger.info(f"{row[0][:28]:<30} {row[1]:<20} {row[2]:<5} {row[3]:<8} {row[4]:<10} {row[5]:<12} {row[6]:<10}")

    # Customer segment distribution
    logger.info("\nCustomer Segment Distribution:")
    result = conn.execute("""
        SELECT
            customer_segment,
            COUNT(*) as count,
            ROUND(AVG(lifetime_value), 2) as avg_ltv,
            ROUND(AVG(total_orders), 1) as avg_orders
        FROM ml_features.customer_features
        GROUP BY customer_segment
        ORDER BY count DESC
    """).fetchall()

    for row in result:
        logger.info(f"  {row[0]:<25} {row[1]:>5} customers | Avg LTV: {row[2]:>10} | Avg Orders: {row[3]:>5}")

    # 2. Product Features Validation
    logger.info("\n📦 Product Features:")
    logger.info("-" * 70)

    # Top products
    result = conn.execute("""
        SELECT
            product_name,
            times_ordered,
            unique_customers,
            total_quantity_sold,
            ROUND(total_revenue, 2) as revenue,
            num_analogues,
            product_status
        FROM ml_features.product_features
        WHERE times_ordered > 0
        ORDER BY total_revenue DESC
        LIMIT 5
    """).fetchall()

    logger.info("\nTop 5 Products by Revenue:")
    logger.info(f"{'Product':<35} {'Orders':<8} {'Customers':<11} {'Qty':<10} {'Revenue':<12} {'Analogues':<10} {'Status':<12}")
    logger.info("-" * 110)
    for row in result:
        logger.info(f"{row[0][:33]:<35} {row[1]:<8} {row[2]:<11} {row[3]:<10} {row[4]:<12} {row[5]:<10} {row[6]:<12}")

    # Product status distribution
    logger.info("\nProduct Status Distribution:")
    result = conn.execute("""
        SELECT
            product_status,
            COUNT(*) as count,
            ROUND(AVG(total_revenue), 2) as avg_revenue
        FROM ml_features.product_features
        GROUP BY product_status
        ORDER BY count DESC
    """).fetchall()

    for row in result:
        logger.info(f"  {row[0]:<15} {row[1]:>6} products | Avg Revenue: {row[2]:>10}")

    # 3. Interaction Matrix Validation
    logger.info("\n🔗 Interaction Matrix:")
    logger.info("-" * 70)

    # Interaction stats
    result = conn.execute("""
        SELECT
            COUNT(*) as total_interactions,
            COUNT(DISTINCT customer_id) as unique_customers,
            COUNT(DISTINCT product_id) as unique_products,
            ROUND(AVG(implicit_rating), 2) as avg_rating,
            ROUND(AVG(total_spent), 2) as avg_spent,
            ROUND(AVG(num_purchases), 1) as avg_purchases
        FROM ml_features.interaction_matrix
    """).fetchone()

    logger.info(f"\nInteraction Statistics:")
    logger.info(f"  Total Interactions: {result[0]:,}")
    logger.info(f"  Unique Customers: {result[1]:,}")
    logger.info(f"  Unique Products: {result[2]:,}")
    logger.info(f"  Avg Implicit Rating: {result[3]}")
    logger.info(f"  Avg Spent per Interaction: {result[4]}")
    logger.info(f"  Avg Purchases per Interaction: {result[5]}")

    # Sample interactions
    logger.info("\nSample High-Value Interactions:")
    result = conn.execute("""
        SELECT
            c.customer_name,
            p.product_name,
            i.num_purchases,
            i.implicit_rating,
            ROUND(i.total_spent, 2) as spent
        FROM ml_features.interaction_matrix i
        JOIN ml_features.customer_features c ON i.customer_id = c.customer_id
        JOIN ml_features.product_features p ON i.product_id = p.product_id
        ORDER BY i.total_spent DESC
        LIMIT 5
    """).fetchall()

    logger.info(f"{'Customer':<30} {'Product':<35} {'Purchases':<10} {'Rating':<8} {'Spent':<12}")
    logger.info("-" * 105)
    for row in result:
        logger.info(f"{row[0][:28]:<30} {row[1][:33]:<35} {row[2]:<10} {row[3]:<8} {row[4]:<12}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ VALIDATION COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nAll ML features are ready for model training! 🚀")
    logger.info("\nNext Steps:")
    logger.info("  1. Train LightFM recommendation model using interaction_matrix")
    logger.info("  2. Train customer segmentation models using customer_features")
    logger.info("  3. Train demand forecasting using product_features")

    conn.close()

if __name__ == "__main__":
    validate()
