#!/usr/bin/env python3
"""
Analyze date ranges in the data to help decide on data scope
"""

import duckdb
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DB = "/opt/dagster/app/data/dbt/concord_bi.duckdb"

def analyze_dates():
    conn = duckdb.connect(OUTPUT_DB, read_only=True)

    logger.info("=" * 70)
    logger.info("DATE RANGE ANALYSIS")
    logger.info("=" * 70)

    # Customer creation dates
    logger.info("\n📅 CUSTOMER DATA:")
    logger.info("-" * 70)
    result = conn.execute("""
        SELECT
            MIN(created_at) as earliest_customer,
            MAX(created_at) as latest_customer,
            COUNT(*) as total_customers,
            COUNT(CASE WHEN created_at >= '2023-01-01' THEN 1 END) as customers_since_2023,
            COUNT(CASE WHEN created_at >= '2024-01-01' THEN 1 END) as customers_since_2024
        FROM stg_customers
        WHERE created_at IS NOT NULL
    """).fetchone()

    logger.info(f"Earliest Customer: {result[0]}")
    logger.info(f"Latest Customer:   {result[1]}")
    logger.info(f"Total Customers:   {result[2]:,}")
    logger.info(f"Since 2023:        {result[3]:,} ({result[3]/result[2]*100:.1f}%)")
    logger.info(f"Since 2024:        {result[4]:,} ({result[4]/result[2]*100:.1f}%)")

    # Sales/Transaction dates
    logger.info("\n💰 SALES/TRANSACTION DATA:")
    logger.info("-" * 70)
    result = conn.execute("""
        SELECT
            MIN(created_at) as earliest_sale,
            MAX(created_at) as latest_sale,
            COUNT(*) as total_sales,
            COUNT(CASE WHEN created_at >= '2023-01-01' THEN 1 END) as sales_since_2023,
            COUNT(CASE WHEN created_at >= '2024-01-01' THEN 1 END) as sales_since_2024,
            COUNT(CASE WHEN created_at >= '2025-01-01' THEN 1 END) as sales_since_2025
        FROM stg_sales
        WHERE created_at IS NOT NULL
    """).fetchone()

    logger.info(f"Earliest Sale: {result[0]}")
    logger.info(f"Latest Sale:   {result[1]}")
    logger.info(f"Total Sales:   {result[2]:,}")
    logger.info(f"Since 2023:    {result[3]:,} ({result[3]/result[2]*100:.1f}%)")
    logger.info(f"Since 2024:    {result[4]:,} ({result[4]/result[2]*100:.1f}%)")
    logger.info(f"Since 2025:    {result[5]:,} ({result[5]/result[2]*100:.1f}%)")

    # Sales by year
    logger.info("\n📊 SALES BY YEAR:")
    logger.info("-" * 70)
    result = conn.execute("""
        SELECT
            EXTRACT(YEAR FROM created_at) as year,
            COUNT(*) as num_sales,
            COUNT(DISTINCT client_agreement_id) as unique_customers,
            ROUND(SUM(
                (SELECT SUM(oi.quantity * oi.price_per_item)
                 FROM stg_order_items oi
                 WHERE oi.order_id = s.order_id)
            ), 2) as total_revenue
        FROM stg_sales s
        WHERE created_at IS NOT NULL
        GROUP BY EXTRACT(YEAR FROM created_at)
        ORDER BY year DESC
    """).fetchall()

    logger.info(f"{'Year':<10} {'Sales':<10} {'Customers':<15} {'Revenue':<15}")
    logger.info("-" * 50)
    for row in result:
        logger.info(f"{int(row[0]):<10} {row[1]:<10} {row[2]:<15} ${row[3]:>12,.2f}")

    # Order items by year
    logger.info("\n📦 ORDER ITEMS BY YEAR:")
    logger.info("-" * 70)
    result = conn.execute("""
        SELECT
            EXTRACT(YEAR FROM oi.created_at) as year,
            COUNT(*) as num_items,
            COUNT(DISTINCT oi.product_id) as unique_products,
            SUM(oi.quantity) as total_quantity,
            ROUND(SUM(oi.quantity * oi.price_per_item), 2) as total_revenue
        FROM stg_order_items oi
        WHERE created_at IS NOT NULL
        GROUP BY EXTRACT(YEAR FROM created_at)
        ORDER BY year DESC
    """).fetchall()

    logger.info(f"{'Year':<10} {'Items':<10} {'Products':<12} {'Quantity':<12} {'Revenue':<15}")
    logger.info("-" * 60)
    for row in result:
        logger.info(f"{int(row[0]):<10} {row[1]:<10} {row[2]:<12} {row[3]:<12,.0f} ${row[4]:>12,.2f}")

    # Interaction matrix date ranges
    logger.info("\n🔗 INTERACTION MATRIX (Customer × Product):")
    logger.info("-" * 70)
    result = conn.execute("""
        SELECT
            MIN(first_purchase_date) as earliest_interaction,
            MAX(last_purchase_date) as latest_interaction,
            COUNT(*) as total_interactions,
            COUNT(CASE WHEN last_purchase_date >= '2023-01-01' THEN 1 END) as interactions_since_2023,
            COUNT(CASE WHEN last_purchase_date >= '2024-01-01' THEN 1 END) as interactions_since_2024,
            COUNT(CASE WHEN last_purchase_date >= '2025-01-01' THEN 1 END) as interactions_since_2025
        FROM ml_features.interaction_matrix
    """).fetchone()

    logger.info(f"Earliest Interaction: {result[0]}")
    logger.info(f"Latest Interaction:   {result[1]}")
    logger.info(f"Total Interactions:   {result[2]:,}")
    logger.info(f"Since 2023:           {result[3]:,} ({result[3]/result[2]*100:.1f}%)")
    logger.info(f"Since 2024:           {result[4]:,} ({result[4]/result[2]*100:.1f}%)")
    logger.info(f"Since 2025:           {result[5]:,} ({result[5]/result[2]*100:.1f}%)")

    # Product creation dates
    logger.info("\n📦 PRODUCT DATA:")
    logger.info("-" * 70)
    result = conn.execute("""
        SELECT
            MIN(created_at) as earliest_product,
            MAX(created_at) as latest_product,
            COUNT(*) as total_products,
            COUNT(CASE WHEN created_at >= '2023-01-01' THEN 1 END) as products_since_2023,
            COUNT(CASE WHEN created_at >= '2024-01-01' THEN 1 END) as products_since_2024
        FROM stg_products
        WHERE created_at IS NOT NULL
    """).fetchone()

    logger.info(f"Earliest Product: {result[0]}")
    logger.info(f"Latest Product:   {result[1]}")
    logger.info(f"Total Products:   {result[2]:,}")
    logger.info(f"Since 2023:       {result[3]:,} ({result[3]/result[2]*100:.1f}%)")
    logger.info(f"Since 2024:       {result[4]:,} ({result[4]/result[2]*100:.1f}%)")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)

    conn.close()

if __name__ == "__main__":
    analyze_dates()
