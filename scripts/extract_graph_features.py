#!/usr/bin/env python3
"""
Extract Rich Features for Graph Neural Network

This script extracts:
1. Product categories and hierarchies
2. Product analogues (similarity graph)
3. Product-CarBrand relationships (compatibility)
4. Co-purchase patterns from transactions
5. Text-based product features
6. Customer regional information

Output: DuckDB database with all graph edges and node features
"""

import os
import pymssql
import pandas as pd
import duckdb
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
def get_mssql_conn():
    return pymssql.connect(
        server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
        port=int(os.environ.get('MSSQL_PORT', '1433')),
        database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
        user=os.environ.get('MSSQL_USER', 'ef_migrator'),
        password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
    )


def extract_product_categories(conn):
    """Extract product category hierarchy (using ProductProductGroup)"""
    logger.info("📦 Extracting product groups...")

    query = """
    SELECT
        ppg.ProductID as product_id,
        pg.ID as group_id,
        pg.Name as group_name,
        pg.IsSubGroup as is_subgroup
    FROM dbo.ProductProductGroup ppg
    INNER JOIN dbo.ProductGroup pg ON ppg.ProductGroupID = pg.ID
    WHERE pg.Deleted = 0
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} product-group relationships")
    logger.info(f"   Unique groups: {df['group_id'].nunique()}")

    return df


def extract_all_product_groups(conn):
    """Extract all product groups as nodes"""
    logger.info("📦 Extracting all product groups...")

    query = """
    SELECT
        pg.ID as group_id,
        pg.Name as group_name,
        pg.IsSubGroup as is_subgroup,
        pg.IsActive as is_active
    FROM dbo.ProductGroup pg
    WHERE pg.Deleted = 0
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} product groups")
    logger.info(f"   Sub-groups: {df['is_subgroup'].sum()}")

    return df


def extract_product_analogues(conn):
    """Extract product analogue (similarity) relationships"""
    logger.info("🔗 Extracting product analogues...")

    query = """
    SELECT
        pa.BaseProductID as product_id,
        pa.AnalogueProductID as analogue_id
    FROM dbo.ProductAnalogue pa
    INNER JOIN dbo.Product p1 ON pa.BaseProductID = p1.ID
    INNER JOIN dbo.Product p2 ON pa.AnalogueProductID = p2.ID
    WHERE pa.Deleted = 0 AND p1.Deleted = 0 AND p2.Deleted = 0
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} analogue relationships")
    logger.info(f"   Products with analogues: {df['product_id'].nunique()}")

    return df


def extract_product_car_brands(conn):
    """Extract product-car brand compatibility"""
    logger.info("🚗 Extracting product-car brand relationships...")

    query = """
    SELECT
        pcb.ProductID as product_id,
        cb.ID as brand_id,
        cb.Name as brand_name
    FROM dbo.ProductCarBrand pcb
    INNER JOIN dbo.CarBrand cb ON pcb.CarBrandID = cb.ID
    INNER JOIN dbo.Product p ON pcb.ProductID = p.ID
    WHERE p.Deleted = 0 AND cb.Deleted = 0
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} product-brand relationships")
    logger.info(f"   Unique brands: {df['brand_id'].nunique()}")

    return df


def extract_copurchase_graph(conn):
    """Extract co-purchase patterns (products bought together)"""
    logger.info("🛒 Extracting co-purchase patterns...")

    # Get products purchased in the same order
    query = """
    SELECT
        oi1.ProductID as product_id_1,
        oi2.ProductID as product_id_2,
        COUNT(DISTINCT o.ID) as copurchase_count
    FROM dbo.[Order] o
    INNER JOIN dbo.OrderItem oi1 ON o.ID = oi1.OrderID
    INNER JOIN dbo.OrderItem oi2 ON o.ID = oi2.OrderID
    WHERE oi1.ProductID < oi2.ProductID  -- Avoid duplicates
      AND o.Created >= '2024-01-01'  -- Recent data only
    GROUP BY oi1.ProductID, oi2.ProductID
    HAVING COUNT(DISTINCT o.ID) >= 2  -- Co-purchased at least twice
    ORDER BY copurchase_count DESC
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} co-purchase edges")
    logger.info(f"   Top co-purchase frequency: {df['copurchase_count'].max() if len(df) > 0 else 0}")

    return df


def extract_product_features(conn):
    """Extract product node features"""
    logger.info("📊 Extracting product features...")

    query = """
    SELECT
        p.ID as product_id,
        p.Name as name,
        p.Description as description,
        p.VendorCode as vendor_code,
        p.Weight as weight,
        p.Size as size,
        p.MeasureUnitID as measure_unit,
        p.IsForSale as is_for_sale,
        p.HasImage as has_image
    FROM dbo.Product p
    WHERE p.Deleted = 0
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} products")
    logger.info(f"   Products for sale: {df['is_for_sale'].sum()}")

    return df


def extract_customer_features(conn):
    """Extract customer node features"""
    logger.info("👥 Extracting customer features...")

    query = """
    SELECT
        c.ID as customer_id,
        c.RegionID as region_id,
        c.RegionCodeID as region_code_id
    FROM dbo.Client c
    WHERE c.Deleted = 0
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} customers")
    logger.info(f"   Unique regions: {df['region_id'].nunique()}")

    return df


def extract_purchase_history(conn):
    """Extract customer-product purchase edges"""
    logger.info("💳 Extracting purchase history...")

    query = """
    SELECT
        ca.ClientID as customer_id,
        oi.ProductID as product_id,
        o.Created as purchase_date,
        oi.Qty as quantity,
        oi.PricePerItem as price
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE o.Created >= '2023-01-01'  -- Last 2 years
    """

    df = pd.read_sql(query, conn)
    logger.info(f"   Found {len(df)} purchase records")
    logger.info(f"   Date range: {df['purchase_date'].min()} to {df['purchase_date'].max()}")

    return df


def save_to_duckdb(data_dict: Dict[str, pd.DataFrame], db_path: str):
    """Save all extracted data to DuckDB"""
    logger.info(f"\n💾 Saving data to {db_path}...")

    conn = duckdb.connect(db_path)

    for table_name, df in data_dict.items():
        if len(df) > 0:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            logger.info(f"   ✅ Saved {table_name}: {len(df)} rows")
        else:
            logger.warning(f"   ⚠️  Skipped {table_name}: empty dataframe")

    conn.close()
    logger.info(f"✅ All data saved to {db_path}")


def main():
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING GRAPH FEATURES FOR GNN")
    logger.info("="*80 + "\n")

    # Connect to MSSQL
    conn = get_mssql_conn()

    try:
        # Extract all features
        data = {
            'product_features': extract_product_features(conn),
            'customer_features': extract_customer_features(conn),
            'product_group_edges': extract_product_categories(conn),  # Product-to-Group edges
            'product_group_nodes': extract_all_product_groups(conn),  # Group nodes
            'product_analogues': extract_product_analogues(conn),
            'product_car_brands': extract_product_car_brands(conn),
            'copurchase_edges': extract_copurchase_graph(conn),
            'purchase_history': extract_purchase_history(conn)
        }

        # Save to DuckDB
        db_path = 'data/graph_features.duckdb'
        os.makedirs('data', exist_ok=True)
        save_to_duckdb(data, db_path)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 EXTRACTION SUMMARY")
        logger.info("="*80)

        logger.info(f"\n✅ Products: {len(data['product_features'])}")
        logger.info(f"✅ Customers: {len(data['customer_features'])}")

        logger.info(f"\n🗂️  Hierarchies:")
        logger.info(f"   - Product-Group edges: {len(data['product_group_edges'])}")
        logger.info(f"   - Group nodes: {len(data['product_group_nodes'])}")

        logger.info(f"\n🔗 Relationships:")
        logger.info(f"   - Analogue edges: {len(data['product_analogues'])}")
        logger.info(f"   - Brand compatibility edges: {len(data['product_car_brands'])}")
        logger.info(f"   - Co-purchase edges: {len(data['copurchase_edges'])}")
        logger.info(f"   - Purchase edges: {len(data['purchase_history'])}")

        total_edges = (len(data['product_group_edges']) +
                      len(data['product_analogues']) +
                      len(data['product_car_brands']) +
                      len(data['copurchase_edges']) +
                      len(data['purchase_history']))

        logger.info(f"\n📈 Total graph edges: {total_edges:,}")

        logger.info("\n✅ Feature extraction complete!")
        logger.info(f"   Data saved to: {db_path}")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
