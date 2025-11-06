#!/usr/bin/env python3
"""
Debug Data Mismatch

Why is BOTH ML and Frequency Baseline getting 0% hit rate?
There must be a data mismatch between DuckDB and MSSQL
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pymssql
import duckdb
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# MSSQL Configuration
MSSQL_CONFIG = {
    "host": "78.152.175.67",
    "port": "1433",
    "database": "ConcordDb_v5",
    "user": "ef_migrator",
    "password": "Grimm_jow92",
}

SPLIT_DATE = "2024-06-30"
TEST_CUSTOMER = 410376


def main():
    logger.info("="*80)
    logger.info("DATA MISMATCH INVESTIGATION")
    logger.info("="*80)
    logger.info(f"Test Customer: {TEST_CUSTOMER}")
    logger.info(f"Split Date: {SPLIT_DATE}\n")

    # Connect to MSSQL
    logger.info("1. Querying MSSQL for customer purchases...")
    mssql_conn = pymssql.connect(
        server=MSSQL_CONFIG['host'],
        port=int(MSSQL_CONFIG['port']),
        user=MSSQL_CONFIG['user'],
        password=MSSQL_CONFIG['password'],
        database=MSSQL_CONFIG['database'],
        tds_version='7.0'
    )

    # Get H1 purchases
    query_h1 = f"""
    SELECT DISTINCT oi.ProductID as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {TEST_CUSTOMER}
        AND o.Created < '{SPLIT_DATE}'
    """

    mssql_h1 = pd.read_sql(query_h1, mssql_conn)
    mssql_h1_products = set(mssql_h1['product_id'].astype(str).tolist())

    # Get H2 purchases
    query_h2 = f"""
    SELECT DISTINCT oi.ProductID as product_id
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.ClientID = {TEST_CUSTOMER}
        AND o.Created >= '{SPLIT_DATE}'
        AND o.Created < '2025-01-01'
    """

    mssql_h2 = pd.read_sql(query_h2, mssql_conn)
    mssql_h2_products = set(mssql_h2['product_id'].astype(str).tolist())

    mssql_conn.close()

    logger.info(f"   MSSQL H1 (before {SPLIT_DATE}): {len(mssql_h1_products)} unique products")
    logger.info(f"   MSSQL H2 (after {SPLIT_DATE}): {len(mssql_h2_products)} unique products")

    # Connect to DuckDB
    logger.info(f"\n2. Querying DuckDB for customer purchases...")
    duckdb_path = Path("data/ml_features/concord_ml.duckdb")
    duckdb_conn = duckdb.connect(str(duckdb_path), read_only=True)

    query_duckdb = f"""
    SELECT DISTINCT product_id
    FROM ml_features.interaction_matrix
    WHERE customer_id = '{TEST_CUSTOMER}'
    """

    duckdb_df = duckdb_conn.execute(query_duckdb).df()
    duckdb_products = set(duckdb_df['product_id'].astype(str).tolist())

    duckdb_conn.close()

    logger.info(f"   DuckDB (all time): {len(duckdb_products)} unique products")

    # Compare
    logger.info(f"\n3. Comparing datasets...")

    # Check if DuckDB has H1 data
    h1_in_duckdb = mssql_h1_products & duckdb_products
    h1_missing = mssql_h1_products - duckdb_products

    logger.info(f"\n   H1 products (before split):")
    logger.info(f"      In both MSSQL and DuckDB: {len(h1_in_duckdb)}/{len(mssql_h1_products)} ({len(h1_in_duckdb)/len(mssql_h1_products)*100:.1f}%)")
    logger.info(f"      Missing from DuckDB: {len(h1_missing)}")

    # Check if DuckDB has H2 data
    h2_in_duckdb = mssql_h2_products & duckdb_products
    h2_missing = mssql_h2_products - duckdb_products

    logger.info(f"\n   H2 products (after split):")
    logger.info(f"      In both MSSQL and DuckDB: {len(h2_in_duckdb)}/{len(mssql_h2_products)} ({len(h2_in_duckdb)/len(mssql_h2_products)*100:.1f}%)")
    logger.info(f"      Missing from DuckDB: {len(h2_missing)}")

    # Check repurchase overlap
    repurchased_in_mssql = mssql_h1_products & mssql_h2_products
    logger.info(f"\n   Repurchased products (H1 → H2):")
    logger.info(f"      MSSQL: {len(repurchased_in_mssql)} products")

    # What products are in DuckDB?
    duckdb_in_h1 = duckdb_products & mssql_h1_products
    duckdb_not_in_h1 = duckdb_products - mssql_h1_products

    logger.info(f"\n   DuckDB product sources:")
    logger.info(f"      From H1: {len(duckdb_in_h1)}/{len(duckdb_products)} ({len(duckdb_in_h1)/len(duckdb_products)*100:.1f}%)")
    logger.info(f"      Not in H1: {len(duckdb_not_in_h1)}/{len(duckdb_products)} ({len(duckdb_not_in_h1)/len(duckdb_products)*100:.1f}%)")

    # CRITICAL: Check if DuckDB recommendations would overlap with H2
    logger.info(f"\n4. Would frequency baseline work?")

    # Get top 50 most frequent from DuckDB
    duckdb_conn = duckdb.connect(str(duckdb_path), read_only=True)
    query_top50 = f"""
    SELECT product_id, num_purchases
    FROM ml_features.interaction_matrix
    WHERE customer_id = '{TEST_CUSTOMER}'
    ORDER BY num_purchases DESC
    LIMIT 50
    """

    top50 = duckdb_conn.execute(query_top50).df()
    top50_products = set(top50['product_id'].astype(str).tolist())
    duckdb_conn.close()

    overlap_with_h2 = top50_products & mssql_h2_products

    logger.info(f"   Top 50 most frequent products from DuckDB:")
    logger.info(f"      Would hit H2: {len(overlap_with_h2)}/50 ({len(overlap_with_h2)/50*100:.1f}%)")

    if len(overlap_with_h2) > 0:
        logger.info(f"      ✅ YES! These products WERE repurchased in H2:")
        for i, product_id in enumerate(list(overlap_with_h2)[:10], 1):
            num_purchases = top50[top50['product_id'].astype(str) == product_id]['num_purchases'].iloc[0]
            logger.info(f"         {i}. Product {product_id} ({num_purchases} purchases in DuckDB)")
    else:
        logger.info(f"      ❌ NO overlap!")

    # Root cause
    logger.info(f"\n{'='*80}")
    logger.info(f"ROOT CAUSE DIAGNOSIS")
    logger.info(f"{'='*80}")

    if len(h1_in_duckdb) < len(mssql_h1_products) * 0.9:
        logger.info(f"🔴 PROBLEM: DuckDB is missing {len(h1_missing)} H1 products!")
        logger.info(f"   DuckDB data is incomplete or stale")
        logger.info(f"   Need to re-extract data from MSSQL")
    elif len(overlap_with_h2) > 0:
        logger.info(f"✅ GOOD NEWS: Top 50 WOULD hit {len(overlap_with_h2)} products!")
        logger.info(f"   But validation shows 0 hits... something else is wrong")
        logger.info(f"   Possible issues:")
        logger.info(f"      - Product ID format mismatch (int vs string)?")
        logger.info(f"      - Customer ID mismatch?")
        logger.info(f"      - Date range issue?")
    else:
        logger.info(f"🔴 SEVERE PROBLEM: Top 50 products have ZERO overlap with H2")
        logger.info(f"   This means customer bought completely different products")
        logger.info(f"   OR there's a fundamental data mismatch")


if __name__ == "__main__":
    main()
