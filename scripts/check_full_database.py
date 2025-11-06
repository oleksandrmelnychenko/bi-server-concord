#!/usr/bin/env python3
"""
Check Full Database - What are we missing?
"""

import pymssql
import pandas as pd

# Connect to MSSQL
conn = pymssql.connect(
    server="78.152.175.67",
    port=1433,
    user="ef_migrator",
    password="Grimm_jow92",
    database="ConcordDb_v5",
    tds_version='7.0'
)

print("="*80)
print("FULL DATABASE ANALYSIS")
print("="*80)

# 1. Check total records WITHOUT filters
print("\n1. Total Records (NO FILTERS):")
query = """
SELECT
    COUNT(DISTINCT ca.ClientID) as total_customers,
    COUNT(DISTINCT o.ID) as total_orders,
    COUNT(DISTINCT oi.ID) as total_order_items,
    COUNT(DISTINCT oi.ProductID) as total_products
FROM dbo.ClientAgreement ca
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
"""
df = pd.read_sql(query, conn)
print(f"   Customers: {df['total_customers'].iloc[0]:,}")
print(f"   Orders: {df['total_orders'].iloc[0]:,}")
print(f"   Order Items: {df['total_order_items'].iloc[0]:,}")
print(f"   Unique Products: {df['total_products'].iloc[0]:,}")

# 2. Check with current filters (Deleted = 0)
print("\n2. With Current Filters (Deleted = 0, etc):")
query = """
SELECT
    COUNT(DISTINCT ca.ClientID) as total_customers,
    COUNT(DISTINCT o.ID) as total_orders,
    COUNT(DISTINCT oi.ID) as total_order_items,
    COUNT(DISTINCT oi.ProductID) as total_products
FROM dbo.ClientAgreement ca
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
WHERE ca.Deleted = 0
    AND o.Deleted = 0
    AND oi.Deleted = 0
    AND o.Created IS NOT NULL
    AND oi.ProductID IS NOT NULL
    AND oi.Qty > 0
"""
df = pd.read_sql(query, conn)
print(f"   Customers: {df['total_customers'].iloc[0]:,}")
print(f"   Orders: {df['total_orders'].iloc[0]:,}")
print(f"   Order Items: {df['total_order_items'].iloc[0]:,}")
print(f"   Unique Products: {df['total_products'].iloc[0]:,}")

# 3. What's being filtered out?
print("\n3. What's Being Filtered Out:")

# Deleted records
query = """
SELECT
    SUM(CASE WHEN ca.Deleted = 1 THEN 1 ELSE 0 END) as deleted_agreements,
    SUM(CASE WHEN o.Deleted = 1 THEN 1 ELSE 0 END) as deleted_orders,
    SUM(CASE WHEN oi.Deleted = 1 THEN 1 ELSE 0 END) as deleted_items,
    COUNT(*) as total
FROM dbo.ClientAgreement ca
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
"""
df = pd.read_sql(query, conn)
print(f"   Deleted Agreements: {df['deleted_agreements'].iloc[0]:,}")
print(f"   Deleted Orders: {df['deleted_orders'].iloc[0]:,}")
print(f"   Deleted Items: {df['deleted_items'].iloc[0]:,}")

# 4. Check date range
print("\n4. Date Range:")
query = """
SELECT
    MIN(o.Created) as earliest_order,
    MAX(o.Created) as latest_order,
    COUNT(DISTINCT YEAR(o.Created)) as years_span
FROM dbo.[Order] o
WHERE o.Deleted = 0
    AND o.Created IS NOT NULL
"""
df = pd.read_sql(query, conn)
print(f"   Earliest: {df['earliest_order'].iloc[0]}")
print(f"   Latest: {df['latest_order'].iloc[0]}")
print(f"   Years: {df['years_span'].iloc[0]}")

# 5. Customer-Product interaction count (what we extract)
print("\n5. Customer-Product Interactions (Aggregated):")
query = """
SELECT COUNT(*) as interaction_count
FROM (
    SELECT
        ca.ClientID,
        oi.ProductID
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE ca.Deleted = 0
        AND o.Deleted = 0
        AND oi.Deleted = 0
        AND o.Created IS NOT NULL
        AND oi.ProductID IS NOT NULL
        AND oi.Qty > 0
    GROUP BY ca.ClientID, oi.ProductID
) as interactions
"""
df = pd.read_sql(query, conn)
print(f"   Total Interactions: {df['interaction_count'].iloc[0]:,}")
print(f"   (This is what we extract to DuckDB)")

# 6. Check if we need to remove ANY filters
print("\n6. RECOMMENDATION:")
print("   If you want FULL database without any filtering:")
print("   - Remove 'Deleted = 0' filters")
print("   - Remove 'IS NOT NULL' filters")
print("   - Remove 'Qty > 0' filter")
print("   This will extract ALL data including deleted/invalid records")

conn.close()

print("\n" + "="*80)
print("Should we extract with NO filters? (y/n)")
