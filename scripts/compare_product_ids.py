#!/usr/bin/env python3
"""
Compare Product IDs - Direct Comparison
"""
import pymssql
import duckdb
from pathlib import Path

TEST_CUSTOMER = 410376

# Connect to MSSQL
print("Connecting to MSSQL...")
mssql_conn = pymssql.connect(
    server="78.152.175.67",
    port=1433,
    user="ef_migrator",
    password="Grimm_jow92",
    database="ConcordDb_v5",
    tds_version='7.0'
)

# Get 10 products from MSSQL
query = f"""
SELECT TOP 10 oi.ProductID
FROM dbo.ClientAgreement ca
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
WHERE ca.ClientID = {TEST_CUSTOMER}
ORDER BY o.Created DESC
"""
cursor = mssql_conn.cursor()
cursor.execute(query)
mssql_products = [str(row[0]) for row in cursor.fetchall()]
mssql_conn.close()

print(f"\nMSSQL Products (10 recent):")
for i, p in enumerate(mssql_products, 1):
    print(f"  {i}. '{p}' (type: {type(p).__name__}, len: {len(p)})")

# Connect to DuckDB
print(f"\nConnecting to DuckDB...")
duckdb_path = Path("data/ml_features/concord_ml.duckdb")
duckdb_conn = duckdb.connect(str(duckdb_path), read_only=True)

# Get 10 products from DuckDB
query = f"""
SELECT product_id
FROM ml_features.interaction_matrix
WHERE customer_id = '{TEST_CUSTOMER}'
ORDER BY last_purchase_date DESC
LIMIT 10
"""
result = duckdb_conn.execute(query).fetchall()
duckdb_products = [str(row[0]) for row in result]
duckdb_conn.close()

print(f"\nDuckDB Products (10 recent):")
for i, p in enumerate(duckdb_products, 1):
    print(f"  {i}. '{p}' (type: {type(p).__name__}, len: {len(p)})")

# Compare
print(f"\nDirect Comparison:")
print(f"  MSSQL sample: {mssql_products[:3]}")
print(f"  DuckDB sample: {duckdb_products[:3]}")

# Check if ANY overlap
mssql_set = set(mssql_products)
duckdb_set = set(duckdb_products)
overlap = mssql_set & duckdb_set
print(f"  Overlap in sample: {len(overlap)}/10")
if overlap:
    print(f"  Matching: {overlap}")
