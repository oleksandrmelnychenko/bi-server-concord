#!/usr/bin/env python3
"""
Check Date Range in DuckDB
"""
import duckdb
from pathlib import Path

TEST_CUSTOMER = 410376

# Connect to DuckDB
duckdb_path = Path("data/ml_features/concord_ml.duckdb")
duckdb_conn = duckdb.connect(str(duckdb_path), read_only=True)

# Get date range for this customer
query = f"""
SELECT
    MIN(first_purchase_date) as earliest,
    MAX(last_purchase_date) as latest,
    COUNT(*) as num_products
FROM ml_features.interaction_matrix
WHERE customer_id = '{TEST_CUSTOMER}'
"""
result = duckdb_conn.execute(query).fetchone()
print(f"Customer {TEST_CUSTOMER} in DuckDB:")
print(f"  Earliest purchase: {result[0]}")
print(f"  Latest purchase: {result[1]}")
print(f"  Number of products: {result[2]}")

# Get count by year
query = f"""
SELECT
    YEAR(last_purchase_date) as year,
    COUNT(*) as num_products
FROM ml_features.interaction_matrix
WHERE customer_id = '{TEST_CUSTOMER}'
GROUP BY YEAR(last_purchase_date)
ORDER BY year
"""
results = duckdb_conn.execute(query).fetchall()
print(f"\nProducts by year (based on last purchase):")
for year, count in results:
    print(f"  {year}: {count} products")

# Check 2024 split
query = f"""
SELECT
    COUNT(CASE WHEN last_purchase_date < '2024-07-01' THEN 1 END) as before_split,
    COUNT(CASE WHEN last_purchase_date >= '2024-07-01' THEN 1 END) as after_split
FROM ml_features.interaction_matrix
WHERE customer_id = '{TEST_CUSTOMER}'
"""
result = duckdb_conn.execute(query).fetchone()
print(f"\n2024 Split (2024-06-30):")
print(f"  Products last purchased BEFORE split: {result[0]}")
print(f"  Products last purchased AFTER split: {result[1]}")

# Sample products from each period
print(f"\nSample products from BEFORE split:")
query = f"""
SELECT product_id, first_purchase_date, last_purchase_date
FROM ml_features.interaction_matrix
WHERE customer_id = '{TEST_CUSTOMER}'
  AND last_purchase_date < '2024-07-01'
ORDER BY last_purchase_date DESC
LIMIT 5
"""
results = duckdb_conn.execute(query).fetchall()
for prod_id, first_date, last_date in results:
    print(f"  {prod_id}: {first_date} → {last_date}")

print(f"\nSample products from AFTER split:")
query = f"""
SELECT product_id, first_purchase_date, last_purchase_date
FROM ml_features.interaction_matrix
WHERE customer_id = '{TEST_CUSTOMER}'
  AND last_purchase_date >= '2024-07-01'
ORDER BY last_purchase_date DESC
LIMIT 5
"""
results = duckdb_conn.execute(query).fetchall()
for prod_id, first_date, last_date in results:
    print(f"  {prod_id}: {first_date} → {last_date}")

duckdb_conn.close()
