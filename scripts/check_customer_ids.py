#!/usr/bin/env python3
"""
Check Customer ID Mismatch

Are we querying for the wrong customer IDs?
"""

import duckdb
import pymssql
import pandas as pd
from pathlib import Path

# Connect to DuckDB
duckdb_path = Path("data/ml_features/concord_ml.duckdb")
duckdb_conn = duckdb.connect(str(duckdb_path), read_only=True)

print("="*80)
print("CUSTOMER ID INVESTIGATION")
print("="*80)

# Get sample customers from DuckDB
print("\n1. Customers in DuckDB:")
query = """
SELECT DISTINCT customer_id
FROM ml_features.interaction_matrix
ORDER BY customer_id
LIMIT 20
"""

duckdb_customers = duckdb_conn.execute(query).df()
print(f"   Total customers in DuckDB: {duckdb_conn.execute('SELECT COUNT(DISTINCT customer_id) FROM ml_features.interaction_matrix').fetchone()[0]}")
print(f"   Sample customer IDs (first 20):")
for i, cid in enumerate(duckdb_customers['customer_id'].tolist(), 1):
    print(f"      {i}. {cid} (type: {type(cid).__name__})")

# Get sample customers from MSSQL
print("\n2. Customers in MSSQL:")
mssql_conn = pymssql.connect(
    server="78.152.175.67",
    port=1433,
    user="ef_migrator",
    password="Grimm_jow92",
    database="ConcordDb_v5",
    tds_version='7.0'
)

query = """
SELECT DISTINCT TOP 20 ca.ClientID
FROM dbo.ClientAgreement ca
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
WHERE ca.Deleted = 0
    AND o.Deleted = 0
ORDER BY ca.ClientID
"""

mssql_customers = pd.read_sql(query, mssql_conn)
print(f"   Sample customer IDs (first 20):")
for i, cid in enumerate(mssql_customers['ClientID'].tolist(), 1):
    print(f"      {i}. {cid} (type: {type(cid).__name__})")

# Check if 410376 exists
print("\n3. Checking test customer 410376:")
test_customer_str = "410376"
test_customer_int = 410376

# In DuckDB
duckdb_count = duckdb_conn.execute(f"SELECT COUNT(*) FROM ml_features.interaction_matrix WHERE customer_id = '{test_customer_str}'").fetchone()[0]
print(f"   DuckDB (string '410376'): {duckdb_count} records")

duckdb_count2 = duckdb_conn.execute(f"SELECT COUNT(*) FROM ml_features.interaction_matrix WHERE customer_id = '{test_customer_int}'").fetchone()[0]
print(f"   DuckDB (int 410376): {duckdb_count2} records")

# In MSSQL
mssql_count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM dbo.ClientAgreement WHERE ClientID = {test_customer_int} AND Deleted = 0", mssql_conn)
print(f"   MSSQL (int 410376): {mssql_count['cnt'].iloc[0]} ClientAgreements")

# Check what's actually in DuckDB for this customer ID pattern
print("\n4. Searching for similar customer IDs in DuckDB:")
similar = duckdb_conn.execute(f"""
SELECT customer_id, COUNT(*) as num_products
FROM ml_features.interaction_matrix
WHERE customer_id LIKE '%410376%'
GROUP BY customer_id
ORDER BY num_products DESC
LIMIT 10
""").df()

if len(similar) > 0:
    print(f"   Found {len(similar)} customers matching pattern:")
    for _, row in similar.iterrows():
        print(f"      Customer '{row['customer_id']}': {row['num_products']} products")
else:
    print(f"   No customers matching '410376' pattern")

# Close connections
duckdb_conn.close()
mssql_conn.close()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("If DuckDB shows 0 records for test customer 410376,")
print("then the data extraction script is pulling different customers entirely!")
