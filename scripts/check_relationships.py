#!/usr/bin/env python3
import os
import pymssql
import pandas as pd

conn = pymssql.connect(
    server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
    port=int(os.environ.get('MSSQL_PORT', '1433')),
    database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    user=os.environ.get('MSSQL_USER', 'ef_migrator'),
    password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
)

print("\n=== Checking All Relationship Tables ===\n")

# ProductAnalogue
query = "SELECT COUNT(*) as count FROM dbo.ProductAnalogue"
df = pd.read_sql(query, conn)
print(f"ProductAnalogue records: {df['count'].iloc[0]:,}")

# ProductCarBrand
query = "SELECT COUNT(*) as count FROM dbo.ProductCarBrand"
df = pd.read_sql(query, conn)
print(f"ProductCarBrand records: {df['count'].iloc[0]:,}")

# Check some samples
print("\n=== Sample ProductAnalogue ===")
query = "SELECT TOP 5 * FROM dbo.ProductAnalogue"
df = pd.read_sql(query, conn)
print(df)

print("\n=== Sample ProductCarBrand ===")
query = "SELECT TOP 5 * FROM dbo.ProductCarBrand"
df = pd.read_sql(query, conn)
print(df)

print("\n=== Sample ProductProductGroup with Group Name ===")
query = """
SELECT TOP 10
    ppg.ProductID,
    pg.ID as group_id,
    pg.Name as group_name,
    pg.IsSubGroup
FROM dbo.ProductProductGroup ppg
INNER JOIN dbo.ProductGroup pg ON ppg.ProductGroupID = pg.ID
WHERE pg.Deleted = 0
"""
df = pd.read_sql(query, conn)
print(df)

conn.close()
