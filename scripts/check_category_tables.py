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

# Check Category table
print("\n=== Category Table Schema ===")
query = """
SELECT
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'Category'
ORDER BY ORDINAL_POSITION
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

print("\n=== Sample Categories ===")
query2 = "SELECT TOP 10 * FROM dbo.Category"
df2 = pd.read_sql(query2, conn)
print(df2)

# Check ProductCategory count
print("\n=== ProductCategory Count ===")
query3 = "SELECT COUNT(*) as count FROM dbo.ProductCategory"
df3 = pd.read_sql(query3, conn)
print(f"Total ProductCategory records: {df3['count'].iloc[0]}")

# Check ProductGroup
print("\n=== ProductGroup Table Schema ===")
query4 = """
SELECT
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'ProductGroup'
ORDER BY ORDINAL_POSITION
"""
df4 = pd.read_sql(query4, conn)
print(df4.to_string(index=False))

print("\n=== Sample ProductGroups ===")
query5 = "SELECT TOP 10 * FROM dbo.ProductGroup WHERE Deleted = 0"
df5 = pd.read_sql(query5, conn)
print(df5)

# Check ProductProductGroup count
print("\n=== ProductProductGroup Count ===")
query6 = "SELECT COUNT(*) as count FROM dbo.ProductProductGroup"
df6 = pd.read_sql(query6, conn)
print(f"Total ProductProductGroup records: {df6['count'].iloc[0]}")

conn.close()
