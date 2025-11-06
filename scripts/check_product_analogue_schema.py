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

print("\n=== ProductAnalogue Schema ===")
query = """
SELECT
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'ProductAnalogue'
ORDER BY ORDINAL_POSITION
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))

print("\n=== Sample ProductAnalogue ===")
query2 = "SELECT TOP 5 * FROM dbo.ProductAnalogue"
df2 = pd.read_sql(query2, conn)
print(df2)

conn.close()
