#!/usr/bin/env python3
"""
List All Tables in Database
"""

import os
import pymssql

# MSSQL Connection
MSSQL_CONFIG = {
    "host": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": os.getenv("MSSQL_PORT", "1433"),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb_v5"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD", "Grimm_jow92"),
}

def main():
    print("🔌 Connecting to database...")
    conn = pymssql.connect(
        server=MSSQL_CONFIG['host'],
        port=int(MSSQL_CONFIG['port']),
        user=MSSQL_CONFIG['user'],
        password=MSSQL_CONFIG['password'],
        database=MSSQL_CONFIG['database'],
        tds_version='7.0'
    )
    print("✓ Connected!\n")

    cursor = conn.cursor()

    # Get all tables
    query = """
    SELECT
        t.TABLE_SCHEMA,
        t.TABLE_NAME,
        p.rows as ROW_COUNT
    FROM INFORMATION_SCHEMA.TABLES t
    LEFT JOIN sys.partitions p ON OBJECT_ID(t.TABLE_SCHEMA + '.' + t.TABLE_NAME) = p.object_id
    WHERE t.TABLE_TYPE = 'BASE TABLE'
        AND p.index_id IN (0, 1)
    ORDER BY p.rows DESC, t.TABLE_SCHEMA, t.TABLE_NAME
    """

    cursor.execute(query)
    tables = cursor.fetchall()

    print("=" * 100)
    print("ALL TABLES IN DATABASE (Sorted by row count)")
    print("=" * 100)
    print(f"{'Schema':<15} {'Table Name':<50} {'Row Count':>15}")
    print("-" * 100)

    total_rows = 0
    total_tables = 0

    for schema, table, row_count in tables:
        if row_count is not None:
            print(f"{schema:<15} {table:<50} {row_count:>15,}")
            total_rows += row_count
            total_tables += 1

    print("-" * 100)
    print(f"{'TOTALS:':<15} {total_tables} tables {total_rows:>29,} rows")
    print("=" * 100)

    # Show top 20 tables with most data
    print("\n\n📊 TOP 20 LARGEST TABLES")
    print("-" * 100)
    cursor.execute(query + " OFFSET 0 ROWS FETCH NEXT 20 ROWS ONLY")
    top_tables = cursor.fetchall()

    for i, (schema, table, row_count) in enumerate(top_tables, 1):
        if row_count:
            print(f"{i:2}. [{schema}].[{table}] - {row_count:,} rows")

    cursor.close()
    conn.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
