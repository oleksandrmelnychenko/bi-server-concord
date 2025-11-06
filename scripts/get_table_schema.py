#!/usr/bin/env python3
"""
Get Table Schemas
"""

import os
import pymssql

MSSQL_CONFIG = {
    "host": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": os.getenv("MSSQL_PORT", "1433"),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb_v5"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD", "Grimm_jow92"),
}

def get_columns(conn, table_name):
    """Get columns for a table"""
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
    """)
    columns = cursor.fetchall()
    cursor.close()
    return columns

def main():
    conn = pymssql.connect(
        server=MSSQL_CONFIG['host'],
        port=int(MSSQL_CONFIG['port']),
        user=MSSQL_CONFIG['user'],
        password=MSSQL_CONFIG['password'],
        database=MSSQL_CONFIG['database'],
        tds_version='7.0'
    )

    tables = ['Client', 'Product', 'Order', 'OrderItem', 'Sale', 'ProductAnalogue', 'ProductPricing']

    for table in tables:
        print(f"\n{'=' * 80}")
        print(f"TABLE: {table}")
        print('=' * 80)
        columns = get_columns(conn, table)
        if columns:
            print(f"{'Column Name':<40} {'Data Type':<20} {'Max Length':<10}")
            print('-' * 80)
            for col_name, data_type, max_len in columns:
                max_len_str = str(max_len) if max_len else ''
                print(f"{col_name:<40} {data_type:<20} {max_len_str:<10}")
        else:
            print("No columns found or table doesn't exist")

    conn.close()

if __name__ == "__main__":
    main()
