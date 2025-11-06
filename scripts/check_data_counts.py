#!/usr/bin/env python3
"""
Quick Data Count Check
Checks row counts in key tables to understand current data volume
"""

import os
import pymssql
from datetime import datetime

# MSSQL Connection
MSSQL_CONFIG = {
    "host": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": os.getenv("MSSQL_PORT", "1433"),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb_v5"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD", "Grimm_jow92"),
}

def connect_db():
    """Connect to MSSQL"""
    return pymssql.connect(
        server=MSSQL_CONFIG['host'],
        port=int(MSSQL_CONFIG['port']),
        user=MSSQL_CONFIG['user'],
        password=MSSQL_CONFIG['password'],
        database=MSSQL_CONFIG['database'],
        tds_version='7.0'
    )

def check_table_count(conn, schema, table):
    """Get row count for a table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM [{schema}].[{table}]")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 80)
    print("CONCORD DATABASE - DATA COUNT CHECK")
    print("=" * 80)
    print(f"Database: {MSSQL_CONFIG['database']}")
    print(f"Server: {MSSQL_CONFIG['host']}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Connect
    print("\n🔌 Connecting to database...")
    conn = connect_db()
    print("✓ Connected!\n")

    # Key tables to check
    tables_to_check = [
        # Customer tables
        ("dbo", "Customers", "Main customer records"),
        ("dbo", "CustomerAddresses", "Customer addresses"),
        ("dbo", "CustomerPrices", "Customer-specific pricing"),

        # Product tables
        ("dbo", "Products", "Main product catalog"),
        ("dbo", "ProductAnalogues", "Product relationships/alternatives"),
        ("dbo", "ProductPrices", "Product pricing history"),
        ("dbo", "ProductCarCompatibilities", "Car compatibility data"),

        # Order/Transaction tables
        ("dbo", "Orders", "Customer orders"),
        ("dbo", "OrderItems", "Order line items"),
        ("dbo", "SupplyOrders", "Supply/procurement orders"),
        ("dbo", "SupplyOrderItems", "Supply order items"),

        # Sales data
        ("dbo", "Sales", "Sales transactions"),
        ("dbo", "SaleItems", "Sale line items"),

        # Inventory
        ("dbo", "Warehouses", "Warehouse locations"),
        ("dbo", "WarehouseStocks", "Current inventory"),
    ]

    print("📊 TABLE DATA COUNTS")
    print("-" * 80)
    print(f"{'Schema':<10} {'Table':<30} {'Count':<15} {'Description':<25}")
    print("-" * 80)

    total_rows = 0

    for schema, table, description in tables_to_check:
        count = check_table_count(conn, schema, table)
        if isinstance(count, int):
            print(f"{schema:<10} {table:<30} {count:>14,} {description:<25}")
            total_rows += count
        else:
            print(f"{schema:<10} {table:<30} {'N/A':>14} {description:<25}")

    print("-" * 80)
    print(f"{'TOTAL ROWS COUNTED:':<40} {total_rows:>14,}")
    print("-" * 80)

    # Additional analysis - date ranges for key tables
    print("\n📅 DATE RANGES (Recent Activity)")
    print("-" * 80)

    date_queries = [
        ("Orders", "SELECT MIN(OrderDate) as MinDate, MAX(OrderDate) as MaxDate, COUNT(*) as Total FROM dbo.Orders"),
        ("Sales", "SELECT MIN(SaleDate) as MinDate, MAX(SaleDate) as MaxDate, COUNT(*) as Total FROM dbo.Sales"),
        ("Products", "SELECT MIN(CreatedAt) as MinDate, MAX(UpdatedAt) as MaxDate, COUNT(*) as Total FROM dbo.Products WHERE CreatedAt IS NOT NULL"),
    ]

    cursor = conn.cursor()
    for name, query in date_queries:
        try:
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                print(f"\n{name}:")
                print(f"  First Record: {result[0]}")
                print(f"  Last Record:  {result[1]}")
                print(f"  Total:        {result[2]:,}")
        except Exception as e:
            print(f"\n{name}: Unable to check ({e})")

    cursor.close()
    conn.close()

    print("\n" + "=" * 80)
    print("✅ DATA COUNT CHECK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
