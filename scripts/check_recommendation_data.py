#!/usr/bin/env python3
"""
Check Recommendation-Relevant Data
Analyzes customer, product, and transaction data for ML recommendations
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

def run_query(conn, query, description):
    """Run query and return results"""
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception as e:
        print(f"  Error in {description}: {e}")
        return None

def main():
    print("=" * 100)
    print("RECOMMENDATION DATA ANALYSIS")
    print("=" * 100)
    print(f"Database: {MSSQL_CONFIG['database']}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # Connect
    print("\n🔌 Connecting...")
    conn = pymssql.connect(
        server=MSSQL_CONFIG['host'],
        port=int(MSSQL_CONFIG['port']),
        user=MSSQL_CONFIG['user'],
        password=MSSQL_CONFIG['password'],
        database=MSSQL_CONFIG['database'],
        tds_version='7.0'
    )
    print("✓ Connected!\n")

    # 1. CUSTOMER DATA
    print("\n" + "=" * 100)
    print("👥 CUSTOMER DATA")
    print("=" * 100)

    # Total clients
    result = run_query(conn, "SELECT COUNT(*) FROM dbo.Client", "Client count")
    if result:
        print(f"Total Clients: {result[0][0]:,}")

    # Client agreements with orders
    result = run_query(conn, """
        SELECT COUNT(DISTINCT ClientAgreementID)
        FROM dbo.[Order]
        WHERE ClientAgreementID IS NOT NULL
    """, "Client agreements with orders")
    if result:
        print(f"Client Agreements with Orders: {result[0][0]:,}")

    # Client agreements with sales
    result = run_query(conn, """
        SELECT COUNT(DISTINCT ClientAgreementID)
        FROM dbo.Sale
        WHERE ClientAgreementID IS NOT NULL
    """, "Client agreements with sales")
    if result:
        print(f"Client Agreements with Sales: {result[0][0]:,}")

    # 2. PRODUCT DATA
    print("\n" + "=" * 100)
    print("📦 PRODUCT DATA")
    print("=" * 100)

    # Total products
    result = run_query(conn, "SELECT COUNT(*) FROM dbo.Product", "Product count")
    if result:
        print(f"Total Products: {result[0][0]:,}")

    # Products with analogues
    result = run_query(conn, """
        SELECT COUNT(DISTINCT BaseProductID)
        FROM dbo.ProductAnalogue
    """, "Products with analogues")
    if result:
        print(f"Products with Analogues: {result[0][0]:,}")

    # Products with pricing
    result = run_query(conn, """
        SELECT COUNT(DISTINCT ProductID)
        FROM dbo.ProductPricing
    """, "Products with pricing")
    if result:
        print(f"Products with Pricing: {result[0][0]:,}")

    # Products in orders
    result = run_query(conn, """
        SELECT COUNT(DISTINCT ProductID)
        FROM dbo.OrderItem
        WHERE ProductID IS NOT NULL
    """, "Products ordered")
    if result:
        print(f"Products Ever Ordered: {result[0][0]:,}")

    # 3. TRANSACTION DATA
    print("\n" + "=" * 100)
    print("💰 TRANSACTION DATA")
    print("=" * 100)

    # Orders
    result = run_query(conn, "SELECT COUNT(*) FROM dbo.[Order]", "Order count")
    if result:
        print(f"Total Orders: {result[0][0]:,}")

    # Order Items
    result = run_query(conn, "SELECT COUNT(*) FROM dbo.OrderItem", "OrderItem count")
    if result:
        print(f"Total Order Items: {result[0][0]:,}")

    # Sales
    result = run_query(conn, "SELECT COUNT(*) FROM dbo.Sale", "Sale count")
    if result:
        print(f"Total Sales: {result[0][0]:,}")

    # Unique agreement-product interactions
    result = run_query(conn, """
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT o.ClientAgreementID, oi.ProductID
            FROM dbo.[Order] o
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.ClientAgreementID IS NOT NULL AND oi.ProductID IS NOT NULL
        ) AS interactions
    """, "Customer-product interactions")
    if result:
        print(f"Unique Client-Product Interactions: {result[0][0]:,}")

    # 4. DATE RANGES
    print("\n" + "=" * 100)
    print("📅 DATE RANGES")
    print("=" * 100)

    # Order dates
    result = run_query(conn, """
        SELECT
            MIN(Created) as FirstOrder,
            MAX(Created) as LastOrder
        FROM dbo.[Order]
        WHERE Created IS NOT NULL
    """, "Order date range")
    if result and result[0][0]:
        print(f"Orders: {result[0][0]} to {result[0][1]}")

    # Sale dates
    result = run_query(conn, """
        SELECT
            MIN(Created) as FirstSale,
            MAX(Created) as LastSale
        FROM dbo.Sale
        WHERE Created IS NOT NULL
    """, "Sale date range")
    if result and result[0][0]:
        print(f"Sales: {result[0][0]} to {result[0][1]}")

    # 5. DATA QUALITY
    print("\n" + "=" * 100)
    print("✅ DATA QUALITY CHECKS")
    print("=" * 100)

    # Orders with client agreements
    result = run_query(conn, """
        SELECT
            COUNT(*) as Total,
            SUM(CASE WHEN ClientAgreementID IS NOT NULL THEN 1 ELSE 0 END) as WithClient,
            CAST(SUM(CASE WHEN ClientAgreementID IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) as Pct
        FROM dbo.[Order]
    """, "Orders with client")
    if result:
        print(f"Orders with Client Agreement: {result[0][1]:,} / {result[0][0]:,} ({result[0][2]}%)")

    # OrderItems with products
    result = run_query(conn, """
        SELECT
            COUNT(*) as Total,
            SUM(CASE WHEN ProductID IS NOT NULL THEN 1 ELSE 0 END) as WithProduct,
            CAST(SUM(CASE WHEN ProductID IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) as Pct
        FROM dbo.OrderItem
    """, "OrderItems with product")
    if result:
        print(f"OrderItems with Product ID: {result[0][1]:,} / {result[0][0]:,} ({result[0][2]}%)")

    # 6. TOP CUSTOMERS
    print("\n" + "=" * 100)
    print("🏆 TOP 10 CUSTOMERS BY ORDER COUNT")
    print("=" * 100)

    result = run_query(conn, """
        SELECT TOP 10
            ca.ID as AgreementID,
            COUNT(DISTINCT o.ID) as OrderCount,
            COUNT(DISTINCT oi.ProductID) as UniqueProducts,
            SUM(oi.Qty) as TotalQty
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        LEFT JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        GROUP BY ca.ID
        ORDER BY COUNT(DISTINCT o.ID) DESC
    """, "Top customers")
    if result:
        print(f"{'Agreement ID':<20} {'Orders':>10} {'Products':>10} {'Total Qty':>15}")
        print("-" * 100)
        for agreement_id, orders, products, qty in result:
            print(f"{str(agreement_id):<20} {orders:>10,} {products:>10,} {qty:>15,.0f}")

    # 7. TOP PRODUCTS
    print("\n" + "=" * 100)
    print("🏆 TOP 10 PRODUCTS BY ORDER COUNT")
    print("=" * 100)

    result = run_query(conn, """
        SELECT TOP 10
            p.ID,
            p.Name,
            COUNT(DISTINCT oi.OrderID) as OrderCount,
            SUM(oi.Qty) as TotalQuantity
        FROM dbo.Product p
        INNER JOIN dbo.OrderItem oi ON p.ID = oi.ProductID
        GROUP BY p.ID, p.Name
        ORDER BY COUNT(DISTINCT oi.OrderID) DESC
    """, "Top products")
    if result:
        print(f"{'Product ID':<15} {'Product Name':<40} {'Orders':>10} {'Quantity':>15}")
        print("-" * 100)
        for product_id, name, orders, quantity in result:
            name_short = name[:37] + '...' if name and len(name) > 40 else (name or 'N/A')
            print(f"{str(product_id):<15} {name_short:<40} {orders:>10,} {quantity:>15,.0f}")

    conn.close()

    print("\n" + "=" * 100)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
