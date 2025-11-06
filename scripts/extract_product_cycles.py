#!/usr/bin/env python3
"""
V3.6 Data Mining - Extract Product Repurchase Cycles

This script analyzes historical purchase data to identify products with
predictable maintenance/repurchase cycles.

Goal: Find products like "oil filters purchased every ~30 days" to enable
      urgency-based recommendations in V3.6.

Output: product_cycles.csv with columns:
    - ProductID
    - ProductName
    - avg_cycle_days
    - cycle_stdev
    - coefficient_variation (CV = stdev / avg)
    - repurchase_count

Low CV (<0.5) = predictable maintenance cycle = GOLD for recommendations!
"""

import pymssql
import os
from dotenv import load_dotenv
import csv
from datetime import datetime

# Load environment variables
load_dotenv()

DB_SERVER = os.getenv('MSSQL_HOST')
DB_PORT = int(os.getenv('MSSQL_PORT', 1433))
DB_NAME = os.getenv('MSSQL_DATABASE')
DB_USER = os.getenv('MSSQL_USER')
DB_PASSWORD = os.getenv('MSSQL_PASSWORD')

def extract_product_cycles():
    """
    Extract products with predictable repurchase cycles
    """
    print(f"[{datetime.now()}] V3.6 Data Mining: Product Repurchase Cycles")
    print("=" * 80)

    # Connect to database
    print(f"\nConnecting to {DB_SERVER}...")
    conn = pymssql.connect(
        server=DB_SERVER,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    cursor = conn.cursor(as_dict=True)

    print("✓ Connected successfully\n")

    # SQL query to find repurchase intervals
    query = """
    -- V3.6: Extract product repurchase cycles
    WITH repurchase_intervals AS (
        SELECT
            oi1.ProductID,
            ca.ClientID,
            DATEDIFF(day, o1.Created, o2.Created) as days_between_purchases
        FROM dbo.OrderItem oi1
        INNER JOIN dbo.[Order] o1 ON oi1.OrderID = o1.ID
        INNER JOIN dbo.ClientAgreement ca ON o1.ClientAgreementID = ca.ID
        INNER JOIN dbo.[Order] o2 ON ca.ID = o2.ClientAgreementID
        INNER JOIN dbo.OrderItem oi2 ON o2.ID = oi2.OrderID AND oi1.ProductID = oi2.ProductID
        WHERE o2.Created > o1.Created
          AND DATEDIFF(day, o1.Created, o2.Created) BETWEEN 7 AND 365  -- 1 week to 1 year
    ),
    product_stats AS (
        SELECT
            ProductID,
            COUNT(*) as repurchase_count,
            AVG(days_between_purchases * 1.0) as avg_cycle_days,
            STDEV(days_between_purchases * 1.0) as cycle_stdev,
            COUNT(DISTINCT ClientID) as customer_count
        FROM repurchase_intervals
        GROUP BY ProductID
        HAVING COUNT(*) >= 10  -- At least 10 repurchases
    )
    SELECT
        ps.ProductID,
        p.Name as ProductName,
        ps.avg_cycle_days,
        ps.cycle_stdev,
        CASE
            WHEN ps.avg_cycle_days > 0
            THEN ps.cycle_stdev / ps.avg_cycle_days
            ELSE NULL
        END as coefficient_variation,
        ps.repurchase_count,
        ps.customer_count
    FROM product_stats ps
    INNER JOIN dbo.Product p ON ps.ProductID = p.ID
    WHERE ps.avg_cycle_days > 0
      AND ps.cycle_stdev / ps.avg_cycle_days < 0.5  -- CV < 0.5 = predictable
    ORDER BY ps.repurchase_count DESC;
    """

    print("Executing cycle analysis query...")
    print("(This may take 2-5 minutes on large datasets)\n")

    cursor.execute(query)
    results = cursor.fetchall()

    print(f"✓ Found {len(results)} products with predictable cycles!\n")

    # Write to CSV
    output_file = 'product_cycles.csv'
    print(f"Writing results to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"✓ Saved {len(results)} products to {output_file}\n")

    # Display summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if results:
        # Calculate statistics
        total_products = len(results)
        total_repurchases = sum(r['repurchase_count'] for r in results)
        avg_cycle = sum(r['avg_cycle_days'] for r in results) / len(results)

        # Count by cycle range
        weekly = sum(1 for r in results if r['avg_cycle_days'] <= 14)
        biweekly = sum(1 for r in results if 14 < r['avg_cycle_days'] <= 28)
        monthly = sum(1 for r in results if 28 < r['avg_cycle_days'] <= 45)
        quarterly = sum(1 for r in results if 45 < r['avg_cycle_days'] <= 120)
        biannual = sum(1 for r in results if r['avg_cycle_days'] > 120)

        print(f"\nTotal predictable products: {total_products}")
        print(f"Total repurchase observations: {total_repurchases:,}")
        print(f"Average cycle: {avg_cycle:.1f} days")
        print(f"\nBreakdown by cycle frequency:")
        print(f"  Weekly (≤14 days):         {weekly:4d} products")
        print(f"  Bi-weekly (15-28 days):    {biweekly:4d} products")
        print(f"  Monthly (29-45 days):      {monthly:4d} products")
        print(f"  Quarterly (46-120 days):   {quarterly:4d} products")
        print(f"  Bi-annual (>120 days):     {biannual:4d} products")

        # Top 10 most predictable products
        print(f"\n" + "=" * 80)
        print("TOP 10 MOST FREQUENT REPURCHASES (Best candidates for urgency scoring)")
        print("=" * 80)
        print(f"{'ProductID':<12} {'Avg Cycle':<12} {'CV':<8} {'Count':<8} {'Product Name':<40}")
        print("-" * 80)

        for i, product in enumerate(results[:10], 1):
            product_id = str(product['ProductID'])
            cycle = f"{product['avg_cycle_days']:.1f} days"
            cv = f"{product['coefficient_variation']:.3f}"
            count = str(product['repurchase_count'])
            name = product['ProductName'][:40]

            print(f"{product_id:<12} {cycle:<12} {cv:<8} {count:<8} {name:<40}")

    cursor.close()
    conn.close()

    print(f"\n{'=' * 80}")
    print(f"[{datetime.now()}] Analysis complete!")
    print(f"Next step: Run extract_product_associations.py")
    print(f"{'=' * 80}\n")

    return len(results)

if __name__ == "__main__":
    try:
        count = extract_product_cycles()
        print(f"\n✓ SUCCESS: Extracted {count} predictable products")
        print(f"✓ Data saved to: product_cycles.csv")
        print(f"✓ Ready for V3.6 urgency scoring implementation\n")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        exit(1)
