#!/usr/bin/env python3
"""
V3.6 Data Mining - Extract Product Association Rules (Complementarity)

This script mines product association rules to identify complementary products
that are frequently purchased together.

Goal: Find patterns like "brake pads → brake fluid" to enable
      complementarity-based recommendations in V3.6.

Output: product_associations.csv with columns:
    - ProductA_ID
    - ProductA_Name
    - ProductB_ID
    - ProductB_Name
    - support (% of orders containing both)
    - confidence (% of ProductA orders that also have ProductB)
    - lift (strength of association)
    - co_occurrence_count

High lift (>1.5) + high confidence (>0.3) = strong complementarity!
"""

import pymssql
import os
from dotenv import load_dotenv
import csv
from datetime import datetime
from collections import defaultdict

# Load environment variables
load_dotenv()

DB_SERVER = os.getenv('MSSQL_HOST')
DB_PORT = int(os.getenv('MSSQL_PORT', 1433))
DB_NAME = os.getenv('MSSQL_DATABASE')
DB_USER = os.getenv('MSSQL_USER')
DB_PASSWORD = os.getenv('MSSQL_PASSWORD')

def extract_product_associations():
    """
    Extract product association rules using market basket analysis
    """
    print(f"[{datetime.now()}] V3.6 Data Mining: Product Association Rules")
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

    # SQL query to find product co-occurrences in orders
    query = """
    -- V3.6: Extract product association rules
    WITH product_pairs AS (
        SELECT
            oi1.ProductID as ProductA,
            oi2.ProductID as ProductB,
            COUNT(DISTINCT o.ID) as co_occurrence_count
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi1 ON o.ID = oi1.OrderID
        INNER JOIN dbo.OrderItem oi2 ON o.ID = oi2.OrderID
        WHERE oi1.ProductID < oi2.ProductID  -- Avoid duplicates and self-pairs
          AND o.Created >= DATEADD(YEAR, -2, GETDATE())  -- Last 2 years only
        GROUP BY oi1.ProductID, oi2.ProductID
        HAVING COUNT(DISTINCT o.ID) >= 10  -- At least 10 co-occurrences
    ),
    product_totals AS (
        SELECT
            ProductID,
            COUNT(DISTINCT OrderID) as total_orders
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        WHERE o.Created >= DATEADD(YEAR, -2, GETDATE())
        GROUP BY ProductID
    ),
    total_orders_count AS (
        SELECT COUNT(DISTINCT ID) as total_orders
        FROM dbo.[Order]
        WHERE Created >= DATEADD(YEAR, -2, GETDATE())
    )
    SELECT
        pp.ProductA as ProductA_ID,
        pa.Name as ProductA_Name,
        pp.ProductB as ProductB_ID,
        pb.Name as ProductB_Name,
        pp.co_occurrence_count,
        pta.total_orders as ProductA_total_orders,
        ptb.total_orders as ProductB_total_orders,
        toc.total_orders as total_orders,
        -- Support: P(A and B)
        CAST(pp.co_occurrence_count AS FLOAT) / toc.total_orders as support,
        -- Confidence: P(B|A) = P(A and B) / P(A)
        CAST(pp.co_occurrence_count AS FLOAT) / pta.total_orders as confidence_A_to_B,
        -- Confidence: P(A|B) = P(A and B) / P(B)
        CAST(pp.co_occurrence_count AS FLOAT) / ptb.total_orders as confidence_B_to_A,
        -- Lift: P(A and B) / (P(A) * P(B))
        (CAST(pp.co_occurrence_count AS FLOAT) / toc.total_orders) /
        ((CAST(pta.total_orders AS FLOAT) / toc.total_orders) *
         (CAST(ptb.total_orders AS FLOAT) / toc.total_orders)) as lift
    FROM product_pairs pp
    INNER JOIN dbo.Product pa ON pp.ProductA = pa.ID
    INNER JOIN dbo.Product pb ON pp.ProductB = pb.ID
    INNER JOIN product_totals pta ON pp.ProductA = pta.ProductID
    INNER JOIN product_totals ptb ON pp.ProductB = ptb.ProductID
    CROSS JOIN total_orders_count toc
    WHERE
        -- Filter for strong associations
        CAST(pp.co_occurrence_count AS FLOAT) / pta.total_orders >= 0.20  -- Confidence >= 20%
        AND (CAST(pp.co_occurrence_count AS FLOAT) / toc.total_orders) /
            ((CAST(pta.total_orders AS FLOAT) / toc.total_orders) *
             (CAST(ptb.total_orders AS FLOAT) / toc.total_orders)) >= 1.5  -- Lift >= 1.5
    ORDER BY lift DESC, co_occurrence_count DESC;
    """

    print("Executing association mining query...")
    print("(This may take 3-7 minutes on large datasets)\n")

    cursor.execute(query)
    results = cursor.fetchall()

    print(f"✓ Found {len(results)} strong product associations!\n")

    # Write to CSV
    output_file = 'product_associations.csv'
    print(f"Writing results to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"✓ Saved {len(results)} associations to {output_file}\n")

    # Display summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if results:
        # Calculate statistics
        total_associations = len(results)
        avg_lift = sum(r['lift'] for r in results) / len(results)
        avg_confidence_a_to_b = sum(r['confidence_A_to_B'] for r in results) / len(results)
        avg_co_occurrence = sum(r['co_occurrence_count'] for r in results) / len(results)

        # Count by lift strength
        very_strong = sum(1 for r in results if r['lift'] >= 3.0)
        strong = sum(1 for r in results if 2.0 <= r['lift'] < 3.0)
        moderate = sum(1 for r in results if 1.5 <= r['lift'] < 2.0)

        print(f"\nTotal strong associations: {total_associations}")
        print(f"Average lift: {avg_lift:.2f}")
        print(f"Average confidence: {avg_confidence_a_to_b:.1%}")
        print(f"Average co-occurrence count: {avg_co_occurrence:.1f}")

        print(f"\nBreakdown by association strength:")
        print(f"  Very Strong (lift ≥3.0):     {very_strong:4d} associations")
        print(f"  Strong (lift 2.0-3.0):       {strong:4d} associations")
        print(f"  Moderate (lift 1.5-2.0):     {moderate:4d} associations")

        # Top 15 strongest associations
        print(f"\n" + "=" * 80)
        print("TOP 15 STRONGEST ASSOCIATIONS (Best candidates for complementarity scoring)")
        print("=" * 80)
        print(f"{'Product A':<30} {'Product B':<30} {'Lift':<8} {'Conf':<8} {'Count':<8}")
        print("-" * 80)

        for i, assoc in enumerate(results[:15], 1):
            product_a = assoc['ProductA_Name'][:28]
            product_b = assoc['ProductB_Name'][:28]
            lift = f"{assoc['lift']:.2f}"
            confidence = f"{assoc['confidence_A_to_B']:.1%}"
            count = str(assoc['co_occurrence_count'])

            print(f"{product_a:<30} {product_b:<30} {lift:<8} {confidence:<8} {count:<8}")

    cursor.close()
    conn.close()

    print(f"\n{'=' * 80}")
    print(f"[{datetime.now()}] Analysis complete!")
    print(f"Next step: Run extract_seasonal_patterns.py")
    print(f"{'=' * 80}\n")

    return len(results)

if __name__ == "__main__":
    try:
        count = extract_product_associations()
        print(f"\n✓ SUCCESS: Extracted {count} product associations")
        print(f"✓ Data saved to: product_associations.csv")
        print(f"✓ Ready for V3.6 complementarity scoring implementation\n")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        exit(1)
