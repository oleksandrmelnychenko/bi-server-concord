#!/usr/bin/env python3
"""
V3.6 Data Mining - Extract Seasonal Purchase Patterns

This script analyzes historical purchase data to identify products with
strong seasonal demand patterns.

Goal: Find products like "winter tires in Nov-Feb" or "air filters in Spring"
      to enable seasonality-based recommendations in V3.6.

Output: seasonal_patterns.csv with columns:
    - ProductID
    - ProductName
    - month (1-12)
    - purchase_count
    - avg_monthly_purchases
    - seasonal_index (ratio to average)
    - peak_months (comma-separated)
    - trough_months (comma-separated)

Seasonal index >1.5 = strong seasonal demand for that month!
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

def extract_seasonal_patterns():
    """
    Extract products with seasonal purchase patterns
    """
    print(f"[{datetime.now()}] V3.6 Data Mining: Seasonal Purchase Patterns")
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

    # SQL query to analyze monthly purchase patterns
    query = """
    -- V3.6: Extract seasonal purchase patterns
    WITH monthly_purchases AS (
        SELECT
            oi.ProductID,
            MONTH(o.Created) as purchase_month,
            COUNT(*) as monthly_count
        FROM dbo.OrderItem oi
        INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
        WHERE o.Created >= DATEADD(YEAR, -3, GETDATE())  -- Last 3 years for seasonality
        GROUP BY oi.ProductID, MONTH(o.Created)
    ),
    product_stats AS (
        SELECT
            ProductID,
            AVG(monthly_count * 1.0) as avg_monthly_purchases,
            SUM(monthly_count) as total_purchases
        FROM monthly_purchases
        GROUP BY ProductID
        HAVING SUM(monthly_count) >= 50  -- At least 50 total purchases
    )
    SELECT
        mp.ProductID,
        p.Name as ProductName,
        mp.purchase_month as month,
        mp.monthly_count as purchase_count,
        ps.avg_monthly_purchases,
        ps.total_purchases,
        -- Seasonal index: ratio of month purchases to average
        CAST(mp.monthly_count AS FLOAT) / ps.avg_monthly_purchases as seasonal_index
    FROM monthly_purchases mp
    INNER JOIN product_stats ps ON mp.ProductID = ps.ProductID
    INNER JOIN dbo.Product p ON mp.ProductID = p.ID
    ORDER BY mp.ProductID, mp.purchase_month;
    """

    print("Executing seasonal pattern analysis query...")
    print("(This may take 2-4 minutes on large datasets)\n")

    cursor.execute(query)
    results = cursor.fetchall()

    print(f"✓ Found {len(results)} product-month observations!\n")

    # Process results to identify products with seasonal patterns
    products_by_id = defaultdict(lambda: {
        'name': None,
        'monthly_data': {},
        'total_purchases': 0,
        'avg_monthly': 0
    })

    for row in results:
        pid = row['ProductID']
        products_by_id[pid]['name'] = row['ProductName']
        products_by_id[pid]['monthly_data'][row['month']] = {
            'count': row['purchase_count'],
            'index': row['seasonal_index']
        }
        products_by_id[pid]['total_purchases'] = row['total_purchases']
        products_by_id[pid]['avg_monthly'] = row['avg_monthly_purchases']

    # Identify seasonal products (variance in monthly indices)
    seasonal_products = []

    for pid, data in products_by_id.items():
        if len(data['monthly_data']) < 8:  # Need at least 8 months of data
            continue

        indices = [m['index'] for m in data['monthly_data'].values()]
        avg_index = sum(indices) / len(indices)
        variance = sum((idx - avg_index) ** 2 for idx in indices) / len(indices)
        std_dev = variance ** 0.5

        # Strong seasonality = high coefficient of variation
        cv = std_dev / avg_index if avg_index > 0 else 0

        if cv >= 0.30:  # CV >= 30% indicates seasonality
            # Find peak and trough months
            peak_months = [
                month for month, mdata in data['monthly_data'].items()
                if mdata['index'] >= 1.3  # 30% above average
            ]
            trough_months = [
                month for month, mdata in data['monthly_data'].items()
                if mdata['index'] <= 0.7  # 30% below average
            ]

            seasonal_products.append({
                'ProductID': pid,
                'ProductName': data['name'],
                'total_purchases': data['total_purchases'],
                'avg_monthly_purchases': data['avg_monthly'],
                'coefficient_variation': cv,
                'peak_months': ','.join(map(str, sorted(peak_months))),
                'trough_months': ','.join(map(str, sorted(trough_months))),
                'monthly_data': data['monthly_data']
            })

    # Sort by coefficient of variation (strongest seasonality first)
    seasonal_products.sort(key=lambda x: x['coefficient_variation'], reverse=True)

    print(f"✓ Identified {len(seasonal_products)} products with seasonal patterns!\n")

    # Write detailed monthly data to CSV
    output_file = 'seasonal_patterns.csv'
    print(f"Writing results to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ProductID', 'ProductName', 'month', 'purchase_count',
                      'avg_monthly_purchases', 'seasonal_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for product in seasonal_products:
            for month in range(1, 13):
                if month in product['monthly_data']:
                    writer.writerow({
                        'ProductID': product['ProductID'],
                        'ProductName': product['ProductName'],
                        'month': month,
                        'purchase_count': product['monthly_data'][month]['count'],
                        'avg_monthly_purchases': round(product['avg_monthly_purchases'], 2),
                        'seasonal_index': round(product['monthly_data'][month]['index'], 3)
                    })

    print(f"✓ Saved seasonal data to {output_file}\n")

    # Write summary CSV with peak/trough info
    summary_file = 'seasonal_summary.csv'
    print(f"Writing summary to {summary_file}...")

    with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ProductID', 'ProductName', 'total_purchases',
                      'avg_monthly_purchases', 'coefficient_variation',
                      'peak_months', 'trough_months']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for product in seasonal_products:
            writer.writerow({
                'ProductID': product['ProductID'],
                'ProductName': product['ProductName'],
                'total_purchases': product['total_purchases'],
                'avg_monthly_purchases': round(product['avg_monthly_purchases'], 2),
                'coefficient_variation': round(product['coefficient_variation'], 3),
                'peak_months': product['peak_months'],
                'trough_months': product['trough_months']
            })

    print(f"✓ Saved summary to {summary_file}\n")

    # Display summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if seasonal_products:
        # Calculate statistics
        total_seasonal = len(seasonal_products)
        avg_cv = sum(p['coefficient_variation'] for p in seasonal_products) / len(seasonal_products)

        # Count by seasonality strength
        very_strong = sum(1 for p in seasonal_products if p['coefficient_variation'] >= 0.60)
        strong = sum(1 for p in seasonal_products if 0.40 <= p['coefficient_variation'] < 0.60)
        moderate = sum(1 for p in seasonal_products if 0.30 <= p['coefficient_variation'] < 0.40)

        print(f"\nTotal seasonal products: {total_seasonal}")
        print(f"Average coefficient of variation: {avg_cv:.2f}")

        print(f"\nBreakdown by seasonality strength:")
        print(f"  Very Strong (CV ≥0.60):     {very_strong:4d} products")
        print(f"  Strong (CV 0.40-0.60):      {strong:4d} products")
        print(f"  Moderate (CV 0.30-0.40):    {moderate:4d} products")

        # Top 15 most seasonal products
        print(f"\n" + "=" * 80)
        print("TOP 15 MOST SEASONAL PRODUCTS (Best candidates for seasonal scoring)")
        print("=" * 80)
        print(f"{'Product Name':<40} {'CV':<8} {'Peak Mos':<12} {'Trough Mos':<12}")
        print("-" * 80)

        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for i, product in enumerate(seasonal_products[:15], 1):
            name = product['ProductName'][:38]
            cv = f"{product['coefficient_variation']:.3f}"

            # Convert month numbers to names
            peak_nums = product['peak_months'].split(',') if product['peak_months'] else []
            peak_names = ','.join(month_names[int(m)] for m in peak_nums if m)

            trough_nums = product['trough_months'].split(',') if product['trough_months'] else []
            trough_names = ','.join(month_names[int(m)] for m in trough_nums if m)

            print(f"{name:<40} {cv:<8} {peak_names:<12} {trough_names:<12}")

    cursor.close()
    conn.close()

    print(f"\n{'=' * 80}")
    print(f"[{datetime.now()}] Analysis complete!")
    print(f"Next step: Generate Week 1 summary report")
    print(f"{'=' * 80}\n")

    return len(seasonal_products)

if __name__ == "__main__":
    try:
        count = extract_seasonal_patterns()
        print(f"\n✓ SUCCESS: Identified {count} seasonal products")
        print(f"✓ Data saved to: seasonal_patterns.csv and seasonal_summary.csv")
        print(f"✓ Ready for V3.6 seasonal scoring implementation\n")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        exit(1)
