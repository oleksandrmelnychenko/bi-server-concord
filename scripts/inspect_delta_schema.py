#!/usr/bin/env python3
"""
Quick script to inspect Delta Lake table schemas
"""

from deltalake import DeltaTable
from pathlib import Path

DELTA_BASE = "/opt/dagster/app/data/delta"

tables = [
    ("customer", "dbo_Client"),
    ("product", "dbo_Product"),
    ("transaction", "dbo_Sale"),
    ("transaction", "dbo_OrderItem"),
    ("product", "dbo_ProductAnalogue"),
    ("product", "dbo_ProductPricing"),
]

for category, table_name in tables:
    path = Path(DELTA_BASE) / category / table_name
    print(f"\n{'='*70}")
    print(f"{table_name}")
    print(f"{'='*70}")

    try:
        dt = DeltaTable(str(path))
        df = dt.to_pandas()

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nColumns:")
        for col in df.columns[:20]:  # First 20 columns
            print(f"  - {col}: {df[col].dtype}")

        if len(df.columns) > 20:
            print(f"  ... and {len(df.columns) - 20} more columns")

        # Show sample data for key columns
        print(f"\nSample data (first 2 rows):")
        key_cols = [c for c in df.columns if c in ['Id', 'Name', 'ClientId', 'ProductId', 'CreatedDate']][:5]
        if key_cols:
            print(df[key_cols].head(2).to_string(index=False))

    except Exception as e:
        print(f"Error: {e}")
