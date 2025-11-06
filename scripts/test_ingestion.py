#!/usr/bin/env python3
"""
Test ingestion - Ingest just the Client table to verify everything works
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pymssql
from deltalake import write_deltalake
from datetime import datetime

# Configuration
MSSQL_CONFIG = {
    "host": "78.152.175.67",
    "port": 1433,
    "database": "ConcordDb_v5",
    "user": "ef_migrator",
    "password": "Grimm_jow92",
}

DELTA_BASE_PATH = "/opt/dagster/app/data/delta"

def test_connection():
    """Test MSSQL connection"""
    print("Testing MSSQL connection...")
    try:
        conn = pymssql.connect(
            server=MSSQL_CONFIG['host'],
            port=MSSQL_CONFIG['port'],
            user=MSSQL_CONFIG['user'],
            password=MSSQL_CONFIG['password'],
            database=MSSQL_CONFIG['database'],
            tds_version='7.0'
        )
        print("✓ Connection successful!")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_query():
    """Test simple query"""
    print("\nTesting query execution...")
    try:
        conn = pymssql.connect(
            server=MSSQL_CONFIG['host'],
            port=MSSQL_CONFIG['port'],
            user=MSSQL_CONFIG['user'],
            password=MSSQL_CONFIG['password'],
            database=MSSQL_CONFIG['database'],
            tds_version='7.0'
        )

        query = "SELECT COUNT(*) as cnt FROM dbo.Client"
        df = pd.read_sql(query, conn)
        count = int(df['cnt'].iloc[0])
        print(f"✓ Query successful! Client table has {count:,} rows")

        conn.close()
        return True
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False

def test_extraction():
    """Test data extraction"""
    print("\nTesting data extraction...")
    try:
        conn = pymssql.connect(
            server=MSSQL_CONFIG['host'],
            port=MSSQL_CONFIG['port'],
            user=MSSQL_CONFIG['user'],
            password=MSSQL_CONFIG['password'],
            database=MSSQL_CONFIG['database'],
            tds_version='7.0'
        )

        query = "SELECT TOP 10 * FROM dbo.Client"
        df = pd.read_sql(query, conn)
        print(f"✓ Extracted {len(df)} sample rows")
        print(f"  Columns: {list(df.columns[:5])}... ({len(df.columns)} total)")

        conn.close()
        return df
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return None

def test_delta_write(df):
    """Test writing to Delta Lake"""
    print("\nTesting Delta Lake write...")
    try:
        import pyarrow as pa
        import uuid

        # Add metadata
        df['_ingested_at'] = datetime.now()
        df['_source_table'] = 'dbo.Client'

        # Convert UUID columns to strings (Delta Lake doesn't support UUID type)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if first non-null value is UUID
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, uuid.UUID):
                    print(f"  Converting UUID column: {col}")
                    df[col] = df[col].astype(str)

        # Drop columns that are all NULL (Delta Lake doesn't like these)
        null_cols = [col for col in df.columns if df[col].isna().all()]
        if null_cols:
            print(f"  Dropping {len(null_cols)} NULL-only columns: {null_cols[:3]}...")
            df = df.drop(columns=null_cols)

        # Create path
        delta_path = Path(DELTA_BASE_PATH) / "test" / "dbo_Client_test"
        delta_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to Arrow table (with safe mode to handle null columns)
        table = pa.Table.from_pandas(df, safe=False)

        # Write
        write_deltalake(
            str(delta_path),
            table,
            mode="overwrite",
            schema_mode="overwrite"
        )

        print(f"✓ Written to Delta Lake: {delta_path}")
        print(f"  Path exists: {delta_path.exists()}")

        # List files
        if delta_path.exists():
            files = list(delta_path.glob("*"))
            print(f"  Files created: {len(files)}")
            for f in files[:3]:
                print(f"    - {f.name}")

        return True
    except Exception as e:
        print(f"✗ Delta write failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("  INGESTION TEST - ConcordDb_v5")
    print("="*70 + "\n")

    # Test 1: Connection
    if not test_connection():
        print("\n❌ Connection test failed. Cannot proceed.")
        return 1

    # Test 2: Query
    if not test_query():
        print("\n❌ Query test failed. Cannot proceed.")
        return 1

    # Test 3: Extraction
    df = test_extraction()
    if df is None:
        print("\n❌ Extraction test failed. Cannot proceed.")
        return 1

    # Test 4: Delta Lake write
    if not test_delta_write(df):
        print("\n❌ Delta write test failed.")
        return 1

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nYou can now run the full ingestion:")
    print("  python scripts/ingest_ml_tables.py --yes")

    return 0

if __name__ == "__main__":
    sys.exit(main())
