#!/usr/bin/env python3
"""
Focused Data Ingestion for ML Tables

Ingests the most critical tables for ML models:
- Customer data (Client, ClientAgreement)
- Product data (Product, ProductAnalogue, ProductPricing, ProductCarBrand)
- Transaction data (Order, OrderItem, Sale, SupplyOrder, SupplyOrderItem)

This is faster than ingesting all 311 tables and gives us what we need for ML.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import uuid
import pandas as pd
import pymssql
import pyarrow as pa
from deltalake import write_deltalake

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MSSQL Connection Configuration
MSSQL_CONFIG = {
    "host": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": int(os.getenv("MSSQL_PORT", "1433")),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb_v5"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD", "Grimm_jow92"),
}

# Delta Lake base path (MinIO or local)
DELTA_BASE_PATH = os.getenv("DELTA_BASE_PATH", "/opt/dagster/app/data/delta")

# ML-critical tables to ingest (in order of priority)
ML_TABLES = {
    "customer": [
        "dbo.Client",
        "dbo.ClientAgreement",
        "dbo.ClientBankDetails",
    ],
    "product": [
        "dbo.Product",
        "dbo.ProductAnalogue",
        "dbo.ProductPricing",
        "dbo.ProductCarBrand",
        "dbo.ProductProductGroup",
        "dbo.ProductOriginalNumber",
    ],
    "transaction": [
        "dbo.Order",
        "dbo.OrderItem",
        "dbo.Sale",
        "dbo.SupplyOrder",
        "dbo.SupplyOrderItem",
        "dbo.SaleInvoiceNumber",
    ],
}


class MLTableIngestion:
    def __init__(self):
        self.conn = None
        self.stats = {
            "total_tables": 0,
            "successful": 0,
            "failed": 0,
            "total_rows": 0,
            "start_time": datetime.now(),
        }

    def connect(self):
        """Connect to MSSQL"""
        try:
            logger.info(f"Connecting to {MSSQL_CONFIG['database']}@{MSSQL_CONFIG['host']}...")
            self.conn = pymssql.connect(
                server=MSSQL_CONFIG['host'],
                port=MSSQL_CONFIG['port'],
                user=MSSQL_CONFIG['user'],
                password=MSSQL_CONFIG['password'],
                database=MSSQL_CONFIG['database'],
                tds_version='7.0',
                timeout=60
            )
            logger.info("✓ Connected successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from MSSQL"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from MSSQL")

    def get_table_row_count(self, table_name: str) -> int:
        """Get approximate row count for a table"""
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name}"
            df = pd.read_sql(query, self.conn)
            return int(df['cnt'].iloc[0])
        except Exception as e:
            logger.warning(f"Could not get row count for {table_name}: {e}")
            return 0

    def ingest_table(self, table_name: str, category: str) -> bool:
        """
        Ingest a single table from MSSQL to Delta Lake

        Args:
            table_name: Full table name (e.g., "dbo.Client")
            category: Category (customer, product, transaction)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Ingesting: {table_name} ({category})")
            logger.info(f"{'='*70}")

            # Get row count first
            row_count = self.get_table_row_count(table_name)
            logger.info(f"Table has approximately {row_count:,} rows")

            if row_count == 0:
                logger.warning(f"⚠️  Table {table_name} is empty, skipping")
                return True

            # Estimate time based on rows
            estimated_seconds = row_count / 10000  # ~10K rows/second
            if estimated_seconds > 10:
                logger.info(f"Estimated time: {estimated_seconds:.0f} seconds")

            # Extract data
            logger.info("Extracting data from MSSQL...")
            start_extract = datetime.now()

            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.conn)

            extract_duration = (datetime.now() - start_extract).total_seconds()
            logger.info(f"✓ Extracted {len(df):,} rows in {extract_duration:.1f}s")

            # Add metadata
            df['_ingested_at'] = datetime.now()
            df['_source_table'] = table_name

            # Convert UUID columns to strings (Delta Lake doesn't support UUID type)
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if first non-null value is UUID
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if isinstance(sample, uuid.UUID):
                        logger.info(f"  Converting UUID column: {col}")
                        df[col] = df[col].astype(str)

            # Drop columns that are all NULL (Delta Lake doesn't like these)
            null_cols = [col for col in df.columns if df[col].isna().all()]
            if null_cols:
                logger.info(f"  Dropping {len(null_cols)} NULL-only columns")
                df = df.drop(columns=null_cols)

            # Convert to Delta Lake path
            delta_path = Path(DELTA_BASE_PATH) / category / table_name.replace('.', '_')
            delta_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to Arrow table (with safe=False to handle nulls)
            table_arrow = pa.Table.from_pandas(df, safe=False)

            # Write to Delta Lake
            logger.info(f"Writing to Delta Lake: {delta_path}")
            start_write = datetime.now()

            write_deltalake(
                str(delta_path),
                table_arrow,
                mode="overwrite",  # First load is always overwrite
                schema_mode="overwrite"
            )

            write_duration = (datetime.now() - start_write).total_seconds()
            logger.info(f"✓ Written to Delta Lake in {write_duration:.1f}s")

            # Update statistics
            self.stats["successful"] += 1
            self.stats["total_rows"] += len(df)

            total_duration = extract_duration + write_duration
            rows_per_second = len(df) / total_duration if total_duration > 0 else 0

            logger.info(f"✅ SUCCESS: {len(df):,} rows ({rows_per_second:.0f} rows/sec)")

            return True

        except Exception as e:
            logger.error(f"❌ FAILED: {table_name} - {e}")
            import traceback
            traceback.print_exc()
            self.stats["failed"] += 1
            return False

    def ingest_all_ml_tables(self):
        """Ingest all ML-critical tables"""
        logger.info("="*70)
        logger.info("ML TABLE INGESTION - ConcordDb_v5")
        logger.info("="*70)

        if not self.connect():
            logger.error("Failed to connect to database. Exiting.")
            return False

        try:
            # Ingest by category
            for category, tables in ML_TABLES.items():
                logger.info(f"\n🔹 Ingesting {category.upper()} tables ({len(tables)} tables)")

                for table in tables:
                    self.stats["total_tables"] += 1
                    self.ingest_table(table, category)

            # Print summary
            self._print_summary()

            return self.stats["failed"] == 0

        finally:
            self.disconnect()

    def _print_summary(self):
        """Print ingestion summary"""
        duration = (datetime.now() - self.stats["start_time"]).total_seconds()

        logger.info("\n" + "="*70)
        logger.info("INGESTION SUMMARY")
        logger.info("="*70)

        logger.info(f"\n📊 Statistics:")
        logger.info(f"  - Total Tables Attempted: {self.stats['total_tables']}")
        logger.info(f"  - Successful: {self.stats['successful']}")
        logger.info(f"  - Failed: {self.stats['failed']}")
        logger.info(f"  - Total Rows Ingested: {self.stats['total_rows']:,}")
        logger.info(f"  - Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

        if self.stats['total_rows'] > 0:
            rows_per_second = self.stats['total_rows'] / duration
            logger.info(f"  - Average Speed: {rows_per_second:.0f} rows/second")

        logger.info(f"\n📁 Data Location: {DELTA_BASE_PATH}/")
        logger.info(f"  - customer/     (customer tables)")
        logger.info(f"  - product/      (product tables)")
        logger.info(f"  - transaction/  (transaction tables)")

        if self.stats["failed"] == 0:
            logger.info("\n✅ ALL TABLES INGESTED SUCCESSFULLY!")
        else:
            logger.warning(f"\n⚠️  {self.stats['failed']} tables failed")

        logger.info("\n💡 Next Steps:")
        logger.info("  1. Verify data: ls -lh " + DELTA_BASE_PATH)
        logger.info("  2. Check MinIO: http://localhost:9001")
        logger.info("  3. Run dbt transformations")
        logger.info("  4. Train ML models")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("  CONCORD BI SERVER - ML TABLE INGESTION")
    print("="*70 + "\n")

    # Configuration check
    logger.info(f"📝 Configuration:")
    logger.info(f"  - Database: {MSSQL_CONFIG['database']}")
    logger.info(f"  - Host: {MSSQL_CONFIG['host']}:{MSSQL_CONFIG['port']}")
    logger.info(f"  - Delta Path: {DELTA_BASE_PATH}")
    logger.info(f"  - Tables to ingest: {sum(len(tables) for tables in ML_TABLES.values())}")

    # Confirm
    if len(sys.argv) > 1 and sys.argv[1] == "--yes":
        proceed = True
    else:
        response = input("\n⚠️  Proceed with ingestion? (yes/no): ")
        proceed = response.lower() in ['yes', 'y']

    if not proceed:
        logger.info("Ingestion cancelled by user")
        return 1

    # Run ingestion
    ingestion = MLTableIngestion()
    success = ingestion.ingest_all_ml_tables()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
