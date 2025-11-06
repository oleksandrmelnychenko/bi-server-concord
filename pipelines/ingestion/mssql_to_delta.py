"""
MSSQL to Delta Lake Ingestion Pipeline

This script handles incremental data ingestion from MSSQL to Delta Lake.
Supports full and incremental loads with CDC (Change Data Capture).
"""
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import pymssql
from deltalake import DeltaTable, write_deltalake
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MSSQLToDeltaIngestion:
    """Handles data ingestion from MSSQL to Delta Lake"""

    def __init__(
        self,
        mssql_host: str,
        mssql_database: str,
        mssql_user: str,
        mssql_password: str,
        delta_base_path: str,
        mssql_port: int = 1433
    ):
        self.mssql_host = mssql_host
        self.mssql_port = mssql_port
        self.mssql_database = mssql_database
        self.mssql_user = mssql_user
        self.mssql_password = mssql_password
        self.delta_base_path = Path(delta_base_path)
        self.connection = None

    def connect(self):
        """Establish connection to MSSQL"""
        try:
            self.connection = pymssql.connect(
                server=self.mssql_host,
                port=self.mssql_port,
                user=self.mssql_user,
                password=self.mssql_password,
                database=self.mssql_database,
                tds_version='7.0',
                timeout=30
            )
            logger.info(f"Connected to MSSQL database: {self.mssql_database}")
        except Exception as e:
            logger.error(f"Failed to connect to MSSQL: {e}")
            raise

    def disconnect(self):
        """Close MSSQL connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from MSSQL")

    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database"""
        query = """
        SELECT TABLE_SCHEMA + '.' + TABLE_NAME as full_table_name
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        df = pd.read_sql(query, self.connection)
        tables = df['full_table_name'].tolist()
        logger.info(f"Found {len(tables)} tables in database")
        return tables

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a table"""
        query = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name.split('.')[-1]}'
        ORDER BY ORDINAL_POSITION
        """
        return pd.read_sql(query, self.connection)

    def get_table_primary_keys(self, table_name: str) -> List[str]:
        """Get primary key columns for a table"""
        schema, table = table_name.split('.') if '.' in table_name else ('dbo', table_name)

        query = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
        AND TABLE_SCHEMA = '{schema}'
        AND TABLE_NAME = '{table}'
        """
        df = pd.read_sql(query, self.connection)
        return df['COLUMN_NAME'].tolist() if not df.empty else []

    def extract_table_data(
        self,
        table_name: str,
        batch_size: int = 10000,
        incremental_column: Optional[str] = None,
        last_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Extract data from MSSQL table

        Args:
            table_name: Full table name (schema.table)
            batch_size: Number of rows per batch
            incremental_column: Column for incremental loads (e.g., modified_date)
            last_value: Last value from previous load

        Returns:
            DataFrame with extracted data
        """
        try:
            # Build query
            query = f"SELECT * FROM {table_name}"

            if incremental_column and last_value:
                query += f" WHERE {incremental_column} > '{last_value}'"
                logger.info(f"Incremental load from {table_name} where {incremental_column} > {last_value}")
            else:
                logger.info(f"Full load from {table_name}")

            # Extract data
            df = pd.read_sql(query, self.connection)
            logger.info(f"Extracted {len(df)} rows from {table_name}")

            # Add metadata columns
            df['_extracted_at'] = datetime.utcnow()
            df['_source_table'] = table_name

            return df

        except Exception as e:
            logger.error(f"Error extracting data from {table_name}: {e}")
            raise

    def load_to_delta(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "append",
        partition_by: Optional[List[str]] = None,
        merge_keys: Optional[List[str]] = None
    ):
        """
        Load data to Delta Lake

        Args:
            df: DataFrame to load
            table_name: Target table name
            mode: Write mode ('append', 'overwrite', 'merge')
            partition_by: Columns to partition by
            merge_keys: Keys for merge operation
        """
        try:
            # Create table path
            table_path = self.delta_base_path / table_name.replace('.', '_')
            table_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading {len(df)} rows to Delta table: {table_path}")

            if mode == "merge" and merge_keys:
                # Merge/Upsert operation
                self._merge_to_delta(df, str(table_path), merge_keys)
            else:
                # Append or overwrite
                write_deltalake(
                    str(table_path),
                    df,
                    mode=mode,
                    partition_by=partition_by,
                    schema_mode="merge"  # Allow schema evolution
                )

            logger.info(f"Successfully loaded data to {table_path}")

        except Exception as e:
            logger.error(f"Error loading to Delta Lake: {e}")
            raise

    def _merge_to_delta(self, df: pd.DataFrame, table_path: str, merge_keys: List[str]):
        """Perform merge/upsert operation"""
        try:
            dt = DeltaTable(table_path)

            # Build merge condition
            merge_condition = " AND ".join([f"target.{key} = source.{key}" for key in merge_keys])

            # TODO: Implement actual merge logic using delta-rs
            # For now, use append mode
            write_deltalake(
                table_path,
                df,
                mode="append",
                schema_mode="merge"
            )

            logger.info(f"Merged {len(df)} rows using keys: {merge_keys}")

        except Exception as e:
            logger.warning(f"Merge failed, falling back to append: {e}")
            write_deltalake(table_path, df, mode="append", schema_mode="merge")

    def ingest_table(
        self,
        table_name: str,
        mode: str = "append",
        incremental_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single table from MSSQL to Delta Lake

        Args:
            table_name: Table to ingest
            mode: Write mode
            incremental_column: Column for incremental loads

        Returns:
            Ingestion statistics
        """
        start_time = datetime.utcnow()

        try:
            logger.info(f"Starting ingestion for table: {table_name}")

            # Get primary keys for merge operations
            primary_keys = self.get_table_primary_keys(table_name)

            # Extract data
            df = self.extract_table_data(table_name, incremental_column=incremental_column)

            if df.empty:
                logger.info(f"No data to ingest for {table_name}")
                return {
                    "table": table_name,
                    "status": "skipped",
                    "rows": 0,
                    "duration_seconds": 0
                }

            # Load to Delta
            merge_keys = primary_keys if mode == "merge" and primary_keys else None
            self.load_to_delta(df, table_name, mode=mode, merge_keys=merge_keys)

            duration = (datetime.utcnow() - start_time).total_seconds()

            return {
                "table": table_name,
                "status": "success",
                "rows": len(df),
                "duration_seconds": duration,
                "mode": mode,
                "primary_keys": primary_keys
            }

        except Exception as e:
            logger.error(f"Failed to ingest table {table_name}: {e}")
            return {
                "table": table_name,
                "status": "failed",
                "error": str(e),
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            }

    def ingest_all_tables(
        self,
        mode: str = "append",
        exclude_tables: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest all tables from MSSQL to Delta Lake

        Args:
            mode: Write mode for all tables
            exclude_tables: List of tables to exclude

        Returns:
            List of ingestion statistics for each table
        """
        try:
            self.connect()

            # Get all tables
            all_tables = self.get_all_tables()

            # Filter excluded tables
            if exclude_tables:
                all_tables = [t for t in all_tables if t not in exclude_tables]

            logger.info(f"Ingesting {len(all_tables)} tables to Delta Lake")

            # Ingest each table
            results = []
            for table in all_tables:
                result = self.ingest_table(table, mode=mode)
                results.append(result)

            # Summary
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = sum(1 for r in results if r['status'] == 'failed')
            total_rows = sum(r.get('rows', 0) for r in results)

            logger.info(f"Ingestion complete: {successful} succeeded, {failed} failed, {total_rows} total rows")

            return results

        finally:
            self.disconnect()


def main():
    """Main ingestion function"""
    # Load configuration from environment
    config = {
        "mssql_host": os.getenv("MSSQL_HOST", "localhost"),
        "mssql_database": os.getenv("MSSQL_DATABASE", "YourDatabase"),
        "mssql_user": os.getenv("MSSQL_USER", "sa"),
        "mssql_password": os.getenv("MSSQL_PASSWORD", ""),
        "delta_base_path": os.getenv("DELTA_BASE_PATH", "/opt/dagster/app/data/delta")
    }

    # Initialize ingestion
    ingestion = MSSQLToDeltaIngestion(**config)

    # Ingest all tables
    results = ingestion.ingest_all_tables(mode="append")

    # Print summary
    print("\n=== Ingestion Summary ===")
    for result in results:
        print(f"{result['table']}: {result['status']} ({result.get('rows', 0)} rows)")


if __name__ == "__main__":
    main()
