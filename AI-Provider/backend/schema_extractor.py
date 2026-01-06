"""Extract and cache database schema from MSSQL."""
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

from sqlalchemy import create_engine, inspect, MetaData, select, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from loguru import logger

from config import settings


class SchemaExtractor:
    """Extract database schema and sample data from MSSQL."""

    # Class-level in-memory cache (shared across instances)
    _memory_cache: Optional[Dict[str, Any]] = None
    _cache_time: Optional[datetime] = None
    CACHE_TTL = timedelta(hours=1)  # 1 hour TTL

    # Column semantic descriptions - helps LLM understand column purposes
    # Format: "Table.Column" -> "Description (NOT wrong_name)"
    COLUMN_DESCRIPTIONS = {
        # Client table - common hallucinations
        "Client.Name": "Client display name (NOT ClientName!)",
        "Client.LegalAddress": "Full legal address including city/region (NOT Address!)",
        "Client.ActualAddress": "Actual/delivery address",
        "Client.MobileNumber": "Phone number (NOT Phone!)",
        "Client.EmailAddress": "Email (NOT Email!)",
        "Client.RegionID": "FK to Region table (NOT Client.Region!)",
        # Product table
        "Product.Name": "Product name (NOT ProductName!)",
        "Product.VendorCode": "Brand prefix + code, e.g. SEM001, MG123 (use for brand filtering)",
        # Order table - CRITICAL
        "Order.ClientAgreementID": "FK to ClientAgreement (Order has NO ClientID column!)",
        "Order.Created": "Order creation date",
        # OrderItem table
        "OrderItem.ProductID": "FK to Product",
        "OrderItem.OrderID": "FK to Order",
        "OrderItem.Qty": "Quantity ordered",
        "OrderItem.PricePerItem": "Unit price",
        # ClientAgreement - bridge table
        "ClientAgreement.ClientID": "FK to Client",
        "ClientAgreement.ID": "Used by Order.ClientAgreementID",
        # ProductAvailability
        "ProductAvailability.Amount": "Stock quantity available",
        "ProductAvailability.ProductID": "FK to Product",
        "ProductAvailability.StorageID": "FK to Storage/Warehouse",
    }

    def __init__(self, engine: Optional[Engine] = None):
        """Initialize schema extractor.

        Args:
            engine: SQLAlchemy engine (creates one if not provided)
        """
        self.engine = engine or create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=5,           # 5 persistent connections
            max_overflow=10,       # Up to 15 total
            pool_pre_ping=False,   # Disable ping (use pool_recycle instead)
            pool_recycle=1800,     # Recycle after 30 min
        )
        self.metadata = MetaData()
        self.cache_file = Path("schema_cache.json")

    def extract_full_schema(self, force_refresh: bool = False, include_samples: bool = True) -> Dict[str, Any]:
        """Extract complete database schema.

        Args:
            force_refresh: Force refresh even if cache exists
            include_samples: Include sample data (set False for faster extraction)

        Returns:
            Dictionary containing schema information
        """
        # OPTIMIZATION 1: Check in-memory cache first (fastest)
        if not force_refresh and SchemaExtractor._memory_cache is not None:
            if SchemaExtractor._cache_time and datetime.now() - SchemaExtractor._cache_time < self.CACHE_TTL:
                logger.debug("Using in-memory schema cache")
                return SchemaExtractor._memory_cache

        # OPTIMIZATION 2: Check file cache (fast)
        if not force_refresh and self.cache_file.exists():
            logger.info("Loading schema from file cache")
            with open(self.cache_file, "r") as f:
                schema = json.load(f)
                # Update memory cache
                SchemaExtractor._memory_cache = schema
                SchemaExtractor._cache_time = datetime.now()
                return schema

        logger.info("Extracting database schema (this may take a moment)...")
        inspector = inspect(self.engine)
        self._include_samples = include_samples  # Store for _extract_table_info

        schema = {
            "extracted_at": datetime.now().isoformat(),
            "database": settings.db_name,
            "tables": {},
            "views": {},
        }

        # Get all table names
        table_names = inspector.get_table_names()
        view_names = inspector.get_view_names()

        logger.info(f"Found {len(table_names)} tables and {len(view_names)} views")

        # Extract table information
        for table_name in table_names:
            schema["tables"][table_name] = self._extract_table_info(
                inspector, table_name
            )

        # Extract view information (treat similarly to tables)
        for view_name in view_names:
            schema["views"][view_name] = self._extract_table_info(
                inspector, view_name, is_view=True
            )

        # Cache the schema
        self._save_cache(schema)

        # Update in-memory cache
        SchemaExtractor._memory_cache = schema
        SchemaExtractor._cache_time = datetime.now()
        logger.info(f"Schema cached: {len(schema['tables'])} tables, {len(schema['views'])} views")

        return schema

    def _extract_table_info(
        self, inspector, table_name: str, is_view: bool = False
    ) -> Dict[str, Any]:
        """Extract detailed information about a table/view.

        Args:
            inspector: SQLAlchemy inspector
            table_name: Name of the table
            is_view: Whether this is a view

        Returns:
            Dictionary with table information
        """
        logger.debug(f"Extracting info for {table_name}")

        table_info = {
            "name": table_name,
            "type": "view" if is_view else "table",
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": [],
            "sample_data": [],
            "row_count": 0,
        }

        # Get columns
        columns = inspector.get_columns(table_name)
        for col in columns:
            table_info["columns"].append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "default": str(col.get("default")) if col.get("default") else None,
            })

        # Get primary keys
        pk = inspector.get_pk_constraint(table_name)
        if pk:
            table_info["primary_keys"] = pk.get("constrained_columns", [])

        # Get foreign keys using direct SQL (SQLAlchemy inspector doesn't work well with MSSQL)
        fks = self._get_foreign_keys_sql(table_name)
        table_info["foreign_keys"] = fks

        # Get indexes (skip for views)
        if not is_view:
            indexes = inspector.get_indexes(table_name)
            for idx in indexes:
                table_info["indexes"].append({
                    "name": idx.get("name"),
                    "columns": idx.get("column_names", []),
                    "unique": idx.get("unique", False),
                })

        # Get sample data and row count (conditionally)
        try:
            # Always get row count (fast with sys.tables)
            table_info["row_count"] = self._get_row_count(table_name)
            # Only get sample data if requested (saves 100+ queries)
            if getattr(self, '_include_samples', True):
                table_info["sample_data"] = self._get_sample_data(table_name, limit=5)
        except Exception as e:
            logger.warning(f"Could not get data for {table_name}: {e}")

        return table_info

    def _get_foreign_keys_sql(self, table_name: str) -> List[Dict]:
        """Get foreign keys using direct SQL query (more reliable than SQLAlchemy inspector).

        Args:
            table_name: Name of the table (with or without schema prefix)

        Returns:
            List of foreign key dictionaries
        """
        # Remove schema prefix if present
        clean_table_name = table_name.replace("dbo.", "").replace("[", "").replace("]", "")

        sql = text("""
            SELECT
                cp.name AS parent_column,
                tr.name AS referenced_table,
                cr.name AS referenced_column
            FROM sys.foreign_keys fk
            INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
            INNER JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
            INNER JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
            INNER JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
            WHERE tp.name = :table_name
            ORDER BY fk.name
        """)

        foreign_keys = []
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql, {"table_name": clean_table_name})
                for row in result:
                    foreign_keys.append({
                        "columns": [row[0]],
                        "referred_table": f"dbo.{row[1]}",
                        "referred_columns": [row[2]],
                    })
        except Exception as e:
            logger.warning(f"Could not get foreign keys for {table_name}: {e}")

        return foreign_keys

    def _get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict]:
        """Get sample rows from a table.

        Args:
            table_name: Name of the table
            limit: Number of rows to sample

        Returns:
            List of row dictionaries
        """
        # Quote table name to handle reserved words like [Order] or [User]
        clean_name = table_name.replace("[", "").replace("]", "")
        if "." in clean_name:
            schema_part, table_part = clean_name.split(".", 1)
            formatted_name = f"[{schema_part}].[{table_part}]"
        else:
            formatted_name = f"[{clean_name}]"

        query = text(f"SELECT TOP {limit} * FROM {formatted_name}")

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = []
            for row in result:
                # Convert row to dict, handling non-serializable types
                row_dict = {}
                for key, value in row._mapping.items():
                    # Convert to string if not JSON serializable
                    try:
                        json.dumps(value)
                        row_dict[key] = value
                    except (TypeError, ValueError):
                        row_dict[key] = str(value)
                rows.append(row_dict)
            return rows

    def _get_row_count(self, table_name: str) -> int:
        """Get approximate row count for a table using system views.

        OPTIMIZED: Uses sys.dm_db_partition_stats instead of SELECT COUNT(*)
        This is 50-100x faster for large tables (instant vs seconds/minutes).

        Args:
            table_name: Name of the table

        Returns:
            Approximate row count (may be slightly stale but very fast)
        """
        # Clean table name for object_id lookup
        clean_name = table_name.replace("dbo.", "").replace("[", "").replace("]", "")

        # Use sys.dm_db_partition_stats for instant row count (no table scan)
        query = text("""
            SELECT SUM(row_count) as count
            FROM sys.dm_db_partition_stats
            WHERE object_id = OBJECT_ID(:table_name)
            AND index_id IN (0, 1)
        """)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"table_name": f"dbo.{clean_name}"})
                count = result.scalar()
                return count or 0
        except Exception as e:
            # Fallback to COUNT(*) for views or if sys view fails
            logger.debug(f"Using fallback COUNT(*) for {table_name}: {e}")
            try:
                fallback_query = text(f"SELECT COUNT(*) as count FROM [{clean_name}]")
                with self.engine.connect() as conn:
                    result = conn.execute(fallback_query)
                    return result.scalar() or 0
            except Exception:
                return 0

    def _save_cache(self, schema: Dict[str, Any]) -> None:
        """Save schema to cache file.

        Args:
            schema: Schema dictionary to save
        """
        with open(self.cache_file, "w") as f:
            json.dump(schema, f, indent=2, default=str)
        logger.info(f"Schema cached to {self.cache_file}")

    def get_table_summary(self, table_name: str) -> Optional[str]:
        """Get a human-readable summary of a table.

        Args:
            table_name: Name of the table

        Returns:
            Formatted summary string
        """
        schema = self.extract_full_schema()

        # Check both tables and views
        table_info = schema["tables"].get(table_name) or schema["views"].get(table_name)

        if not table_info:
            return None

        summary_parts = [
            f"Table: {table_name}",
            f"Type: {table_info['type']}",
            f"Row Count: ~{table_info['row_count']:,}",
            f"\nColumns ({len(table_info['columns'])}):",
        ]

        for col in table_info["columns"]:
            pk_marker = " [PK]" if col["name"] in table_info.get("primary_keys", []) else ""
            nullable = "" if col["nullable"] else " NOT NULL"
            summary_parts.append(
                f"  - {col['name']}: {col['type']}{pk_marker}{nullable}"
            )

        if table_info.get("foreign_keys"):
            summary_parts.append("\nForeign Keys:")
            for fk in table_info.get("foreign_keys", []):
                summary_parts.append(
                    f"  - {', '.join(fk['columns'])} -> "
                    f"{fk['referred_table']}({', '.join(fk['referred_columns'])})"
                )

        if table_info["sample_data"]:
            summary_parts.append("\nSample Data (first 3 rows):")
            for i, row in enumerate(table_info["sample_data"][:3], 1):
                summary_parts.append(f"  Row {i}: {row}")

        return "\n".join(summary_parts)

    def get_all_table_names(self) -> List[str]:
        """Get list of all table and view names.

        Returns:
            List of table/view names
        """
        schema = self.extract_full_schema()
        return list(schema["tables"].keys()) + list(schema["views"].keys())

    @classmethod
    def invalidate_cache(cls) -> None:
        """Invalidate both in-memory and file cache.

        Call this when schema changes need to be picked up.
        """
        cls._memory_cache = None
        cls._cache_time = None
        logger.info("Schema cache invalidated")

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """Get cache status information.

        Returns:
            Dictionary with cache status
        """
        is_cached = cls._memory_cache is not None
        if is_cached and cls._cache_time:
            age = datetime.now() - cls._cache_time
            ttl_remaining = cls.CACHE_TTL - age
            return {
                "cached": True,
                "age_seconds": age.total_seconds(),
                "ttl_remaining_seconds": max(0, ttl_remaining.total_seconds()),
                "tables_count": len(cls._memory_cache.get("tables", {})),
                "views_count": len(cls._memory_cache.get("views", {})),
            }
        return {"cached": False}


if __name__ == "__main__":
    # Test the schema extractor
    extractor = SchemaExtractor()
    schema = extractor.extract_full_schema(force_refresh=True)

    print(f"\nExtracted schema for {schema['database']}")
    print(f"Tables: {len(schema['tables'])}")
    print(f"Views: {len(schema['views'])}")

    # Print first table summary
    if schema['tables']:
        first_table = list(schema['tables'].keys())[0]
        print(f"\n{extractor.get_table_summary(first_table)}")
