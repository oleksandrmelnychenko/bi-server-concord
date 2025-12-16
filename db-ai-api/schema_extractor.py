"""Extract and cache database schema from MSSQL."""
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, inspect, MetaData, select, text
from sqlalchemy.engine import Engine
from loguru import logger

from config import settings


class SchemaExtractor:
    """Extract database schema and sample data from MSSQL."""

    def __init__(self, engine: Optional[Engine] = None):
        """Initialize schema extractor.

        Args:
            engine: SQLAlchemy engine (creates one if not provided)
        """
        self.engine = engine or create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.metadata = MetaData()
        self.cache_file = Path("schema_cache.json")

    def extract_full_schema(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Extract complete database schema.

        Args:
            force_refresh: Force refresh even if cache exists

        Returns:
            Dictionary containing schema information
        """
        # Check cache first
        if not force_refresh and self.cache_file.exists():
            logger.info("Loading schema from cache")
            with open(self.cache_file, "r") as f:
                return json.load(f)

        logger.info("Extracting database schema...")
        inspector = inspect(self.engine)

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

        # Get foreign keys
        fks = inspector.get_foreign_keys(table_name)
        for fk in fks:
            table_info["foreign_keys"].append({
                "columns": fk.get("constrained_columns", []),
                "referred_table": fk.get("referred_table"),
                "referred_columns": fk.get("referred_columns", []),
            })

        # Get indexes (skip for views)
        if not is_view:
            indexes = inspector.get_indexes(table_name)
            for idx in indexes:
                table_info["indexes"].append({
                    "name": idx.get("name"),
                    "columns": idx.get("column_names", []),
                    "unique": idx.get("unique", False),
                })

        # Get sample data and row count
        try:
            table_info["sample_data"] = self._get_sample_data(table_name, limit=5)
            table_info["row_count"] = self._get_row_count(table_name)
        except Exception as e:
            logger.warning(f"Could not get sample data for {table_name}: {e}")

        return table_info

    def _get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict]:
        """Get sample rows from a table.

        Args:
            table_name: Name of the table
            limit: Number of rows to sample

        Returns:
            List of row dictionaries
        """
        query = text(f"SELECT TOP {limit} * FROM {table_name}")

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
        """Get approximate row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            Row count
        """
        query = text(f"SELECT COUNT(*) as count FROM {table_name}")

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.scalar()

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
            pk_marker = " [PK]" if col["name"] in table_info["primary_keys"] else ""
            nullable = "" if col["nullable"] else " NOT NULL"
            summary_parts.append(
                f"  - {col['name']}: {col['type']}{pk_marker}{nullable}"
            )

        if table_info["foreign_keys"]:
            summary_parts.append("\nForeign Keys:")
            for fk in table_info["foreign_keys"]:
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
