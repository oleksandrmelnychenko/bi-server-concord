"""Learn database schema using pymssql (alternative to pyodbc)."""
import pymssql
import json
from datetime import datetime
from config import settings


def connect():
    """Connect to database using pymssql."""
    return pymssql.connect(
        server=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        as_dict=True,
        timeout=30
    )


def get_tables():
    """Get all tables in the database."""
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            TABLE_SCHEMA,
            TABLE_NAME,
            TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW')
        ORDER BY TABLE_SCHEMA, TABLE_NAME
    """)

    tables = cursor.fetchall()
    conn.close()
    return tables


def get_table_columns(schema, table):
    """Get columns for a specific table."""
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
    """, (schema, table))

    columns = cursor.fetchall()
    conn.close()
    return columns


def get_row_count(schema, table):
    """Get approximate row count."""
    try:
        conn = connect()
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) as count FROM [{schema}].[{table}]")
        result = cursor.fetchone()
        count = result['count'] if result else 0

        conn.close()
        return count
    except Exception as e:
        print(f"Could not get row count for {schema}.{table}: {e}")
        return 0


def get_sample_data(schema, table, limit=3):
    """Get sample rows from a table."""
    try:
        conn = connect()
        cursor = conn.cursor()

        cursor.execute(f"SELECT TOP {limit} * FROM [{schema}].[{table}]")
        samples = cursor.fetchall()

        # Convert to JSON serializable
        for row in samples:
            for key, value in row.items():
                if isinstance(value, (datetime,)):
                    row[key] = value.isoformat()
                elif isinstance(value, bytes):
                    row[key] = value.hex()[:50] + "..." if len(value.hex()) > 50 else value.hex()

        conn.close()
        return samples
    except Exception as e:
        print(f"Could not get sample data for {schema}.{table}: {e}")
        return []


def learn_schema():
    """Learn the entire database schema."""
    print("="*70)
    print(f"Learning schema for database: {settings.db_name}")
    print(f"Host: {settings.db_host}")
    print("="*70)
    print()

    # Get all tables
    tables = get_tables()
    print(f"Found {len(tables)} tables/views")
    print()

    schema_data = {
        "database": settings.db_name,
        "extracted_at": datetime.now().isoformat(),
        "tables": {},
        "views": {}
    }

    # Learn each table
    for i, table_info in enumerate(tables, 1):
        schema = table_info['TABLE_SCHEMA']
        table_name = table_info['TABLE_NAME']
        table_type = table_info['TABLE_TYPE']

        print(f"[{i}/{len(tables)}] Learning: {schema}.{table_name} ({table_type})")

        # Get columns
        columns = get_table_columns(schema, table_name)

        # Get row count
        row_count = get_row_count(schema, table_name)

        # Get sample data
        samples = get_sample_data(schema, table_name, limit=3)

        table_data = {
            "schema": schema,
            "name": table_name,
            "type": "view" if table_type == "VIEW" else "table",
            "row_count": row_count,
            "columns": [
                {
                    "name": col['COLUMN_NAME'],
                    "type": col['DATA_TYPE'],
                    "max_length": col['CHARACTER_MAXIMUM_LENGTH'],
                    "nullable": col['IS_NULLABLE'] == 'YES',
                    "default": col['COLUMN_DEFAULT']
                }
                for col in columns
            ],
            "sample_data": samples
        }

        if table_type == "VIEW":
            schema_data["views"][f"{schema}.{table_name}"] = table_data
        else:
            schema_data["tables"][f"{schema}.{table_name}"] = table_data

        print(f"  - Columns: {len(columns)}")
        print(f"  - Rows: {row_count:,}")
        print()

    # Save to file
    output_file = "schema_cache.json"
    with open(output_file, 'w') as f:
        json.dump(schema_data, f, indent=2, default=str)

    print("="*70)
    print(f"Schema learned successfully!")
    print(f"Tables: {len(schema_data['tables'])}")
    print(f"Views: {len(schema_data['views'])}")
    print(f"Saved to: {output_file}")
    print("="*70)

    # Print summary
    print()
    print("Top 10 largest tables:")
    all_tables = list(schema_data['tables'].values()) + list(schema_data['views'].values())
    sorted_tables = sorted(all_tables, key=lambda x: x['row_count'], reverse=True)

    for i, table in enumerate(sorted_tables[:10], 1):
        print(f"{i:2d}. {table['schema']}.{table['name']:30s} - {table['row_count']:>10,} rows ({len(table['columns'])} columns)")

    return schema_data


if __name__ == "__main__":
    try:
        schema = learn_schema()
    except Exception as e:
        print(f"Error learning schema: {e}")
        import traceback
        traceback.print_exc()
