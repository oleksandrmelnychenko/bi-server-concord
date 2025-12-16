"""Extract complete schema from ConcordDb_v5."""
import pymssql
import json
from datetime import datetime
import sys

# Direct connection settings
DB_HOST = '78.152.175.67'
DB_PORT = 1433
DB_USER = 'ef_migrator'
DB_PASSWORD = 'Grimm_jow92'
DB_NAME = 'ConcordDb_v5'


def connect():
    """Create database connection."""
    return pymssql.connect(
        server=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        timeout=60
    )


def extract_schema():
    """Extract complete database schema."""
    print("="*70)
    print(f"Extracting Schema: {DB_NAME}")
    print("="*70)
    print()

    conn = connect()
    cursor = conn.cursor(as_dict=True)

    # Get all tables
    print("Step 1: Getting table list...")
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
    print(f"Found {len(tables)} tables/views")
    print()

    schema_data = {
        "database": DB_NAME,
        "extracted_at": datetime.now().isoformat(),
        "host": DB_HOST,
        "tables": {},
        "views": {}
    }

    # Process each table
    print("Step 2: Extracting table details...")
    for i, table_info in enumerate(tables, 1):
        schema = table_info['TABLE_SCHEMA']
        table_name = table_info['TABLE_NAME']
        table_type = table_info['TABLE_TYPE']

        full_name = f"{schema}.{table_name}"

        # Progress indicator
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(tables)} tables processed ({i*100//len(tables)}%)")

        try:
            # Get columns
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
            """, (schema, table_name))

            columns = cursor.fetchall()

            # Get primary keys
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = %s
                  AND TABLE_NAME = %s
                  AND CONSTRAINT_NAME LIKE 'PK_%'
            """, (schema, table_name))

            pk_rows = cursor.fetchall()
            primary_keys = [row['COLUMN_NAME'] for row in pk_rows]

            # Get foreign keys
            cursor.execute("""
                SELECT
                    COL_NAME(fc.parent_object_id, fc.parent_column_id) AS ColumnName,
                    OBJECT_NAME(fc.referenced_object_id) AS ReferencedTable,
                    COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS ReferencedColumn
                FROM sys.foreign_key_columns AS fc
                INNER JOIN sys.objects AS o ON fc.constraint_object_id = o.object_id
                WHERE OBJECT_NAME(fc.parent_object_id) = %s
            """, (table_name,))

            fk_rows = cursor.fetchall()
            foreign_keys = [
                {
                    "column": row['ColumnName'],
                    "references_table": row['ReferencedTable'],
                    "references_column": row['ReferencedColumn']
                }
                for row in fk_rows
            ]

            # Get row count (with timeout protection)
            try:
                cursor.execute(f"SELECT COUNT_BIG(*) as count FROM [{schema}].[{table_name}]")
                row_count = cursor.fetchone()['count']
            except Exception:
                row_count = -1  # Unknown

            # Get sample data (first 3 rows)
            samples = []
            try:
                cursor.execute(f"SELECT TOP 3 * FROM [{schema}].[{table_name}]")
                for row in cursor.fetchall():
                    sample_row = {}
                    for key, value in row.items():
                        if value is None:
                            sample_row[key] = None
                        elif isinstance(value, datetime):
                            sample_row[key] = value.isoformat()
                        elif isinstance(value, bytes):
                            sample_row[key] = f"<binary {len(value)} bytes>"
                        else:
                            sample_row[key] = str(value)[:100]  # Limit string length
                    samples.append(sample_row)
            except Exception:
                samples = []

            # Build table data
            table_data = {
                "schema": schema,
                "name": table_name,
                "full_name": full_name,
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
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "sample_data": samples
            }

            # Store in appropriate category
            if table_type == "VIEW":
                schema_data["views"][full_name] = table_data
            else:
                schema_data["tables"][full_name] = table_data

        except Exception as e:
            print(f"  Warning: Could not fully extract {full_name}: {e}")
            continue

    conn.close()

    # Save to file
    output_file = "schema_cache.json"
    print()
    print(f"Step 3: Saving schema to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(schema_data, f, indent=2, default=str)

    print()
    print("="*70)
    print("Schema Extraction Complete!")
    print("="*70)
    print(f"Tables: {len(schema_data['tables'])}")
    print(f"Views: {len(schema_data['views'])}")
    print(f"Total: {len(schema_data['tables']) + len(schema_data['views'])}")
    print(f"Saved to: {output_file}")
    print()

    # Show top tables by row count
    print("Top 20 largest tables by row count:")
    all_tables = list(schema_data['tables'].values()) + list(schema_data['views'].values())
    sorted_tables = sorted(
        [t for t in all_tables if t['row_count'] >= 0],
        key=lambda x: x['row_count'],
        reverse=True
    )[:20]

    for i, table in enumerate(sorted_tables, 1):
        print(f"{i:2d}. {table['full_name']:50s} - {table['row_count']:>12,} rows ({len(table['columns'])} cols)")

    print()
    print("="*70)

    return schema_data


if __name__ == "__main__":
    try:
        schema = extract_schema()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
