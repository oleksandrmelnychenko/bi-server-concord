# -*- coding: utf-8 -*-
"""
Full Database Extractor - Extract ALL tables for RAG embeddings
"""
import pymssql
import json
import os
from datetime import datetime
from tqdm import tqdm

# Database configuration
DB_HOST = '78.152.175.67'
DB_PORT = 1433
DB_USER = 'ef_migrator'
DB_PASSWORD = 'Grimm_jow92'
DB_NAME = 'ConcordDb_v5'

# Output configuration
OUTPUT_DIR = "data"
OUTPUT_FILE = "all_documents.json"

# Limits to prevent memory issues
MAX_ROWS_PER_TABLE = 100000  # Max rows per table
SKIP_TABLES = [
    # Skip very large mapping tables without useful text
    'ProductAnalogue',      # 1.7M - just ID mappings
    'ProductPricing',       # 1.1M - just prices
    'ProductSlug',          # 739K - URL slugs
    'ProductProductGroup',  # 726K - ID mappings
    'ProductOriginalNumber', # 668K - ID mappings
    'ProductGroupDiscount', # 420K - ID mappings
    'SaleExchangeRate',     # 303K - exchange rates
    'AuditEntityProperty',  # 94K - audit logs
    '__EFMigrationsHistory',
    'sysdiagrams',
]


def get_connection():
    return pymssql.connect(
        server=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        timeout=300,
        charset='UTF-8'
    )


def get_all_tables(cursor):
    """Get all tables with row counts."""
    cursor.execute('''
        SELECT
            t.TABLE_SCHEMA,
            t.TABLE_NAME,
            p.rows as row_count
        FROM INFORMATION_SCHEMA.TABLES t
        JOIN sys.tables st ON t.TABLE_NAME = st.name
        JOIN sys.partitions p ON st.object_id = p.object_id AND p.index_id IN (0, 1)
        WHERE t.TABLE_TYPE = 'BASE TABLE'
        ORDER BY p.rows DESC
    ''')
    return cursor.fetchall()


def get_table_columns(cursor, schema, table):
    """Get column info for a table."""
    cursor.execute('''
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
    ''', (schema, table))
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_primary_key(cursor, schema, table):
    """Get primary key column for a table."""
    cursor.execute('''
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
        AND TABLE_SCHEMA = %s AND TABLE_NAME = %s
    ''', (schema, table))
    row = cursor.fetchone()
    return row[0] if row else 'ID'


def format_value(value):
    """Format a value for document text."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime('%d.%m.%Y')
    if isinstance(value, bool):
        return "Так" if value else "Ні"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:,.2f}"
    return str(value)


def extract_table(cursor, schema, table, columns, pk_col, limit=MAX_ROWS_PER_TABLE):
    """Extract documents from a table."""
    documents = []
    full_name = f"{schema}.{table}"

    # Select columns (prioritize useful ones)
    text_cols = ['Name', 'Description', 'Title', 'Content', 'Comment', 'Note', 'Address',
                 'ActualAddress', 'Email', 'EmailAddress', 'Phone', 'TIN', 'USREOU']
    id_cols = ['ID', pk_col, 'NetUID']
    date_cols = ['Created', 'Updated', 'Modified', 'Date']

    # Build select list
    select_cols = []
    for col in columns.keys():
        if col in text_cols or col in id_cols or col in date_cols:
            select_cols.append(col)
        elif columns[col] in ('nvarchar', 'varchar', 'ntext', 'text'):
            select_cols.append(col)

    # Always include primary key
    if pk_col not in select_cols:
        select_cols.insert(0, pk_col)

    # Limit columns to prevent huge queries
    select_cols = select_cols[:20]

    # Build query
    cols_sql = ', '.join([f'[{c}]' for c in select_cols])
    query = f"SELECT TOP {limit} {cols_sql} FROM [{schema}].[{table}] WHERE Deleted = 0 OR Deleted IS NULL"

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except Exception as e:
        # Try without Deleted filter
        try:
            query = f"SELECT TOP {limit} {cols_sql} FROM [{schema}].[{table}]"
            cursor.execute(query)
            rows = cursor.fetchall()
        except:
            return []

    for row in rows:
        row_dict = dict(zip(select_cols, row))

        # Build document content
        pk_value = row_dict.get(pk_col, row_dict.get('ID', ''))
        name_value = row_dict.get('Name', '')

        # Build text content
        content_parts = []

        # Add table context
        table_context = get_table_context(table)
        if table_context:
            content_parts.append(table_context)

        # Add name if available
        if name_value:
            content_parts.append(f"Назва: {name_value}")

        # Add other text fields
        for col, val in row_dict.items():
            if col in (pk_col, 'ID', 'Name', 'NetUID', 'Deleted'):
                continue
            if val and columns.get(col) in ('nvarchar', 'varchar', 'ntext', 'text'):
                formatted = format_value(val)
                if formatted and len(formatted) > 1:
                    content_parts.append(f"{col}: {formatted}")

        # Add dates
        for col in date_cols:
            if col in row_dict and row_dict[col]:
                content_parts.append(f"{col}: {format_value(row_dict[col])}")

        content = "\n".join(content_parts)

        if not content.strip():
            content = f"Запис з таблиці {table}, ID: {pk_value}"

        doc = {
            "id": f"{full_name}_{pk_value}",
            "table": full_name,
            "primary_key": pk_col,
            "primary_key_value": str(pk_value),
            "content": content,
            "raw_data": {k: format_value(v) for k, v in row_dict.items() if v is not None}
        }
        documents.append(doc)

    return documents


def get_table_context(table):
    """Get Ukrainian context for table types."""
    contexts = {
        'Client': 'Клієнт, замовник, покупець',
        'Product': 'Товар, продукт, продукція',
        'Order': 'Замовлення, заказ',
        'OrderItem': 'Позиція замовлення, товар у замовленні',
        'Sale': 'Продаж, реалізація, угода',
        'Debt': 'Борг, заборгованість, неоплачено',
        'Payment': 'Оплата, платіж',
        'Invoice': 'Рахунок, накладна',
        'Delivery': 'Доставка, відправка',
        'User': 'Користувач, менеджер',
        'Region': 'Регіон, область',
        'Agreement': 'Договір, угода, контракт',
        'Price': 'Ціна, прайс',
        'Stock': 'Склад, залишки',
        'Return': 'Повернення',
        'Discount': 'Знижка',
        'Category': 'Категорія, група',
        'Brand': 'Бренд, виробник, марка',
        'Car': 'Автомобіль, машина',
        'Specification': 'Специфікація, характеристика',
        'Image': 'Зображення, фото',
        'Address': 'Адреса',
        'Bank': 'Банк, банківські реквізити',
        'Contact': 'Контакт',
        'Document': 'Документ',
        'Currency': 'Валюта',
        'Report': 'Звіт',
    }

    for key, context in contexts.items():
        if key in table:
            return context
    return ""


def main():
    print("\n" + "="*60)
    print("FULL DATABASE EXTRACTOR")
    print("="*60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    # Get all tables
    print("Getting table list...")
    tables = get_all_tables(cursor)
    print(f"Found {len(tables)} tables\n")

    # Filter tables
    tables_to_process = []
    for schema, table, row_count in tables:
        if table in SKIP_TABLES:
            print(f"  SKIP: {schema}.{table} ({row_count:,} rows) - in skip list")
            continue
        if row_count == 0:
            continue
        tables_to_process.append((schema, table, row_count))

    print(f"\nProcessing {len(tables_to_process)} tables...\n")

    all_documents = []
    stats = {
        "tables_processed": 0,
        "tables_skipped": 0,
        "total_documents": 0,
        "started_at": datetime.now().isoformat()
    }

    for schema, table, row_count in tqdm(tables_to_process, desc="Extracting"):
        try:
            columns = get_table_columns(cursor, schema, table)
            pk_col = get_primary_key(cursor, schema, table)

            # Adjust limit for very large tables
            limit = min(row_count, MAX_ROWS_PER_TABLE)

            docs = extract_table(cursor, schema, table, columns, pk_col, limit)

            if docs:
                all_documents.extend(docs)
                stats["tables_processed"] += 1
            else:
                stats["tables_skipped"] += 1

        except Exception as e:
            print(f"\n  ERROR: {schema}.{table}: {str(e)[:50]}")
            stats["tables_skipped"] += 1

    conn.close()

    stats["total_documents"] = len(all_documents)
    stats["completed_at"] = datetime.now().isoformat()

    # Save documents
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"\nSaving {len(all_documents):,} documents to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=None)

    # Save stats
    stats_path = os.path.join(OUTPUT_DIR, "extraction_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Tables processed: {stats['tables_processed']}")
    print(f"Tables skipped: {stats['tables_skipped']}")
    print(f"Documents extracted: {stats['total_documents']:,}")
    print(f"Output: {output_path}")
    print("="*60 + "\n")

    return output_path


if __name__ == "__main__":
    main()
