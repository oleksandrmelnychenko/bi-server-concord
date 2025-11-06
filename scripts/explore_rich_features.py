#!/usr/bin/env python3
"""
Explore Database Schema for Rich Features

This script discovers:
1. Product categories and hierarchies
2. Product specifications and attributes
3. Customer information (fleet type, industry, etc.)
4. Any other tables that could provide features for GNN
"""

import os
import pymssql
import pandas as pd
from typing import Dict, List

# Database connection
conn = pymssql.connect(
    server=os.environ.get('MSSQL_HOST', '78.152.175.67'),
    port=int(os.environ.get('MSSQL_PORT', '1433')),
    database=os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    user=os.environ.get('MSSQL_USER', 'ef_migrator'),
    password=os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
)

print("\n" + "="*80)
print("DATABASE SCHEMA EXPLORATION FOR GNN FEATURES")
print("="*80)

# 1. Get all tables in the database
print("\n📊 ALL TABLES IN DATABASE:")
print("-" * 80)

query_tables = """
SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    TABLE_TYPE
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_SCHEMA, TABLE_NAME
"""

df_tables = pd.read_sql(query_tables, conn)
print(f"\nTotal tables: {len(df_tables)}")
print(df_tables.to_string(index=False))

# 2. Look for product-related tables
print("\n\n🏷️  PRODUCT-RELATED TABLES:")
print("-" * 80)

product_tables = df_tables[df_tables['TABLE_NAME'].str.contains('Product|Item|Category|Brand|Supplier', case=False, na=False)]
print(product_tables.to_string(index=False))

# 3. Explore Product table structure
print("\n\n📦 PRODUCT TABLE STRUCTURE:")
print("-" * 80)

query_product_columns = """
SELECT
    COLUMN_NAME,
    DATA_TYPE,
    CHARACTER_MAXIMUM_LENGTH,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'Product'
ORDER BY ORDINAL_POSITION
"""

df_product_cols = pd.read_sql(query_product_columns, conn)
print(df_product_cols.to_string(index=False))

# 4. Sample product data
print("\n\n📦 SAMPLE PRODUCT DATA (first 5 products):")
print("-" * 80)

query_sample_products = """
SELECT TOP 5 *
FROM dbo.Product
"""

df_sample_products = pd.read_sql(query_sample_products, conn)
print(df_sample_products.to_string(index=False))

# 5. Look for customer-related tables
print("\n\n👥 CUSTOMER-RELATED TABLES:")
print("-" * 80)

customer_tables = df_tables[df_tables['TABLE_NAME'].str.contains('Client|Customer|Agreement|Account', case=False, na=False)]
print(customer_tables.to_string(index=False))

# 6. Explore Client/ClientAgreement structure
print("\n\n👤 CLIENT TABLE STRUCTURE:")
print("-" * 80)

# First check if Client table exists
if 'Client' in df_tables['TABLE_NAME'].values:
    query_client_columns = """
    SELECT
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'Client'
    ORDER BY ORDINAL_POSITION
    """

    df_client_cols = pd.read_sql(query_client_columns, conn)
    print(df_client_cols.to_string(index=False))

    # Sample client data
    print("\n\n👤 SAMPLE CLIENT DATA (first 5 customers):")
    print("-" * 80)

    query_sample_clients = """
    SELECT TOP 5 *
    FROM dbo.Client
    """

    df_sample_clients = pd.read_sql(query_sample_clients, conn)
    print(df_sample_clients.to_string(index=False))

# 7. Look for category/hierarchy tables
print("\n\n🗂️  CATEGORY/HIERARCHY TABLES:")
print("-" * 80)

category_tables = df_tables[df_tables['TABLE_NAME'].str.contains('Category|Type|Class|Group|Family', case=False, na=False)]
if len(category_tables) > 0:
    print(category_tables.to_string(index=False))

    # Explore each category table
    for _, row in category_tables.iterrows():
        table_name = row['TABLE_NAME']
        print(f"\n\n📁 {table_name} STRUCTURE:")
        print("-" * 80)

        query_cat_columns = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """

        df_cat_cols = pd.read_sql(query_cat_columns, conn)
        print(df_cat_cols.to_string(index=False))

        # Sample data
        query_sample = f"SELECT TOP 10 * FROM dbo.[{table_name}]"
        try:
            df_sample = pd.read_sql(query_sample, conn)
            print(f"\n📊 Sample data:")
            print(df_sample.to_string(index=False))
        except:
            print(f"\n⚠️  Could not fetch sample data")
else:
    print("No explicit category tables found")

# 8. Check for product attributes/specifications
print("\n\n🔍 LOOKING FOR PRODUCT SPECIFICATIONS/ATTRIBUTES:")
print("-" * 80)

spec_tables = df_tables[df_tables['TABLE_NAME'].str.contains('Spec|Attribute|Property|Feature|Detail', case=False, na=False)]
if len(spec_tables) > 0:
    print(spec_tables.to_string(index=False))
else:
    print("No explicit specification tables found")

# 9. Analyze Product table for implicit features
print("\n\n🔬 ANALYZING PRODUCT TABLE FOR IMPLICIT FEATURES:")
print("-" * 80)

# Check what columns exist in Product that could be features
interesting_columns = []
for col in df_product_cols['COLUMN_NAME'].values:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['category', 'type', 'brand', 'supplier', 'manufacturer',
                                                   'description', 'spec', 'model', 'series', 'family',
                                                   'weight', 'size', 'dimension', 'color', 'material']):
        interesting_columns.append(col)

if interesting_columns:
    print(f"\n✅ Found potentially useful product features:")
    for col in interesting_columns:
        print(f"   - {col}")

    # Get sample values for these columns
    print(f"\n📊 Sample values for these features:")
    query_features = f"""
    SELECT TOP 10
        ID,
        {', '.join(interesting_columns)}
    FROM dbo.Product
    WHERE {interesting_columns[0]} IS NOT NULL
    """

    try:
        df_features = pd.read_sql(query_features, conn)
        print(df_features.to_string(index=False))
    except Exception as e:
        print(f"⚠️  Error fetching feature samples: {e}")
else:
    print("No obvious feature columns found in Product table")

# 10. Analyze Client table for implicit features
if 'Client' in df_tables['TABLE_NAME'].values:
    print("\n\n🔬 ANALYZING CLIENT TABLE FOR IMPLICIT FEATURES:")
    print("-" * 80)

    interesting_client_cols = []
    for col in df_client_cols['COLUMN_NAME'].values:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['industry', 'type', 'size', 'fleet', 'business',
                                                       'segment', 'region', 'location', 'category']):
            interesting_client_cols.append(col)

    if interesting_client_cols:
        print(f"\n✅ Found potentially useful customer features:")
        for col in interesting_client_cols:
            print(f"   - {col}")

        # Get sample values
        print(f"\n📊 Sample values for these features:")
        query_client_features = f"""
        SELECT TOP 10
            ID,
            {', '.join(interesting_client_cols)}
        FROM dbo.Client
        WHERE {interesting_client_cols[0]} IS NOT NULL
        """

        try:
            df_client_features = pd.read_sql(query_client_features, conn)
            print(df_client_features.to_string(index=False))
        except Exception as e:
            print(f"⚠️  Error fetching client feature samples: {e}")
    else:
        print("No obvious feature columns found in Client table")

# 11. Summary and recommendations
print("\n\n" + "="*80)
print("🎯 SUMMARY: AVAILABLE FEATURES FOR GNN")
print("="*80)

print("\n📦 PRODUCT FEATURES:")
if interesting_columns:
    print(f"   ✅ Found {len(interesting_columns)} product feature columns")
    for col in interesting_columns:
        print(f"      - {col}")
else:
    print("   ⚠️  Limited product features - may need to extract from descriptions")

print("\n👥 CUSTOMER FEATURES:")
if 'Client' in df_tables['TABLE_NAME'].values and interesting_client_cols:
    print(f"   ✅ Found {len(interesting_client_cols)} customer feature columns")
    for col in interesting_client_cols:
        print(f"      - {col}")
else:
    print("   ⚠️  Limited customer features - may rely on purchase behavior")

print("\n🗂️  HIERARCHICAL FEATURES:")
if len(category_tables) > 0:
    print(f"   ✅ Found {len(category_tables)} category/hierarchy tables")
    for _, row in category_tables.iterrows():
        print(f"      - {row['TABLE_NAME']}")
else:
    print("   ⚠️  No explicit hierarchies - will need to derive from product data")

print("\n\n🚀 NEXT STEPS FOR GNN IMPLEMENTATION:")
print("-" * 80)
print("1. Extract product categories/types from Product table")
print("2. Build product co-purchase graph from transaction history")
print("3. Create product similarity graph from available features")
print("4. Extract customer segments from Client table (if available)")
print("5. Build heterogeneous graph: Customer-Product-Category")
print("6. Implement LightGCN or GraphSAGE model")
print("7. Train with message passing on the graph")

conn.close()
print("\n✅ Schema exploration complete!\n")
