#!/usr/bin/env python3

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductCategoryExplorer:
    """
    Explore the database to find existing product category/classification data.

    Looking for:
    - Product table columns (CategoryID, ProductCategoryID, etc.)
    - Category tables (ProductCategory, Category, etc.)
    - Product metadata (Name patterns, Type fields, etc.)
    """

    def __init__(self, conn):
        self.conn = conn

    def explore_product_table_structure(self):
        """Get Product table column names and types"""
        logger.info("="*80)
        logger.info("EXPLORING PRODUCT TABLE STRUCTURE")
        logger.info("="*80)

        cursor = self.conn.cursor(as_dict=True)

        # Get Product table columns
        query = """
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'Product'
        ORDER BY ORDINAL_POSITION
        """

        cursor.execute(query)
        columns = list(cursor)
        cursor.close()

        logger.info(f"\nProduct table has {len(columns)} columns:")

        category_related = []
        for col in columns:
            col_name = col['COLUMN_NAME']
            data_type = col['DATA_TYPE']
            nullable = col['IS_NULLABLE']

            # Highlight potentially category-related columns
            if any(keyword in col_name.lower() for keyword in ['category', 'type', 'group', 'class', 'kind', 'supplier']):
                category_related.append(col)
                logger.info(f"  🔍 {col_name} ({data_type}) - {nullable}")
            else:
                logger.info(f"     {col_name} ({data_type}) - {nullable}")

        if category_related:
            logger.info(f"\n✓ Found {len(category_related)} potentially category-related columns:")
            for col in category_related:
                logger.info(f"  - {col['COLUMN_NAME']}")
        else:
            logger.info("\n⚠️  No obvious category-related columns found in Product table")

        return columns

    def search_for_category_tables(self):
        """Search for tables with 'category' in the name"""
        logger.info("\n" + "="*80)
        logger.info("SEARCHING FOR CATEGORY TABLES")
        logger.info("="*80)

        cursor = self.conn.cursor(as_dict=True)

        # Get all table names
        query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
              AND TABLE_CATALOG = 'ConcordDb_v5'
        ORDER BY TABLE_NAME
        """

        cursor.execute(query)
        tables = [row['TABLE_NAME'] for row in cursor]
        cursor.close()

        logger.info(f"\nDatabase has {len(tables)} tables")

        # Look for category-related tables
        category_keywords = ['category', 'type', 'group', 'class', 'taxonomy', 'classification']
        category_tables = []

        for table in tables:
            if any(keyword in table.lower() for keyword in category_keywords):
                category_tables.append(table)

        if category_tables:
            logger.info(f"\n✓ Found {len(category_tables)} potentially category-related tables:")
            for table in category_tables:
                logger.info(f"  🔍 {table}")
                self.explore_table_structure(table)
        else:
            logger.info("\n⚠️  No obvious category-related tables found")

        return category_tables

    def explore_table_structure(self, table_name: str):
        """Get structure of a specific table"""
        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """

        cursor.execute(query)
        columns = list(cursor)
        cursor.close()

        logger.info(f"     Columns: {', '.join([c['COLUMN_NAME'] for c in columns])}")

        # Get row count
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) as cnt FROM dbo.[{table_name}]")
            count = cursor.fetchone()[0]
            logger.info(f"     Row count: {count:,}")
        except Exception as e:
            logger.warning(f"     Could not get row count: {e}")
        finally:
            cursor.close()

    def sample_product_names(self, limit: int = 20):
        """Sample product names to infer categories manually"""
        logger.info("\n" + "="*80)
        logger.info("SAMPLING PRODUCT NAMES")
        logger.info("="*80)

        cursor = self.conn.cursor(as_dict=True)

        query = f"""
        SELECT TOP {limit}
            ID,
            Name,
            VendorCode
        FROM dbo.Product
        WHERE Name IS NOT NULL
        ORDER BY ID
        """

        cursor.execute(query)
        products = list(cursor)
        cursor.close()

        logger.info(f"\nSample of {len(products)} products:")
        for p in products:
            logger.info(f"  Product {p['ID']}: {p['Name']} (Vendor: {p.get('VendorCode', 'N/A')})")

        logger.info("\n💡 Analysis: Can we infer categories from product names?")

        # Try to detect common patterns
        name_patterns = {}
        for p in products:
            name = p['Name'].lower()

            # Common auto parts keywords
            keywords = {
                'brake': 'Brake System',
                'filter': 'Filters',
                'oil': 'Fluids & Oils',
                'tire': 'Tires',
                'light': 'Lighting',
                'battery': 'Electrical',
                'glass': 'Glass & Mirrors',
                'exhaust': 'Exhaust System',
                'engine': 'Engine Parts',
                'transmission': 'Transmission',
                'suspension': 'Suspension',
                'bearing': 'Bearings & Bushings'
            }

            for keyword, category in keywords.items():
                if keyword in name:
                    if category not in name_patterns:
                        name_patterns[category] = []
                    name_patterns[category].append(p['Name'])
                    break

        if name_patterns:
            logger.info(f"\n✓ Detected potential categories from names:")
            for category, products in name_patterns.items():
                logger.info(f"  {category}: {len(products)} products")
                logger.info(f"    Example: {products[0]}")
        else:
            logger.info("\n⚠️  Could not infer obvious categories from product names")

    def check_supplier_data(self):
        """Check if products have supplier information (useful for Track 2)"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING SUPPLIER DATA")
        logger.info("="*80)

        cursor = self.conn.cursor(as_dict=True)

        # Check for Supplier table
        query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
              AND TABLE_NAME LIKE '%Supplier%'
        """

        cursor.execute(query)
        supplier_tables = [row['TABLE_NAME'] for row in cursor]

        if supplier_tables:
            logger.info(f"\n✓ Found {len(supplier_tables)} supplier-related tables:")
            for table in supplier_tables:
                logger.info(f"  - {table}")
                self.explore_table_structure(table)
        else:
            logger.info("\n⚠️  No supplier-related tables found")

        cursor.close()

    def run_exploration(self):
        """Run full exploration"""
        logger.info("="*80)
        logger.info("PRODUCT CATEGORY DATA EXPLORATION")
        logger.info("="*80)
        logger.info("Searching for existing category/classification data...")
        logger.info("="*80)

        # Step 1: Product table structure
        self.explore_product_table_structure()

        # Step 2: Search for category tables
        self.search_for_category_tables()

        # Step 3: Sample product names
        self.sample_product_names(limit=30)

        # Step 4: Check supplier data
        self.check_supplier_data()

        # Summary
        logger.info("\n" + "="*80)
        logger.info("EXPLORATION SUMMARY")
        logger.info("="*80)
        logger.info("""
Next Steps:

If categories found:
  → Implement V3.5 with category-based scoring
  → Expected: +5-8% precision improvement

If no categories:
  Option A: Create category mappings manually (business team)
  Option B: Infer categories from product names (NLP)
  Option C: Focus on supplier loyalty instead
        """)


def main():
    conn = get_connection()
    try:
        explorer = ProductCategoryExplorer(conn)
        explorer.run_exploration()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
