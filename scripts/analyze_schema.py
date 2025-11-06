#!/usr/bin/env python3
"""
Concord Database Schema Analyzer
Analyzes MSSQL database schema to identify tables, relationships, and ML opportunities
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
import pymssql
import pandas as pd
from collections import defaultdict

# MSSQL Connection Configuration
MSSQL_CONFIG = {
    "host": os.getenv("MSSQL_HOST", "78.152.175.67"),
    "port": os.getenv("MSSQL_PORT", "1433"),
    "database": os.getenv("MSSQL_DATABASE", "ConcordDb"),
    "user": os.getenv("MSSQL_USER", "ef_migrator"),
    "password": os.getenv("MSSQL_PASSWORD", "Grimm_jow92"),
}


class SchemaAnalyzer:
    def __init__(self):
        self.conn = None
        self.schema_data = {
            "metadata": {
                "analyzed_at": datetime.now().isoformat(),
                "database": MSSQL_CONFIG["database"],
                "server": MSSQL_CONFIG["host"],
            },
            "tables": [],
            "relationships": [],
            "ml_candidates": {
                "customer_tables": [],
                "product_tables": [],
                "transaction_tables": [],
                "time_series_tables": [],
            },
        }

    def connect(self):
        """Establish connection to MSSQL"""
        try:
            print(f"Connecting to {MSSQL_CONFIG['database']}@{MSSQL_CONFIG['host']}...")
            self.conn = pymssql.connect(
                server=MSSQL_CONFIG['host'],
                port=int(MSSQL_CONFIG['port']),
                user=MSSQL_CONFIG['user'],
                password=MSSQL_CONFIG['password'],
                database=MSSQL_CONFIG['database'],
                tds_version='7.0'
            )
            print("✓ Connected successfully")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def get_all_tables(self) -> List[Dict[str, Any]]:
        """Get all tables with basic metadata"""
        print("\n📊 Discovering tables...")
        query = """
        SELECT
            t.TABLE_SCHEMA,
            t.TABLE_NAME,
            t.TABLE_TYPE,
            p.rows as ROW_COUNT
        FROM INFORMATION_SCHEMA.TABLES t
        LEFT JOIN sys.partitions p ON OBJECT_ID(t.TABLE_SCHEMA + '.' + t.TABLE_NAME) = p.object_id
        WHERE t.TABLE_TYPE = 'BASE TABLE'
            AND p.index_id IN (0, 1)
        ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME
        """
        df = pd.read_sql(query, self.conn)
        print(f"✓ Found {len(df)} tables")
        return df.to_dict("records")

    def get_table_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get detailed column information for a table"""
        query = """
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """
        df = pd.read_sql(query, self.conn, params=[schema, table])
        return df.to_dict("records")

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        """Get primary key columns"""
        query = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
            AND TABLE_NAME = %s
            AND CONSTRAINT_NAME LIKE 'PK_%'
        ORDER BY ORDINAL_POSITION
        """
        df = pd.read_sql(query, self.conn, params=[schema, table])
        return df["COLUMN_NAME"].tolist() if not df.empty else []

    def get_foreign_keys(self) -> List[Dict[str, Any]]:
        """Get all foreign key relationships"""
        print("\n🔗 Analyzing relationships...")
        query = """
        SELECT
            fk.name AS FK_NAME,
            OBJECT_SCHEMA_NAME(fk.parent_object_id) AS FK_SCHEMA,
            OBJECT_NAME(fk.parent_object_id) AS FK_TABLE,
            COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS FK_COLUMN,
            OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS REFERENCED_SCHEMA,
            OBJECT_NAME(fk.referenced_object_id) AS REFERENCED_TABLE,
            COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS REFERENCED_COLUMN
        FROM sys.foreign_keys fk
        INNER JOIN sys.foreign_key_columns fkc
            ON fk.object_id = fkc.constraint_object_id
        ORDER BY FK_SCHEMA, FK_TABLE
        """
        df = pd.read_sql(query, self.conn)
        print(f"✓ Found {len(df)} foreign key relationships")
        return df.to_dict("records")

    def analyze_table(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis of a single table"""
        schema = table_info["TABLE_SCHEMA"]
        table = table_info["TABLE_NAME"]
        full_name = f"{schema}.{table}"

        print(f"  Analyzing {full_name}...", end="")

        # Get columns
        columns = self.get_table_columns(schema, table)

        # Get primary keys
        primary_keys = self.get_primary_keys(schema, table)

        # Identify ML-relevant patterns
        table_lower = table.lower()
        ml_signals = self._classify_table(table_lower, columns)

        table_data = {
            "schema": schema,
            "name": table,
            "full_name": full_name,
            "row_count": table_info.get("ROW_COUNT", 0),
            "column_count": len(columns),
            "columns": columns,
            "primary_keys": primary_keys,
            "ml_signals": ml_signals,
        }

        print(f" ✓ ({len(columns)} cols, {table_data['row_count']} rows)")
        return table_data

    def _classify_table(
        self, table_name: str, columns: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Classify table for ML purposes based on name and columns"""
        column_names = [col["COLUMN_NAME"].lower() for col in columns]

        # Customer-related indicators
        is_customer = any(
            keyword in table_name
            for keyword in [
                "customer",
                "client",
                "user",
                "account",
                "contact",
                "person",
            ]
        ) or any(
            keyword in " ".join(column_names)
            for keyword in ["customerid", "clientid", "email", "phone"]
        )

        # Product-related indicators
        is_product = any(
            keyword in table_name
            for keyword in [
                "product",
                "item",
                "goods",
                "inventory",
                "catalog",
                "article",
            ]
        ) or any(
            keyword in " ".join(column_names)
            for keyword in ["productid", "sku", "itemcode", "price"]
        )

        # Transaction-related indicators
        is_transaction = any(
            keyword in table_name
            for keyword in [
                "order",
                "sale",
                "transaction",
                "invoice",
                "purchase",
                "payment",
            ]
        ) or any(
            keyword in " ".join(column_names)
            for keyword in [
                "orderid",
                "transactionid",
                "amount",
                "total",
                "quantity",
            ]
        )

        # Time series indicators (has date column and numerical data)
        has_date = any(col["DATA_TYPE"] in ["date", "datetime", "datetime2"] for col in columns)
        has_numeric = any(
            col["DATA_TYPE"]
            in ["int", "bigint", "decimal", "numeric", "float", "money"]
            for col in columns
        )
        is_time_series = has_date and has_numeric and is_transaction

        return {
            "is_customer": is_customer,
            "is_product": is_product,
            "is_transaction": is_transaction,
            "is_time_series": is_time_series,
            "has_date": has_date,
            "has_numeric": has_numeric,
        }

    def analyze_full_schema(self):
        """Run complete schema analysis"""
        print("=" * 80)
        print("CONCORD DATABASE SCHEMA ANALYZER")
        print("=" * 80)

        if not self.connect():
            return False

        try:
            # Get all tables
            tables = self.get_all_tables()

            # Analyze each table
            print(f"\n🔍 Analyzing {len(tables)} tables in detail...")
            for table_info in tables:
                table_data = self.analyze_table(table_info)
                self.schema_data["tables"].append(table_data)

                # Categorize for ML
                signals = table_data["ml_signals"]
                if signals["is_customer"]:
                    self.schema_data["ml_candidates"]["customer_tables"].append(
                        table_data["full_name"]
                    )
                if signals["is_product"]:
                    self.schema_data["ml_candidates"]["product_tables"].append(
                        table_data["full_name"]
                    )
                if signals["is_transaction"]:
                    self.schema_data["ml_candidates"]["transaction_tables"].append(
                        table_data["full_name"]
                    )
                if signals["is_time_series"]:
                    self.schema_data["ml_candidates"]["time_series_tables"].append(
                        table_data["full_name"]
                    )

            # Get relationships
            relationships = self.get_foreign_keys()
            self.schema_data["relationships"] = relationships

            # Print summary
            self._print_summary()

            return True

        except Exception as e:
            print(f"\n✗ Analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            if self.conn:
                self.conn.close()

    def _print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        total_tables = len(self.schema_data["tables"])
        total_columns = sum(t["column_count"] for t in self.schema_data["tables"])
        total_rows = sum(t["row_count"] for t in self.schema_data["tables"])
        total_relationships = len(self.schema_data["relationships"])

        print(f"\n📊 Database Statistics:")
        print(f"  - Total Tables: {total_tables}")
        print(f"  - Total Columns: {total_columns:,}")
        print(f"  - Total Rows: {total_rows:,}")
        print(f"  - Foreign Key Relationships: {total_relationships}")

        print(f"\n🤖 ML-Relevant Tables:")
        ml = self.schema_data["ml_candidates"]
        print(f"  - Customer Tables: {len(ml['customer_tables'])}")
        print(f"  - Product Tables: {len(ml['product_tables'])}")
        print(f"  - Transaction Tables: {len(ml['transaction_tables'])}")
        print(f"  - Time Series Tables: {len(ml['time_series_tables'])}")

        print(f"\n📋 Top 10 Largest Tables:")
        sorted_tables = sorted(
            self.schema_data["tables"], key=lambda x: x["row_count"], reverse=True
        )
        for i, table in enumerate(sorted_tables[:10], 1):
            print(
                f"  {i:2}. {table['full_name']:50} - {table['row_count']:>12,} rows"
            )

        if ml["customer_tables"]:
            print(f"\n👥 Customer Tables Found:")
            for table in ml["customer_tables"][:10]:
                print(f"  - {table}")

        if ml["product_tables"]:
            print(f"\n📦 Product Tables Found:")
            for table in ml["product_tables"][:10]:
                print(f"  - {table}")

        if ml["transaction_tables"]:
            print(f"\n💳 Transaction Tables Found:")
            for table in ml["transaction_tables"][:10]:
                print(f"  - {table}")

    def save_results(self, output_path: str = "schema_analysis.json"):
        """Save analysis results to JSON file"""
        with open(output_path, "w") as f:
            json.dump(self.schema_data, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")

    def generate_html_report(self, output_path: str = "schema_analysis.html"):
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Concord Database Schema Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #3498db; color: white; padding: 20px; border-radius: 5px; text-align: center; }}
        .stat-box h3 {{ margin: 0; font-size: 2em; }}
        .stat-box p {{ margin: 5px 0 0 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #34495e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f8f9fa; }}
        .ml-section {{ background: #e8f4f8; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 0.85em; margin: 2px; }}
        .badge-customer {{ background: #3498db; color: white; }}
        .badge-product {{ background: #2ecc71; color: white; }}
        .badge-transaction {{ background: #e74c3c; color: white; }}
        .badge-timeseries {{ background: #f39c12; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Concord Database Schema Analysis</h1>
        <p><strong>Analyzed:</strong> {self.schema_data['metadata']['analyzed_at']}</p>
        <p><strong>Database:</strong> {self.schema_data['metadata']['database']}</p>

        <div class="stat-grid">
            <div class="stat-box">
                <h3>{len(self.schema_data['tables'])}</h3>
                <p>Tables</p>
            </div>
            <div class="stat-box">
                <h3>{sum(t['column_count'] for t in self.schema_data['tables']):,}</h3>
                <p>Columns</p>
            </div>
            <div class="stat-box">
                <h3>{sum(t['row_count'] for t in self.schema_data['tables']):,}</h3>
                <p>Rows</p>
            </div>
            <div class="stat-box">
                <h3>{len(self.schema_data['relationships'])}</h3>
                <p>Relationships</p>
            </div>
        </div>

        <div class="ml-section">
            <h2>🤖 ML-Relevant Tables</h2>
            <p><strong>Customer Tables:</strong> {len(self.schema_data['ml_candidates']['customer_tables'])}</p>
            <p><strong>Product Tables:</strong> {len(self.schema_data['ml_candidates']['product_tables'])}</p>
            <p><strong>Transaction Tables:</strong> {len(self.schema_data['ml_candidates']['transaction_tables'])}</p>
            <p><strong>Time Series Tables:</strong> {len(self.schema_data['ml_candidates']['time_series_tables'])}</p>
        </div>

        <h2>📋 All Tables</h2>
        <table>
            <thead>
                <tr>
                    <th>Schema</th>
                    <th>Table</th>
                    <th>Columns</th>
                    <th>Rows</th>
                    <th>ML Signals</th>
                </tr>
            </thead>
            <tbody>
"""

        for table in sorted(
            self.schema_data["tables"], key=lambda x: x["row_count"], reverse=True
        ):
            badges = []
            if table["ml_signals"]["is_customer"]:
                badges.append('<span class="badge badge-customer">Customer</span>')
            if table["ml_signals"]["is_product"]:
                badges.append('<span class="badge badge-product">Product</span>')
            if table["ml_signals"]["is_transaction"]:
                badges.append('<span class="badge badge-transaction">Transaction</span>')
            if table["ml_signals"]["is_time_series"]:
                badges.append('<span class="badge badge-timeseries">TimeSeries</span>')

            html += f"""
                <tr>
                    <td>{table['schema']}</td>
                    <td><strong>{table['name']}</strong></td>
                    <td>{table['column_count']}</td>
                    <td>{table['row_count']:,}</td>
                    <td>{''.join(badges) if badges else '-'}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html)
        print(f"📄 HTML report saved to: {output_path}")


def main():
    analyzer = SchemaAnalyzer()

    if analyzer.analyze_full_schema():
        # Save results
        output_dir = "/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/data/schema"
        os.makedirs(output_dir, exist_ok=True)

        json_path = f"{output_dir}/schema_analysis.json"
        html_path = f"{output_dir}/schema_analysis.html"

        analyzer.save_results(json_path)
        analyzer.generate_html_report(html_path)

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nView results:")
        print(f"  - JSON: {json_path}")
        print(f"  - HTML: {html_path}")

        return 0
    else:
        print("\n❌ Analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
