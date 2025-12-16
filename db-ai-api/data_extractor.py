"""
Data Extractor for RAG System
Extracts all data from MSSQL database and creates Ukrainian language documents
"""
import pymssql
import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime
from utils.language_utils import create_ukrainian_document
from db_pool import get_connection


class DataExtractor:
    """Extract data from MSSQL and create Ukrainian documents."""

    def __init__(self, output_dir: str = "data"):
        """
        Initialize DataExtractor.

        Args:
            output_dir: Directory to save extracted documents
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load schema from cache
        with open("schema_cache.json", "r", encoding="utf-8") as f:
            schema_data = json.load(f)

        # Extract tables dict from schema
        if "tables" in schema_data:
            self.schema = schema_data["tables"]
        else:
            self.schema = schema_data

        print(f"✓ Loaded schema: {len(self.schema)} tables")

    def get_table_primary_key(self, table_name: str) -> str:
        """
        Get primary key column for a table.

        Args:
            table_name: Table name

        Returns:
            Primary key column name (default: 'ID')
        """
        table_info = self.schema.get(table_name, {})

        # Try primary_keys array first
        primary_keys = table_info.get("primary_keys", [])
        if primary_keys and len(primary_keys) > 0:
            return primary_keys[0]

        # Fallback: look in columns
        columns = table_info.get("columns", [])

        # Look for 'ID' column
        for col in columns:
            col_name = col.get("name", col.get("column_name", ""))
            if col_name.upper() == "ID":
                return col_name

        # Last resort: use first column
        if columns:
            return columns[0].get("name", columns[0].get("column_name", "ID"))

        return "ID"

    def should_extract_table(self, table_name: str) -> bool:
        """
        Determine if table should be extracted.

        Args:
            table_name: Table name

        Returns:
            True if should extract
        """
        # Skip system tables
        skip_patterns = [
            "sys", "INFORMATION_SCHEMA", "trace_xe",
            "MSreplication", "queue_messages", "service_queues"
        ]

        for pattern in skip_patterns:
            if pattern.lower() in table_name.lower():
                return False

        return True

    def get_table_sample_size(self, table_name: str) -> Optional[int]:
        """
        Get recommended sample size for a table.
        Very large tables will be sampled rather than fully extracted.

        Args:
            table_name: Table name

        Returns:
            Sample size or None for full extraction
        """
        # Categories for sampling
        large_reference_tables = [
            "Product", "ProductPricing", "ProductImage", "ProductGroup",
            "Client", "Organization", "ClientWorkplace"
        ]

        transaction_tables = [
            "Order", "OrderItem", "Sale", "Payment", "Invoice",
            "Delivery", "Movement", "Stock"
        ]

        # Large reference: sample 10000
        for pattern in large_reference_tables:
            if pattern.lower() in table_name.lower():
                return 10000

        # Transactions: sample 50000
        for pattern in transaction_tables:
            if pattern.lower() in table_name.lower():
                return 50000

        # Small tables: extract all
        return None

    def extract_table_data(self, table_name: str,
                          sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract data from a single table.

        Args:
            table_name: Table name
            sample_size: Optional sample size (None = all rows)

        Returns:
            List of row dictionaries
        """
        conn = get_connection()
        cursor = conn.cursor(as_dict=True)

        try:
            # Build query - table_name may already include schema (dbo.)
            if "." in table_name:
                table_ref = f"[{table_name.replace('.', '].[')}]"
            else:
                table_ref = f"[dbo].[{table_name}]"

            if sample_size:
                query = f"SELECT TOP {sample_size} * FROM {table_ref}"
            else:
                query = f"SELECT * FROM {table_ref}"

            # Add WHERE Deleted = 0 if column exists
            table_info = self.schema.get(table_name, {})
            columns = [col.get("name", col.get("column_name", "")) for col in table_info.get("columns", [])]

            if "Deleted" in columns:
                if "WHERE" in query:
                    query += " AND Deleted = 0"
                else:
                    query += " WHERE Deleted = 0"

            # Execute
            cursor.execute(query)
            rows = cursor.fetchall()

            # Convert datetime objects, UUIDs, and Decimals to JSON-serializable types
            import uuid
            from decimal import Decimal
            result = []
            for row in rows:
                cleaned_row = {}
                for key, value in row.items():
                    if value is None:
                        cleaned_row[key] = None
                    elif isinstance(value, datetime):
                        cleaned_row[key] = value.isoformat()
                    elif isinstance(value, uuid.UUID):
                        cleaned_row[key] = str(value)
                    elif isinstance(value, Decimal):
                        cleaned_row[key] = float(value)
                    elif isinstance(value, bytes):
                        # Skip binary data (images, etc.)
                        continue
                    else:
                        cleaned_row[key] = value
                result.append(cleaned_row)

            return result

        except Exception as e:
            print(f"  ✗ Error extracting {table_name}: {e}")
            return []

        finally:
            cursor.close()
            conn.close()

    def create_documents_from_table(self, table_name: str,
                                   rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create Ukrainian documents from table rows.

        Args:
            table_name: Table name
            rows: List of row dictionaries

        Returns:
            List of documents with metadata
        """
        documents = []
        primary_key = self.get_table_primary_key(table_name)

        for row in rows:
            # Create Ukrainian document
            document_text = create_ukrainian_document(
                table_name=table_name,
                row_data=row,
                primary_key=primary_key
            )

            # Get primary key value
            pk_value = row.get(primary_key, "unknown")

            # Create document with metadata
            doc = {
                "id": f"{table_name}_{pk_value}",
                "table": table_name,
                "primary_key": primary_key,
                "primary_key_value": pk_value,
                "content": document_text,
                "raw_data": row,
                "created_at": datetime.now().isoformat()
            }

            documents.append(doc)

        return documents

    def extract_all_data(self, max_tables: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract all data from database.

        Args:
            max_tables: Optional limit on number of tables (for testing)

        Returns:
            Dictionary with extraction statistics
        """
        print("\n" + "="*60)
        print("🚀 STARTING FULL DATA EXTRACTION")
        print("="*60 + "\n")

        all_documents = []
        stats = {
            "total_tables": 0,
            "extracted_tables": 0,
            "skipped_tables": 0,
            "total_documents": 0,
            "errors": [],
            "started_at": datetime.now().isoformat()
        }

        # Get tables to extract
        tables_to_extract = [
            name for name in self.schema.keys()
            if self.should_extract_table(name)
        ]

        if max_tables:
            tables_to_extract = tables_to_extract[:max_tables]

        stats["total_tables"] = len(tables_to_extract)

        print(f"📊 Tables to process: {len(tables_to_extract)}")
        print(f"💾 Output directory: {self.output_dir}\n")

        # Process each table
        for table_name in tqdm(tables_to_extract, desc="Extracting tables"):
            try:
                # Get sample size
                sample_size = self.get_table_sample_size(table_name)

                # Extract data
                rows = self.extract_table_data(table_name, sample_size)

                if not rows:
                    stats["skipped_tables"] += 1
                    continue

                # Create documents
                documents = self.create_documents_from_table(table_name, rows)
                all_documents.extend(documents)

                stats["extracted_tables"] += 1
                stats["total_documents"] += len(documents)

                # Print progress every 10 tables
                if stats["extracted_tables"] % 10 == 0:
                    print(f"\n  ✓ Processed {stats['extracted_tables']} tables, "
                          f"{stats['total_documents']} documents so far...")

            except Exception as e:
                error_msg = f"Error processing {table_name}: {str(e)}"
                stats["errors"].append(error_msg)
                print(f"\n  ✗ {error_msg}")

        stats["completed_at"] = datetime.now().isoformat()

        # Save documents
        self.save_documents(all_documents, stats)

        return stats

    def save_documents(self, documents: List[Dict[str, Any]],
                      stats: Dict[str, Any]) -> None:
        """
        Save extracted documents to files.

        Args:
            documents: List of documents
            stats: Extraction statistics
        """
        print("\n" + "="*60)
        print("💾 SAVING DOCUMENTS")
        print("="*60 + "\n")

        # Save all documents
        doc_file = os.path.join(self.output_dir, "extracted_documents.json")
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(documents)} documents to {doc_file}")

        # Save statistics
        stats_file = os.path.join(self.output_dir, "extraction_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved statistics to {stats_file}")

        # Print summary
        print("\n" + "="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total tables processed: {stats['extracted_tables']}/{stats['total_tables']}")
        print(f"Total documents created: {stats['total_documents']}")
        print(f"Skipped tables: {stats['skipped_tables']}")
        print(f"Errors: {len(stats['errors'])}")

        if stats["errors"]:
            print("\n⚠️  Errors encountered:")
            for error in stats["errors"][:10]:  # Show first 10
                print(f"  - {error}")

        print("="*60 + "\n")


def main():
    """Main extraction function."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract data from MSSQL database")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--max-tables", type=int, help="Max tables to process (for testing)")
    parser.add_argument("--test", action="store_true", help="Test mode (5 tables only)")

    args = parser.parse_args()

    # Create extractor
    extractor = DataExtractor(output_dir=args.output_dir)

    # Extract data
    max_tables = 5 if args.test else args.max_tables
    stats = extractor.extract_all_data(max_tables=max_tables)

    print("\n✅ Data extraction complete!")
    print(f"📁 Documents saved to: {args.output_dir}/extracted_documents.json")

    return stats


if __name__ == "__main__":
    main()
