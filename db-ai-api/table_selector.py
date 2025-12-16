"""RAG-based table selector using semantic search."""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from schema_extractor import SchemaExtractor
from config import settings


class TableSelector:
    """Use semantic search to find relevant tables for a natural language query."""

    def __init__(self, schema_extractor: Optional[SchemaExtractor] = None):
        """Initialize table selector.

        Args:
            schema_extractor: SchemaExtractor instance (creates one if not provided)
        """
        self.schema_extractor = schema_extractor or SchemaExtractor()
        self.collection_name = f"tables_{settings.db_name}"

        # Initialize ChromaDB client
        db_path = Path(settings.vector_db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def index_schema(self, force_refresh: bool = False) -> None:
        """Index all tables and views into the vector database.

        Args:
            force_refresh: Force re-indexing even if already indexed
        """
        # Check if already indexed
        if not force_refresh and self.collection.count() > 0:
            logger.info(
                f"Collection already has {self.collection.count()} items, skipping indexing"
            )
            return

        logger.info("Indexing database schema into vector database...")

        # Get full schema
        schema = self.schema_extractor.extract_full_schema(force_refresh=force_refresh)

        documents = []
        metadatas = []
        ids = []

        # Index tables
        for table_name, table_info in schema["tables"].items():
            doc, metadata, doc_id = self._create_table_document(table_name, table_info)
            documents.append(doc)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Index views
        for view_name, view_info in schema["views"].items():
            doc, metadata, doc_id = self._create_table_document(view_name, view_info)
            documents.append(doc)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Clear existing collection if force refresh
        if force_refresh:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            self.collection.add(
                documents=batch_docs, metadatas=batch_metas, ids=batch_ids
            )

        logger.info(f"Indexed {len(documents)} tables/views")

    def _create_table_document(
        self, table_name: str, table_info: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any], str]:
        """Create a searchable document from table information.

        Args:
            table_name: Name of the table
            table_info: Table information dictionary

        Returns:
            Tuple of (document_text, metadata, document_id)
        """
        # Create rich text description for embedding
        doc_parts = [
            f"Table: {table_name}",
            f"Type: {table_info['type']}",
            f"Row Count: {table_info.get('row_count', 0)}",
        ]

        # Add column information
        column_descriptions = []
        for col in table_info.get("columns", []):
            col_desc = f"{col['name']} ({col['type']})"
            if not col.get("nullable", True):
                col_desc += " NOT NULL"
            column_descriptions.append(col_desc)

        if column_descriptions:
            doc_parts.append(f"Columns: {', '.join(column_descriptions)}")

        # Add foreign key relationships
        if table_info.get("foreign_keys"):
            fk_descriptions = []
            for fk in table_info["foreign_keys"]:
                fk_desc = f"{', '.join(fk['columns'])} references {fk['referred_table']}"
                fk_descriptions.append(fk_desc)
            doc_parts.append(f"Foreign Keys: {'; '.join(fk_descriptions)}")

        # Add sample data context (just column names and first row)
        if table_info.get("sample_data"):
            sample = table_info["sample_data"][0]
            sample_str = ", ".join(
                f"{k}={v}" for k, v in list(sample.items())[:5]
            )  # First 5 columns
            doc_parts.append(f"Sample: {sample_str}")

        document = "\n".join(doc_parts)

        # Metadata for filtering
        metadata = {
            "table_name": table_name,
            "type": table_info["type"],
            "row_count": table_info.get("row_count", 0),
            "column_count": len(table_info.get("columns", [])),
        }

        doc_id = f"{table_info['type']}_{table_name}"

        return document, metadata, doc_id

    def find_relevant_tables(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find tables most relevant to a natural language query.

        Args:
            query: Natural language query
            top_k: Number of tables to return (uses settings default if not provided)

        Returns:
            List of table information dictionaries with relevance scores
        """
        if top_k is None:
            top_k = settings.top_k_tables

        logger.info(f"Searching for top {top_k} tables relevant to: {query}")

        # Query the vector database
        results = self.collection.query(query_texts=[query], n_results=top_k)

        relevant_tables = []

        if results and results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                relevant_tables.append({
                    "rank": i + 1,
                    "table_name": metadata["table_name"],
                    "type": metadata["type"],
                    "row_count": metadata["row_count"],
                    "column_count": metadata["column_count"],
                    "relevance_score": 1 - distance,  # Convert distance to similarity
                    "summary": doc,
                })

        logger.info(f"Found {len(relevant_tables)} relevant tables")

        return relevant_tables

    def get_context_for_query(self, query: str, top_k: Optional[int] = None) -> str:
        """Get formatted context about relevant tables for LLM.

        Args:
            query: Natural language query
            top_k: Number of tables to include

        Returns:
            Formatted context string
        """
        relevant_tables = self.find_relevant_tables(query, top_k)

        if not relevant_tables:
            return "No relevant tables found."

        context_parts = [
            f"Based on your query, here are the {len(relevant_tables)} most relevant tables:\n"
        ]

        for table in relevant_tables:
            table_name = table["table_name"]
            # Get full table summary
            summary = self.schema_extractor.get_table_summary(table_name)
            context_parts.append(f"\n{'='*60}")
            context_parts.append(f"Rank {table['rank']} - Relevance: {table['relevance_score']:.2f}")
            context_parts.append(summary or f"Table: {table_name}")

        return "\n".join(context_parts)


if __name__ == "__main__":
    # Test the table selector
    selector = TableSelector()

    # Index the schema
    selector.index_schema(force_refresh=True)

    # Test search
    test_queries = [
        "Show me customer orders",
        "Find product inventory",
        "Get employee salaries",
    ]

    for test_query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {test_query}")
        print(selector.get_context_for_query(test_query, top_k=3))
