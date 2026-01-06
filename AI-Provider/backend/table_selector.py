"""RAG-based table selector using semantic search."""
import json
import hashlib
import zlib
from collections import OrderedDict
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from loguru import logger

from schema_extractor import SchemaExtractor
from config import settings

# Optional backends (faiss is preferred; chroma is fallback)
try:
    from chromadb_manager import chroma_manager
except Exception:  # pragma: no cover - chroma optional
    chroma_manager = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - will be needed for FAISS
    SentenceTransformer = None

try:
    from faiss_store import FaissVectorStore
except Exception:  # pragma: no cover - optional until installed
    FaissVectorStore = None

# Import column semantic descriptions for anti-hallucination
try:
    from schema_extractor import SchemaExtractor as SE
    COLUMN_DESCRIPTIONS = getattr(SE, 'COLUMN_DESCRIPTIONS', {})
except (ImportError, AttributeError):
    COLUMN_DESCRIPTIONS = {}


class TableSelector:
    """Use semantic search to find relevant tables for a natural language query."""

    # Cache version - bump this when context format changes to invalidate old caches
    CONTEXT_CACHE_VERSION = 4  # v4: Added column semantic descriptions for anti-hallucination

    # Core business tables that should ALWAYS be included in context
    CORE_TABLES = [
        "dbo.Product",          # Main products table - товари
        "dbo.OrderItem",        # Order line items - позиції замовлень
        "dbo.[Order]",          # Orders - замовлення (reserved word, needs brackets)
        "dbo.Client",           # Clients/customers - клієнти
        "dbo.ClientAgreement",  # Client agreements - договори (CRITICAL: links Client to Order!)
        "dbo.Sale",             # Sales - продажі (linked via OrderID and ClientAgreementID)
        "dbo.Region",           # Regions - області, регіони (for address filtering)
        "dbo.ProductAvailability",  # Stock/inventory - залишки на складі (Amount = quantity)
        "dbo.Debt",             # Debts/receivables - борги (CRITICAL: for debt queries!)
        "dbo.ClientInDebt",     # Client-debt relationship - зв'язок клієнт-борг (links Client to Debt)
        "dbo.Payment",          # Payments - платежі
        "dbo.Bank",             # Banks - банки
    ]

    def __init__(self, schema_extractor: Optional[SchemaExtractor] = None):
        """Initialize table selector.

        Args:
            schema_extractor: SchemaExtractor instance (creates one if not provided)
        """
        self.schema_extractor = schema_extractor or SchemaExtractor()
        self.collection_name = f"tables_{settings.db_name}"
        self.backend = settings.vector_store_backend.lower()

        self.embedding_model = None
        self.vector_store = None

        if self.backend == "faiss" and FaissVectorStore and SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(settings.embedding_model)
                self.vector_store = FaissVectorStore(
                    index_dir=Path(settings.faiss_index_path),
                    index_type=settings.faiss_index_type,
                    nlist=settings.faiss_nlist,
                )
                logger.info(f"Using FAISS backend at {settings.faiss_index_path}")
            except Exception as err:
                logger.warning(f"Failed to initialize FAISS backend, falling back to Chroma: {err}")
                self.backend = "chroma"

        # Fallback to Chroma
        if self.backend != "faiss":
            if chroma_manager is None:
                raise RuntimeError("Chroma backend requested but chromadb is unavailable")

            # Use shared ChromaDB manager (saves ~400MB by sharing embedding model)
            self.client = chroma_manager.get_client(settings.vector_db_path)
            # Get or create collection with shared embedding function
            self.collection = self._get_or_create_collection()
            logger.info(f"Using Chroma backend at {settings.vector_db_path}")

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection using shared manager."""
        return chroma_manager.get_collection(
            db_path=settings.vector_db_path,
            collection_name=self.collection_name
        )

    def _safe_collection_count(self) -> int:
        """Get collection count with guardrails for invalid collections."""
        try:
            return self.collection.count()
        except Exception as err:
            logger.warning(f"Collection count failed, recreating: {err}")
            self.collection = self._get_or_create_collection()
            try:
                return self.collection.count()
            except Exception as retry_err:
                logger.warning(f"Collection count failed after recreate: {retry_err}")
                return 0

    def index_schema(self, force_refresh: bool = False) -> None:
        """Index all tables and views into the vector database.

        Args:
            force_refresh: Force re-indexing even if already indexed
        """
        # FAISS path
        if self.backend == "faiss":
            # If already exists and not forcing, skip rebuild
            if not force_refresh and self.vector_store.load():
                logger.info("FAISS index already present; skipping rebuild")
                return

            logger.info("Indexing database schema into FAISS store...")
        else:
            # Check if already indexed
            if not force_refresh:
                count = self._safe_collection_count()
                if count > 0:
                    logger.info(
                        f"Collection already has {count} items, skipping indexing"
                    )
                    return

            # Fallback to ensure collection is valid before indexing
            if self.collection is None:
                self.collection = self._get_or_create_collection()

            logger.info("Indexing database schema into Chroma vector database...")

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

        # Apply E5-style formatting for Chroma if using an E5 model
        if self.backend != "faiss" and "e5" in settings.embedding_model.lower():
            documents = [self._format_for_embedding(doc, is_query=False) for doc in documents]

        if self.backend == "faiss":
            self._index_faiss(documents, metadatas, ids)
            return

        # Clear existing collection if force refresh (Chroma)
        if force_refresh:
            try:
                # Invalidate cache BEFORE deleting to avoid stale references
                chroma_manager.invalidate_collection(
                    db_path=settings.vector_db_path,
                    collection_name=self.collection_name
                )
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass  # Collection may not exist
            self.collection = chroma_manager.get_collection(
                db_path=settings.vector_db_path,
                collection_name=self.collection_name
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

    # ---------------------------------------------------------------------
    # FAISS helpers
    # ---------------------------------------------------------------------
    def _format_for_embedding(self, text: str, is_query: bool = False) -> str:
        """Apply model-specific formatting (e.g., E5 requires query/passages)."""
        if "e5" in settings.embedding_model.lower():
            prefix = "query: " if is_query else "passage: "
            return f"{prefix}{text}"
        return text

    def _encode_texts(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts using the configured embedding model."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized (required for FAISS backend)")
        formatted = [self._format_for_embedding(t, is_query=is_query) for t in texts]
        embeddings = self.embedding_model.encode(
            formatted,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _index_faiss(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Build FAISS index from documents."""
        if not self.vector_store:
            raise RuntimeError("FAISS backend not initialized")

        embeddings = self._encode_texts(documents, is_query=False)
        self.vector_store.rebuild(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )

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

        relevant_tables = []

        if self.backend == "faiss":
            try:
                query_embedding = self._encode_texts([query], is_query=True)[0]
                results = self.vector_store.search(query_embedding, top_k=top_k)
            except Exception as err:
                logger.error(f"FAISS query failed: {err}")
                return []
        else:
            # Query the vector database using cached embeddings
            query_text = self._format_for_embedding(query, is_query=True)
            query_embedding = chroma_manager.get_cached_embedding(query_text)
            try:
                results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            except Exception as err:
                logger.warning(f"Collection query failed, recreating: {err}")
                self.collection = self._get_or_create_collection()
                try:
                    results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
                except Exception as retry_err:
                    logger.error(f"Collection query failed after recreate: {retry_err}")
                    return []

        if self.backend == "faiss":
            for i, item in enumerate(results):
                meta = item.get("metadata", {})
                relevant_tables.append({
                    "rank": i + 1,
                    "table_name": meta.get("table_name"),
                    "type": meta.get("type"),
                    "row_count": meta.get("row_count"),
                    "column_count": meta.get("column_count"),
                    "relevance_score": float(item.get("score", 0.0)),
                    "summary": item.get("document", ""),
                })
        elif results and results["documents"] and results["documents"][0]:
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

    # Class-level context cache with LRU eviction and compression
    # Format: OrderedDict[cache_key -> compressed_context_bytes]
    # Using zlib compression reduces memory by ~80% (context is highly compressible SQL text)
    _context_cache: OrderedDict = OrderedDict()
    _context_cache_max_size = 100

    def get_context_for_query(self, query: str, top_k: Optional[int] = None) -> str:
        """Get formatted context about relevant tables for LLM.

        Format as CREATE TABLE statements - this is what LLMs understand best.
        ALWAYS includes core business tables + RAG-selected tables.
        Also includes a list of ALL available non-empty tables so LLM knows what exists.

        OPTIMIZED: Uses OrderedDict for true LRU eviction + zlib compression for 80% memory savings.

        Args:
            query: Natural language query
            top_k: Number of additional tables from RAG

        Returns:
            Formatted context string with CREATE TABLE statements
        """
        # Generate cache key from normalized query + version (auto-invalidates on format change)
        normalized = query.lower().strip()
        cache_key = hashlib.md5(f"v{self.CONTEXT_CACHE_VERSION}:{normalized}:{top_k}".encode()).hexdigest()

        # Check cache with LRU access update
        if cache_key in self._context_cache:
            # Move to end (most recently used) for proper LRU
            self._context_cache.move_to_end(cache_key)
            # Decompress and return
            compressed = self._context_cache[cache_key]
            context = zlib.decompress(compressed).decode('utf-8')
            logger.debug(f"Cache hit for query (key: {cache_key[:8]})")
            return context

        # Generate context
        context = self._generate_context(query, top_k)

        # Compress for storage (80% memory savings for SQL text)
        compressed = zlib.compress(context.encode('utf-8'), level=6)

        # LRU eviction: remove oldest (first) entries until under max size
        while len(self._context_cache) >= self._context_cache_max_size:
            self._context_cache.popitem(last=False)  # Remove oldest (first)

        # Add new entry at end (most recently used)
        self._context_cache[cache_key] = compressed

        return context

    def _generate_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Generate tiered schema context for LLM.

        Uses 3-tier approach to include ALL tables while fitting in context window:
        - Tier 1 (FULL): Core business tables with complete schema
        - Tier 2 (KEY): RAG-selected tables with key columns only
        - Tier 3 (COMPACT): All other tables as one-liners

        Args:
            query: Natural language query
            top_k: Number of tables for RAG selection

        Returns:
            Formatted context string with tiered table definitions
        """
        schema = self.schema_extractor.extract_full_schema()
        tier1_parts = []  # Core tables - full detail
        tier2_parts = []  # RAG tables - key columns
        tier3_parts = []  # Other tables - compact list
        included_tables = set()

        # TIER 1: Core business tables with FULL schema
        tier1_parts.append("-- === TIER 1: CORE TABLES (full schema) ===")
        for core_table in self.CORE_TABLES:
            lookup_name = core_table.replace("[", "").replace("]", "").replace("dbo.", "")
            table_info = schema["tables"].get(lookup_name) or schema["views"].get(lookup_name)

            if table_info:
                create_stmt = self._format_as_create_table(core_table, table_info)
                tier1_parts.append(create_stmt)
                included_tables.add(lookup_name)

        # TIER 2: RAG-selected tables with ultra-compact format
        relevant_tables = self.find_relevant_tables(query, top_k)
        rag_tables = []

        for table in relevant_tables:
            raw_name = table["table_name"]
            # Clean dbo. prefix for lookups (RAG returns dbo.TableName)
            clean_name = raw_name.replace("dbo.", "")
            if clean_name in included_tables:
                continue

            table_info = schema["tables"].get(clean_name) or schema["views"].get(clean_name)
            if table_info:
                # Use ultra-compact format for Tier 2 (saves ~60% tokens)
                compact_stmt = self._format_as_ultra_compact(clean_name, table_info)
                rag_tables.append(compact_stmt)
                included_tables.add(clean_name)

        if rag_tables:
            tier2_parts.append("-- === TIER 2: RELEVANT TABLES (ID*, FKcol→Table, key cols) ===")
            tier2_parts.extend(rag_tables)

        # TIER 3: Removed - was adding 200+ tables that confused models
        # Models work better with focused context (Tier 1 + Tier 2 only)

        # Combine tiers (without Tier 3)
        all_parts = tier1_parts + [""] + tier2_parts

        if not tier1_parts[1:]:  # Only header, no tables
            return "No relevant tables found."

        return "\n".join(all_parts)

    def get_selected_tables(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Get list of tables that would be selected for a query.

        Returns combined list of:
        - CORE_TABLES (always included)
        - RAG-selected tables (based on query similarity)

        Args:
            query: Natural language query
            top_k: Number of RAG tables to select

        Returns:
            List of table names (normalized, without dbo. prefix)
        """
        tables = []

        # Add core tables
        for core_table in self.CORE_TABLES:
            normalized = core_table.replace("dbo.", "").replace("[", "").replace("]", "")
            tables.append(normalized)

        # Add RAG-selected tables
        relevant = self.find_relevant_tables(query, top_k)
        for table in relevant:
            table_name = table["table_name"]
            if table_name not in tables:
                tables.append(table_name)

        return tables

    @classmethod
    def invalidate_context_cache(cls) -> None:
        """Clear the context cache.

        Call this when schema changes or force refresh is needed.
        """
        cls._context_cache.clear()
        logger.info("Context cache invalidated")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get context cache statistics including compression info."""
        compressed_size = sum(len(v) for v in cls._context_cache.values())
        return {
            "cache_size": len(cls._context_cache),
            "max_size": cls._context_cache_max_size,
            "compressed_bytes": compressed_size,
            "avg_compressed_size": compressed_size // max(len(cls._context_cache), 1),
            "compression_enabled": True,
        }

    # Priority columns that should always be included for each table
    PRIORITY_COLUMNS = {
        "Client": ["ID", "Name", "FullName", "LegalAddress", "ActualAddress", "DeliveryAddress",
                   "MobileNumber", "EmailAddress", "RegionID", "Deleted", "TIN", "USREOU"],
        "ClientAgreement": ["ID", "ClientID", "Name", "Deleted", "Created"],
        "Order": ["ID", "ClientAgreementID", "OrderDate", "OrderNumber", "Deleted", "Status", "TotalSum"],
        "OrderItem": ["ID", "OrderID", "ProductID", "Qty", "PricePerItem", "Deleted", "Discount"],
        "Product": ["ID", "Name", "VendorCode", "Description", "Deleted", "Price", "CategoryID"],
        "Sale": ["ID", "OrderID", "ClientAgreementID", "SaleDate", "TotalSum", "Deleted"],
        "Region": ["ID", "Name", "ParentID", "Deleted"],
        "ProductAvailability": ["ID", "ProductID", "StorageID", "Amount", "Deleted", "Created"],
    }

    def _format_as_create_table(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Format table info as CREATE TABLE statement.

        Prioritizes important columns and includes foreign key relationships.

        Args:
            table_name: Name of the table
            table_info: Table information dict

        Returns:
            CREATE TABLE statement string with foreign key comments
        """
        columns = table_info.get("columns", [])
        primary_keys = table_info.get("primary_keys", [])
        foreign_keys = table_info.get("foreign_keys", [])

        # Get table short name for priority lookup
        short_name = table_name.replace("dbo.", "").replace("[", "").replace("]", "")
        priority_cols = self.PRIORITY_COLUMNS.get(short_name, [])

        # Sort columns: priority columns first, then others
        def column_sort_key(col):
            name = col["name"]
            if name in priority_cols:
                return (0, priority_cols.index(name))  # Priority columns first, in order
            return (1, name)  # Others alphabetically

        sorted_columns = sorted(columns, key=column_sort_key)

        col_defs = []
        for col in sorted_columns:
            col_name = col["name"]
            col_type = col["type"]

            # Map types to T-SQL
            type_map = {
                "bigint": "bigint",
                "int": "int",
                "nvarchar": "nvarchar",
                "varchar": "varchar",
                "bit": "bit",
                "datetime2": "datetime2",
                "float": "float",
                "money": "money",
                "decimal": "decimal",
                "uniqueidentifier": "uniqueidentifier",
            }

            sql_type = type_map.get(col_type.lower().split("(")[0], col_type)

            # Add length for string types
            if "nvarchar" in col_type.lower() or "varchar" in col_type.lower():
                max_len = col.get("max_length")
                if max_len and max_len > 0:
                    sql_type = f"nvarchar({max_len})"
                else:
                    sql_type = "nvarchar(max)"

            pk_marker = " PRIMARY KEY" if col_name in primary_keys else ""
            null_marker = "" if col.get("nullable", True) else " NOT NULL"

            # Mark foreign key columns
            fk_marker = ""
            for fk in foreign_keys:
                if col_name in fk.get("columns", []):
                    fk_marker = f" -- FK -> {fk['referred_table']}"
                    break

            # Add semantic description if available (anti-hallucination)
            col_key = f"{short_name}.{col_name}"
            semantic_desc = COLUMN_DESCRIPTIONS.get(col_key, "")
            if semantic_desc and not fk_marker:
                fk_marker = f" -- {semantic_desc}"
            elif semantic_desc and fk_marker:
                fk_marker = f"{fk_marker} | {semantic_desc}"

            col_defs.append(f"    {col_name} {sql_type}{pk_marker}{null_marker}{fk_marker}")

        # Include up to 25 columns for tables with priority columns, 20 otherwise
        max_cols = 25 if priority_cols else 20
        columns_str = ",\n".join(col_defs[:max_cols])

        if len(col_defs) > max_cols:
            columns_str += f"\n    -- ... and {len(col_defs) - max_cols} more columns"

        return f"CREATE TABLE {table_name} (\n{columns_str}\n);"

    def _format_as_compact_list(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Format table as compact one-liner for Tier 3.

        Shows FK relationships inline: TableName(ID, ClientID->Client, Name, ...)

        Args:
            table_name: Name of the table
            table_info: Table information dict

        Returns:
            Compact one-liner with FK references
        """
        columns = table_info.get("columns", [])
        foreign_keys = table_info.get("foreign_keys", [])

        # Build FK lookup: column_name -> referenced_table
        fk_map = {}
        for fk in foreign_keys:
            for col in fk.get("columns", []):
                ref_table = fk["referred_table"].replace("dbo.", "")
                fk_map[col] = ref_table

        # Format columns, showing FK references
        col_parts = []
        for c in columns[:8]:
            name = c["name"]
            if name in fk_map:
                col_parts.append(f"{name}->{fk_map[name]}")
            else:
                col_parts.append(name)

        cols_str = ", ".join(col_parts)
        if len(columns) > 8:
            cols_str += ", ..."

        return f"dbo.{table_name}({cols_str})"

    def generate_compact_full_schema(self) -> str:
        """Generate ultra-compact schema for ALL tables.

        Creates a compact one-line representation of every table in the database,
        including columns and FK relationships. This allows the LLM to know about
        ALL 300+ tables while fitting in ~15-20KB of context.

        Format: TABLE: TableName(col1, col2, ...) FK: col->RefTable, ...

        Returns:
            Compact schema string with all tables (sorted alphabetically)
        """
        schema = self.schema_extractor.extract_full_schema()
        lines = []

        for table_name in sorted(schema["tables"].keys()):
            table_info = schema["tables"][table_name]

            # Skip empty tables
            if table_info.get("row_count", 0) == 0:
                continue

            # Get first 8 columns (most important ones)
            columns = table_info.get("columns", [])
            col_names = [c["name"] for c in columns[:8]]

            # Add "..." if more columns exist
            if len(columns) > 8:
                col_names.append("...")

            # Get FK relationships
            foreign_keys = table_info.get("foreign_keys", [])
            fk_parts = []
            for fk in foreign_keys:
                if fk.get("columns"):
                    fk_col = fk["columns"][0]
                    ref_table = fk["referred_table"].replace("dbo.", "")
                    fk_parts.append(f"{fk_col}->{ref_table}")

            # Build line
            line = f"TABLE: {table_name}({', '.join(col_names)})"
            if fk_parts:
                line += f" FK: {', '.join(fk_parts)}"

            lines.append(line)

        # Add views as well
        for view_name in sorted(schema["views"].keys()):
            view_info = schema["views"][view_name]

            columns = view_info.get("columns", [])
            col_names = [c["name"] for c in columns[:6]]
            if len(columns) > 6:
                col_names.append("...")

            lines.append(f"VIEW: {view_name}({', '.join(col_names)})")

        return "\n".join(lines)

    def get_full_schema_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Get focused schema context for relevant tables only.

        Returns only core + RAG-selected tables (no full schema dump).
        This keeps context small enough for smaller models to handle.

        Args:
            query: Natural language query
            top_k: Number of tables for RAG selection

        Returns:
            Focused context string with relevant tables only
        """
        # Only return detailed context for relevant tables
        # Removed compact_schema which was 300+ table one-liners that confused models
        return self._generate_context(query, top_k)

    def _format_as_key_columns(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Format table with only key columns for Tier 2.

        Includes: ID, Name, foreign key columns, and Deleted flag.

        Args:
            table_name: Name of the table
            table_info: Table information dict

        Returns:
            CREATE TABLE statement with key columns only
        """
        columns = table_info.get("columns", [])
        primary_keys = table_info.get("primary_keys", [])
        foreign_keys = table_info.get("foreign_keys", [])

        # Collect FK column names
        fk_columns = set()
        for fk in foreign_keys:
            fk_columns.update(fk.get("columns", []))

        # Key columns: ID, Name-like, FKs, Deleted, and a few important ones
        key_patterns = {"ID", "Name", "Deleted", "Created", "Date", "Status", "Amount", "Qty", "Sum", "Price"}

        # Filter to key columns only
        key_cols = []
        for col in columns:
            col_name = col["name"]
            # Include if: primary key, foreign key, or matches key pattern
            is_key = (
                col_name in primary_keys or
                col_name in fk_columns or
                any(pattern in col_name for pattern in key_patterns)
            )
            if is_key:
                key_cols.append(col)

        # Limit to 10 columns max
        key_cols = key_cols[:10]

        if not key_cols:
            # Fallback: just use first 5 columns
            key_cols = columns[:5]

        col_defs = []
        for col in key_cols:
            col_name = col["name"]
            col_type = col["type"].split("(")[0]  # Short type name

            # Short type mapping
            short_types = {
                "nvarchar": "nv", "varchar": "vc", "bigint": "big",
                "datetime2": "dt", "uniqueidentifier": "uid",
            }
            sql_type = short_types.get(col_type.lower(), col_type.lower())

            pk_marker = " PK" if col_name in primary_keys else ""

            # Mark FK columns
            fk_marker = ""
            for fk in foreign_keys:
                if col_name in fk.get("columns", []):
                    ref_table = fk["referred_table"].replace("dbo.", "")
                    fk_marker = f" -> {ref_table}"
                    break

            col_defs.append(f"  {col_name} {sql_type}{pk_marker}{fk_marker}")

        columns_str = ",\n".join(col_defs)
        total_cols = len(columns)

        return f"-- {table_name} ({total_cols} cols)\nCREATE TABLE {table_name} (\n{columns_str}\n);"

    def _format_as_ultra_compact(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Format table as ultra-compact one-liner for optimized context.

        Format: TableName: ID*, Name, FKCol→RefTable, Deleted

        Args:
            table_name: Name of the table
            table_info: Table information dict

        Returns:
            Ultra-compact one-liner with FK references
        """
        columns = table_info.get("columns", [])
        primary_keys = table_info.get("primary_keys", [])
        foreign_keys = table_info.get("foreign_keys", [])

        # Build FK lookup
        fk_map = {}
        for fk in foreign_keys:
            for col in fk.get("columns", []):
                ref_table = fk["referred_table"].replace("dbo.", "").replace("[", "").replace("]", "")
                fk_map[col] = ref_table

        # Key patterns to include
        key_patterns = {"ID", "Name", "Deleted", "Created", "Date", "Status", "Amount", "Qty", "Price"}

        # Filter to key columns
        col_parts = []
        fk_columns = set(fk_map.keys())

        for col in columns:
            name = col["name"]
            is_key = (
                name in primary_keys or
                name in fk_columns or
                any(pattern in name for pattern in key_patterns)
            )
            if is_key and len(col_parts) < 8:
                if name in primary_keys:
                    col_parts.append(f"{name}*")
                elif name in fk_map:
                    col_parts.append(f"{name}→{fk_map[name]}")
                else:
                    col_parts.append(name)

        cols_str = ", ".join(col_parts)
        return f"{table_name}: {cols_str}"


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
