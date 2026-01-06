"""Retrieve similar query examples from ChromaDB."""
from pathlib import Path
from typing import Dict, List, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Multilingual model for Ukrainian + English support
MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Import variation tools (optional, graceful fallback)
try:
    from .variations.transformers import KeyboardMapper, Transliterator
    VARIATION_TOOLS_AVAILABLE = True
except ImportError:
    VARIATION_TOOLS_AVAILABLE = False
    KeyboardMapper = None
    Transliterator = None


class QueryExampleRetriever:
    """Retrieve similar query examples using semantic search."""

    COLLECTION_NAME = "query_examples"

    def __init__(self, db_path: Optional[str] = None, enable_query_correction: bool = True):
        """Initialize retriever with ChromaDB.

        Args:
            db_path: Path to ChromaDB storage
            enable_query_correction: If True, attempt to correct query errors
        """
        self.db_path = Path(db_path or Path(__file__).parent.parent / "chroma_db_examples_v2")
        self.enable_query_correction = enable_query_correction

        # Initialize variation tools if available
        self._init_variation_tools()

        if not self.db_path.exists():
            logger.warning(f"ChromaDB path does not exist: {self.db_path}")
            self.collection = None
            return

        # Initialize multilingual embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MULTILINGUAL_MODEL
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.client.get_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            logger.info(f"Loaded query examples collection ({self.collection.count()} examples)")
        except Exception as e:
            logger.warning(f"Collection not found: {e}")
            self.collection = None

    def _init_variation_tools(self):
        """Initialize variation detection/correction tools."""
        if VARIATION_TOOLS_AVAILABLE and KeyboardMapper and Transliterator:
            self.keyboard_mapper = KeyboardMapper()
            self.transliterator = Transliterator()
            logger.debug("Query correction tools initialized")
        else:
            self.keyboard_mapper = None
            self.transliterator = None
            logger.debug("Query correction tools not available")

    def is_available(self) -> bool:
        """Check if the retriever is ready to use."""
        return self.collection is not None and self.collection.count() > 0

    def correct_query(self, query: str) -> List[str]:
        """
        Attempt to correct a query that might have errors.

        Detects and fixes:
        - Keyboard layout errors (Ukrainian typed with English keyboard)
        - Transliterated input (Latin spelling of Ukrainian words)

        Args:
            query: User input query

        Returns:
            List of corrected query variants (including original)
        """
        corrections = [query]

        if not self.enable_query_correction:
            return corrections

        # Check for keyboard layout error
        if self.keyboard_mapper:
            layout_error = self.keyboard_mapper.detect_layout_error(query)
            if layout_error:
                corrected = self.keyboard_mapper.correct_layout_error(query)
                if corrected and corrected != query:
                    corrections.append(corrected)
                    logger.debug(f"Keyboard correction: '{query}' -> '{corrected}'")

        # Check for transliteration
        if self.transliterator:
            if self.transliterator.detect_transliteration(query):
                ukrainian = self.transliterator.transliterate_to_ukrainian(query)
                if ukrainian and ukrainian != query:
                    corrections.append(ukrainian)
                    logger.debug(f"Transliteration: '{query}' -> '{ukrainian}'")

        return list(set(corrections))

    def find_similar_with_correction(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Find similar examples with automatic query correction.

        This method first tries to detect and correct common errors:
        - Keyboard layout errors (e.g., "njdfhb" -> "товари")
        - Transliterated input (e.g., "prodazhi" -> "продажі")

        Then searches with both original and corrected queries.

        Args:
            query: User query (may contain errors)
            top_k: Number of results to return
            domain_filter: Optional domain filter
            min_score: Minimum similarity score

        Returns:
            List of similar examples, best matches first
        """
        # Get all query variants (original + corrections)
        queries = self.correct_query(query)

        # Search with all variants
        all_results = []
        seen_ids = set()

        for q in queries:
            results = self.find_similar(
                query=q,
                top_k=top_k,
                domain_filter=domain_filter,
                min_score=min_score
            )

            for result in results:
                result_id = result.get("id", "")
                if result_id not in seen_ids:
                    # Mark if from corrected query
                    if q != query:
                        result["corrected_query"] = q
                    all_results.append(result)
                    seen_ids.add(result_id)

        # Sort by similarity and return top_k
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return all_results[:top_k]

    def find_similar(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find similar query examples.

        Args:
            query: Natural language query to match
            top_k: Number of examples to return
            domain_filter: Optional domain to filter by (products, sales, etc.)
            min_score: Minimum similarity score (0-1)

        Returns:
            List of similar examples with metadata
        """
        if not self.is_available():
            logger.warning("Query examples collection not available")
            return []

        # Build filter if domain specified
        where_filter = None
        if domain_filter:
            where_filter = {"domain": domain_filter}

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        examples = []
        for i, (doc_id, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # Convert distance to similarity score (cosine distance -> similarity)
            similarity = 1 - distance

            # Skip low-relevance matches
            if similarity < min_score:
                continue

            examples.append({
                "rank": i + 1,
                "id": metadata.get("id", ""),
                "domain": metadata.get("domain", ""),
                "category": metadata.get("category", ""),
                "complexity": metadata.get("complexity", ""),
                "question_en": metadata.get("question_en", ""),
                "question_uk": metadata.get("question_uk", ""),
                "sql": metadata.get("sql", ""),
                "tables_used": metadata.get("tables_used", "").split(",") if metadata.get("tables_used") else [],
                "similarity_score": round(similarity, 3),
            })

        logger.debug(f"Found {len(examples)} similar examples for: {query[:50]}...")

        return examples

    def format_examples_for_prompt(
        self,
        examples: List[Dict[str, Any]],
        include_ukrainian: bool = True,
    ) -> str:
        """Format examples for inclusion in LLM prompt.

        Args:
            examples: List of example dictionaries
            include_ukrainian: Whether to include Ukrainian questions

        Returns:
            Formatted string for prompt
        """
        if not examples:
            return ""

        lines = ["SIMILAR QUERY EXAMPLES:"]

        for i, ex in enumerate(examples, 1):
            # Show English question
            lines.append(f"\nExample {i}:")
            lines.append(f"Q: \"{ex['question_en']}\"")

            # Optionally show Ukrainian
            if include_ukrainian and ex.get("question_uk"):
                lines.append(f"Q (UK): \"{ex['question_uk']}\"")

            # Show SQL
            lines.append(f"SQL: {ex['sql']}")

        return "\n".join(lines)

    def get_examples_by_domain(self, domain: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get examples from a specific domain.

        Args:
            domain: Domain name (products, sales, customers, etc.)
            limit: Maximum examples to return

        Returns:
            List of examples
        """
        if not self.is_available():
            return []

        results = self.collection.get(
            where={"domain": domain},
            include=["metadatas"],
            limit=limit,
        )

        examples = []
        for metadata in results.get("metadatas", []):
            examples.append({
                "id": metadata.get("id", ""),
                "domain": metadata.get("domain", ""),
                "category": metadata.get("category", ""),
                "question_en": metadata.get("question_en", ""),
                "question_uk": metadata.get("question_uk", ""),
                "sql": metadata.get("sql", ""),
                "tables_used": metadata.get("tables_used", "").split(",") if metadata.get("tables_used") else [],
            })

        return examples

    def get_examples_by_tables(self, tables: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Get examples that use specific tables.

        Args:
            tables: List of table names to match
            limit: Maximum examples to return

        Returns:
            List of matching examples
        """
        if not self.is_available():
            return []

        # ChromaDB doesn't support complex where clauses, so we'll search and filter
        # Use table name as query to find relevant examples
        query = " ".join(tables)

        results = self.collection.query(
            query_texts=[query],
            n_results=limit * 2,  # Get extra to filter
            include=["metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        examples = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            example_tables = metadata.get("tables_used", "").split(",")

            # Check if any requested table is used
            if any(t in example_tables for t in tables):
                examples.append({
                    "id": metadata.get("id", ""),
                    "domain": metadata.get("domain", ""),
                    "question_en": metadata.get("question_en", ""),
                    "question_uk": metadata.get("question_uk", ""),
                    "sql": metadata.get("sql", ""),
                    "tables_used": example_tables,
                    "similarity_score": round(1 - distance, 3),
                })

            if len(examples) >= limit:
                break

        return examples

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Statistics dictionary
        """
        if not self.is_available():
            return {"available": False, "total_examples": 0}

        count = self.collection.count()

        # Get domain breakdown
        results = self.collection.get(include=["metadatas"])
        domains = {}
        for metadata in results.get("metadatas", []):
            domain = metadata.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1

        return {
            "available": True,
            "total_examples": count,
            "domains": domains,
            "db_path": str(self.db_path),
        }


if __name__ == "__main__":
    # Test the retriever
    retriever = QueryExampleRetriever()

    print("\n" + "=" * 60)
    print("QUERY EXAMPLE RETRIEVER TEST")
    print("=" * 60)

    stats = retriever.get_stats()
    print(f"Available: {stats['available']}")
    print(f"Total examples: {stats['total_examples']}")
    if stats.get('domains'):
        print("By domain:")
        for domain, count in stats['domains'].items():
            print(f"  - {domain}: {count}")

    if retriever.is_available():
        # Test queries
        test_queries = [
            "Show top 10 products by sales",
            "Покажи топ клієнтів",
            "Total revenue this year",
            "Борги клієнтів",
            "Stock levels by warehouse",
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print("-" * 60)

            examples = retriever.find_similar(query, top_k=2)

            if examples:
                print(retriever.format_examples_for_prompt(examples))
            else:
                print("No similar examples found")
