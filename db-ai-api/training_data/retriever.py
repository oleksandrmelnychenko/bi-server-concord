"""Retrieve similar query examples from ChromaDB."""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

from loguru import logger


# Module-level cache for query corrections (shared across all retrievers)
# This caches the result of the 8-transformer pipeline
@lru_cache(maxsize=512)
def _cached_correct_query(query: str, enable_correction: bool) -> Tuple[str, ...]:
    """Cached query correction using the 8-transformer pipeline."""
    if not enable_correction:
        return (query,)

    if not VARIATION_TOOLS_AVAILABLE:
        return (query,)

    corrections = [query]

    # Detect language
    has_cyrillic = any(ord(c) > 1024 and ord(c) < 1280 for c in query)
    lang = "uk" if has_cyrillic else "en"

    # Use class-level singleton transformers
    cls = QueryExampleRetriever

    # 1. Keyboard layout error
    if cls._keyboard_mapper:
        layout_error = cls._keyboard_mapper.detect_layout_error(query)
        if layout_error:
            corrected = cls._keyboard_mapper.correct_layout_error(query)
            if corrected and corrected != query:
                corrections.append(corrected)

    # 2. Transliteration
    if cls._transliterator:
        if cls._transliterator.detect_transliteration(query):
            ukrainian = cls._transliterator.transliterate_to_ukrainian(query)
            if ukrainian and ukrainian != query:
                corrections.append(ukrainian)

    # 3. Synonyms
    if cls._synonym_expander:
        if cls._synonym_expander.can_transform(query, lang):
            for variant in cls._synonym_expander.transform(query, lang)[:2]:
                if variant and variant != query and variant not in corrections:
                    corrections.append(variant)

    # 4. Number normalization
    if cls._number_handler:
        if cls._number_handler.can_transform(query, lang):
            for variant in cls._number_handler.words_to_digits(query, lang)[:1]:
                if variant and variant != query and variant not in corrections:
                    corrections.append(variant)

    # 5. Formality variants
    if cls._formality_shifter:
        if cls._formality_shifter.can_transform(query, lang):
            for variant in cls._formality_shifter.transform(query, lang)[:1]:
                if variant and variant != query and variant not in corrections:
                    corrections.append(variant)

    # 6. Typo correction
    if cls._typo_corrector:
        if cls._typo_corrector.can_transform(query, lang):
            for variant in cls._typo_corrector.transform(query, lang)[:2]:
                if variant and variant != query and variant not in corrections:
                    corrections.append(variant)

    # 7. Question form variants
    if cls._question_former:
        if cls._question_former.can_transform(query, lang):
            for variant in cls._question_former.transform(query, lang)[:1]:
                if variant and variant != query and variant not in corrections:
                    corrections.append(variant)

    # 8. Language mixing
    if cls._language_mixer:
        if cls._language_mixer.can_transform(query, lang):
            for variant in cls._language_mixer.transform(query, lang)[:1]:
                if variant and variant != query and variant not in corrections:
                    corrections.append(variant)

    # Return as tuple (hashable) limited to 8 variants
    return tuple(set(corrections))[:8]

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared ChromaDB manager (optional, for use with main API)
try:
    from chromadb_manager import chroma_manager
    CHROMA_MANAGER_AVAILABLE = True
except ImportError:
    CHROMA_MANAGER_AVAILABLE = False
    # Fallback to direct imports for standalone use
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions

# Multilingual model for Ukrainian + English support
MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Import ALL variation tools (optional, graceful fallback)
VARIATION_TOOLS_AVAILABLE = False
KeyboardMapper = None
Transliterator = None
SynonymExpander = None
NumberHandler = None
FormalityShifter = None
TypoCorrector = None  # Changed from TypoGenerator - corrects typos instead of generating them
QuestionFormer = None
LanguageMixer = None

try:
    from .variations.transformers import (
        KeyboardMapper, Transliterator, SynonymExpander, NumberHandler,
        FormalityShifter, TypoCorrector, QuestionFormer, LanguageMixer
    )
    VARIATION_TOOLS_AVAILABLE = True
except ImportError:
    pass

if not VARIATION_TOOLS_AVAILABLE:
    try:
        # Fallback for running as main script
        from variations.transformers import (
            KeyboardMapper, Transliterator, SynonymExpander, NumberHandler,
            FormalityShifter, TypoCorrector, QuestionFormer, LanguageMixer
        )
        VARIATION_TOOLS_AVAILABLE = True
    except ImportError:
        pass


class QueryExampleRetriever:
    """Retrieve similar query examples using semantic search."""

    COLLECTION_NAME = "query_examples"

    # Class-level singleton transformers (shared across all instances)
    # Saves 80-400MB memory by not duplicating transformer instances
    _transformers_initialized: bool = False
    _keyboard_mapper = None
    _transliterator = None
    _synonym_expander = None
    _number_handler = None
    _formality_shifter = None
    _typo_corrector = None
    _question_former = None
    _language_mixer = None

    def __init__(self, db_path: Optional[str] = None, enable_query_correction: bool = True):
        """Initialize retriever with ChromaDB.

        Args:
            db_path: Path to ChromaDB storage
            enable_query_correction: If True, attempt to correct query errors
        """
        self.db_path = Path(db_path or Path(__file__).parent.parent / "chroma_db_examples_v2")
        self.enable_query_correction = enable_query_correction

        # Initialize variation tools as class-level singletons (once for all instances)
        self._init_variation_tools()

        if not self.db_path.exists():
            logger.warning(f"ChromaDB path does not exist: {self.db_path}")
            self.collection = None
            return

        # Use shared ChromaDB manager if available (saves ~400MB memory)
        if CHROMA_MANAGER_AVAILABLE:
            try:
                self.collection = chroma_manager.get_collection(
                    db_path=str(self.db_path),
                    collection_name=self.COLLECTION_NAME
                )
                logger.info(f"Loaded query examples collection via shared manager ({self.collection.count()} examples)")
            except Exception as e:
                logger.warning(f"Collection not found: {e}")
                self.collection = None
        else:
            # Fallback: create own ChromaDB client (for standalone use)
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=MULTILINGUAL_MODEL
            )
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
        """Initialize ALL 8 variation detection/correction tools as class-level singletons.

        OPTIMIZED: Transformers are now class-level, initialized once and shared across
        all QueryExampleRetriever instances. Saves 80-400MB memory.
        """
        # Skip if already initialized (singleton pattern)
        if QueryExampleRetriever._transformers_initialized:
            return

        if not VARIATION_TOOLS_AVAILABLE:
            logger.debug("Query correction tools not available")
            QueryExampleRetriever._transformers_initialized = True
            return

        tools_initialized = []

        if KeyboardMapper:
            QueryExampleRetriever._keyboard_mapper = KeyboardMapper()
            tools_initialized.append("KeyboardMapper")

        if Transliterator:
            QueryExampleRetriever._transliterator = Transliterator()
            tools_initialized.append("Transliterator")

        if SynonymExpander:
            QueryExampleRetriever._synonym_expander = SynonymExpander()
            tools_initialized.append("SynonymExpander")

        if NumberHandler:
            QueryExampleRetriever._number_handler = NumberHandler()
            tools_initialized.append("NumberHandler")

        if FormalityShifter:
            QueryExampleRetriever._formality_shifter = FormalityShifter()
            tools_initialized.append("FormalityShifter")

        if TypoCorrector:
            QueryExampleRetriever._typo_corrector = TypoCorrector()
            tools_initialized.append("TypoCorrector")

        if QuestionFormer:
            QueryExampleRetriever._question_former = QuestionFormer()
            tools_initialized.append("QuestionFormer")

        if LanguageMixer:
            QueryExampleRetriever._language_mixer = LanguageMixer()
            tools_initialized.append("LanguageMixer")

        QueryExampleRetriever._transformers_initialized = True

        if tools_initialized:
            logger.info(f"Query correction tools initialized (singleton): {', '.join(tools_initialized)}")
        else:
            logger.debug("No query correction tools available")

    def is_available(self) -> bool:
        """Check if the retriever is ready to use."""
        return self.collection is not None and self.collection.count() > 0

    def correct_query(self, query: str) -> List[str]:
        """
        Attempt to correct a query using ALL 8 transformers (cached).

        Detects and fixes:
        1. Keyboard layout errors (Ukrainian typed with English keyboard)
        2. Transliterated input (Latin spelling of Ukrainian words)
        3. Synonym variants (alternative business terms)
        4. Number format normalization (words to digits)
        5. Formality variants (casual/formal)
        6. Typo detection and correction
        7. Question form variants
        8. Language mixing (UK+EN)

        Args:
            query: User input query

        Returns:
            List of corrected query variants (including original)
        """
        # Use cached function for the 8-transformer pipeline
        result = _cached_correct_query(query, self.enable_query_correction)
        return list(result)

    def find_similar_with_correction(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Find similar examples with automatic query correction.

        OPTIMIZED: Uses batch embedding - single ChromaDB query with multiple texts
        instead of separate queries for each correction variant.

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
        if not self.is_available():
            logger.warning("Query examples collection not available")
            return []

        # Get all query variants (original + corrections) - limit to 3 for performance
        queries = self.correct_query(query)[:3]

        # Deduplicate queries
        unique_queries = list(dict.fromkeys(queries))

        # Build filter if domain specified
        where_filter = None
        if domain_filter:
            where_filter = {"domain": domain_filter}

        # BATCH QUERY: Single ChromaDB call with multiple query_texts
        # This is 60-80% faster than separate queries for each correction
        results = self.collection.query(
            query_texts=unique_queries,
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"],
        )

        if not results or not results["ids"]:
            return []

        # Merge and deduplicate results from all query variants
        all_results = []
        seen_ids = set()

        for query_idx, q in enumerate(unique_queries):
            if query_idx >= len(results["ids"]) or not results["ids"][query_idx]:
                continue

            for i, (doc_id, metadata, distance) in enumerate(zip(
                results["ids"][query_idx],
                results["metadatas"][query_idx],
                results["distances"][query_idx],
            )):
                result_id = metadata.get("id", doc_id)

                if result_id in seen_ids:
                    continue

                # Convert distance to similarity score
                similarity = 1 - distance

                if similarity < min_score:
                    continue

                result = {
                    "rank": len(all_results) + 1,
                    "id": result_id,
                    "domain": metadata.get("domain", ""),
                    "category": metadata.get("category", ""),
                    "complexity": metadata.get("complexity", ""),
                    "question_en": metadata.get("question_en", ""),
                    "question_uk": metadata.get("question_uk", ""),
                    "sql": metadata.get("sql", ""),
                    "tables_used": metadata.get("tables_used", "").split(",") if metadata.get("tables_used") else [],
                    "similarity_score": round(similarity, 3),
                }

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
