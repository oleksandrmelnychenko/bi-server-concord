"""Retrieve similar query examples from FAISS index."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings


# Module-level embedding model (shared across all retrievers)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or create the shared embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info(f"Model loaded: {_embedding_model.get_sentence_embedding_dimension()}d")
    return _embedding_model


class QueryExampleRetriever:
    """Retrieve similar query examples using FAISS + e5-large."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        enable_query_correction: bool = True,
    ):
        """Initialize retriever with FAISS index.

        Args:
            db_path: Path to FAISS index directory
            enable_query_correction: If True, attempt to correct query errors
        """
        self.db_path = Path(db_path or settings.query_examples_db)
        self.enable_query_correction = enable_query_correction

        self.index_file = self.db_path / "examples.faiss"
        self.meta_file = self.db_path / "examples_meta.json"

        self.index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []

        self._load()

    def _load(self) -> bool:
        """Load FAISS index and metadata."""
        if not self.index_file.exists() or not self.meta_file.exists():
            logger.warning(f"FAISS index not found at {self.db_path}")
            return False

        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(f"Loaded FAISS training examples: {len(self._metadata)} examples")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.index = None
            self._metadata = []
            return False

    def is_available(self) -> bool:
        """Check if the retriever is ready to use."""
        return self.index is not None and len(self._metadata) > 0

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query using e5-large format."""
        model = get_embedding_model()
        # E5 format: "query: <text>" for queries
        text = f"query: {query}"
        embedding = model.encode(
            [text],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)

    def find_similar_with_correction(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find similar examples with optional query correction.

        Args:
            query: User query
            top_k: Number of results to return
            domain_filter: Optional domain filter
            min_score: Minimum similarity score

        Returns:
            List of similar examples, best matches first
        """
        return self.find_similar(query, top_k, domain_filter, min_score)

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find similar query examples.

        Args:
            query: Natural language query to match
            top_k: Number of examples to return
            domain_filter: Optional domain to filter by
            min_score: Minimum similarity score (0-1)

        Returns:
            List of similar examples with metadata
        """
        if not self.is_available():
            logger.warning("FAISS index not available")
            return []

        # Embed query
        query_embedding = self._embed_query(query)

        # Search FAISS (get extra results for filtering)
        search_k = top_k * 3 if domain_filter else top_k
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            # Skip low scores
            if score < min_score:
                continue

            meta = self._metadata[idx]

            # Apply domain filter
            if domain_filter and meta.get("domain") != domain_filter:
                continue

            results.append({
                "rank": len(results) + 1,
                "id": meta.get("id", ""),
                "domain": meta.get("domain", ""),
                "category": meta.get("category", ""),
                "complexity": meta.get("complexity", ""),
                "question_en": meta.get("question_en", ""),
                "question_uk": meta.get("question_uk", ""),
                "sql": meta.get("sql", ""),
                "tables_used": meta.get("tables_used", "").split(",") if meta.get("tables_used") else [],
                "notes": meta.get("notes", ""),
                "similarity_score": round(float(score), 3),
            })

            if len(results) >= top_k:
                break

        # Prefer examples that include notes/caveats when scores are similar
        if results:
            results = sorted(
                results,
                key=lambda r: (
                    0 if r.get("notes") else 1,  # notes first
                    -r.get("similarity_score", 0.0),
                    r.get("id", ""),
                ),
            )
            # Re-rank after sorting
            for i, item in enumerate(results, 1):
                item["rank"] = i

        logger.debug(f"Found {len(results)} similar examples for: {query[:50]}...")
        return results

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
            lines.append(f"\nExample {i}:")
            lines.append(f"Q: \"{ex['question_en']}\"")

            if include_ukrainian and ex.get("question_uk"):
                lines.append(f"Q (UK): \"{ex['question_uk']}\"")

            lines.append(f"SQL: {ex['sql']}")

            if ex.get("notes"):
                lines.append(f"Note: {ex['notes']}")

        return "\n".join(lines)

    def get_examples_by_domain(self, domain: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get examples from a specific domain."""
        if not self.is_available():
            return []

        results = []
        for meta in self._metadata:
            if meta.get("domain") == domain:
                results.append({
                    "id": meta.get("id", ""),
                    "domain": meta.get("domain", ""),
                    "category": meta.get("category", ""),
                    "question_en": meta.get("question_en", ""),
                    "question_uk": meta.get("question_uk", ""),
                    "sql": meta.get("sql", ""),
                    "tables_used": meta.get("tables_used", "").split(",") if meta.get("tables_used") else [],
                })
                if len(results) >= limit:
                    break

        return results

    def get_examples_by_tables(self, tables: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Get examples that use specific tables."""
        if not self.is_available():
            return []

        # Use table names as query
        query = " ".join(tables)
        return self.find_similar(query, top_k=limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        if not self.is_available():
            return {"available": False, "total_examples": 0}

        # Count by domain
        domains = {}
        for meta in self._metadata:
            domain = meta.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1

        return {
            "available": True,
            "total_examples": len(self._metadata),
            "domains": domains,
            "db_path": str(self.db_path),
            "embedding_model": settings.embedding_model,
        }


if __name__ == "__main__":
    # Test the retriever
    retriever = QueryExampleRetriever()

    print("\n" + "=" * 60)
    print("QUERY EXAMPLE RETRIEVER TEST (FAISS + e5-large)")
    print("=" * 60)

    stats = retriever.get_stats()
    print(f"Available: {stats['available']}")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Embedding model: {stats.get('embedding_model', 'N/A')}")
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
            "Топ постачальників",
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
