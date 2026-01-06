"""Training data module for Query Examples RAG.

This module provides tools for building and using a bilingual (EN/UK)
query examples dataset for few-shot RAG retrieval in SQL generation.

Components:
- QueryExampleRetriever: Semantic search for similar query examples
- SQLValidator: Validate SQL queries against the database
- QueryExampleEmbedder: Embed examples into ChromaDB
"""

from .retriever import QueryExampleRetriever
from .validator import SQLValidator
from .embedder import QueryExampleEmbedder

__all__ = [
    "QueryExampleRetriever",
    "SQLValidator",
    "QueryExampleEmbedder",
]
