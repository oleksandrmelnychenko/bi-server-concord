"""Singleton ChromaDB manager for efficient resource sharing.

This module provides a shared ChromaDB client and embedding function
to avoid loading multiple embedding models into memory.
"""
from pathlib import Path
from typing import Optional, Tuple, List
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from loguru import logger

from config import settings


# Module-level embedding cache (saves 100-300ms per cached hit)
# This is outside the class to work with @lru_cache
_embedding_cache_enabled = True


@lru_cache(maxsize=2048)
def _cached_encode_single(text: str) -> Tuple[float, ...]:
    """Cache single text embeddings."""
    if chroma_manager is None or chroma_manager.embedding_fn is None:
        raise ValueError("ChromaDB manager not initialized")
    result = chroma_manager.embedding_fn([text])
    return tuple(result[0])


class ChromaDBManager:
    """Singleton manager for ChromaDB resources.

    Provides shared access to:
    - ChromaDB PersistentClient
    - SentenceTransformer embedding function

    This avoids loading multiple ~400MB embedding models into memory.
    """

    _instance: Optional["ChromaDBManager"] = None
    _initialized: bool = False

    # Default embedding model (multilingual for Ukrainian/English)
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize ChromaDB resources (only on first call)."""
        if ChromaDBManager._initialized:
            return

        logger.info("Initializing ChromaDB manager (singleton)...")

        # Initialize embedding function (loads model into memory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBEDDING_MODEL
        )
        logger.info(f"Loaded embedding model: {self.EMBEDDING_MODEL}")

        # Store paths for different databases
        self._clients = {}
        self._collections = {}

        ChromaDBManager._initialized = True
        logger.info("ChromaDB manager initialized")

    def get_client(self, db_path: str) -> chromadb.PersistentClient:
        """Get or create a ChromaDB client for the given path.

        Args:
            db_path: Path to the ChromaDB storage directory

        Returns:
            ChromaDB PersistentClient
        """
        if db_path not in self._clients:
            path = Path(db_path)
            path.mkdir(parents=True, exist_ok=True)

            self._clients[db_path] = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.debug(f"Created ChromaDB client for: {db_path}")

        return self._clients[db_path]

    def get_collection(self, db_path: str, collection_name: str):
        """Get or create a collection with shared embedding function.

        Args:
            db_path: Path to the ChromaDB storage
            collection_name: Name of the collection

        Returns:
            ChromaDB Collection
        """
        cache_key = f"{db_path}:{collection_name}"

        if cache_key not in self._collections:
            client = self.get_client(db_path)
            try:
                # Try to get existing collection (without embedding function to avoid conflict)
                collection = client.get_collection(name=collection_name)
                try:
                    count = collection.count()
                    logger.info(f"Loaded existing collection '{collection_name}' ({count} docs)")
                except Exception as count_err:
                    logger.warning(f"Collection '{collection_name}' invalid, recreating: {count_err}")
                    try:
                        client.delete_collection(name=collection_name)
                    except Exception:
                        pass
                    collection = client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_fn,
                        metadata={"hnsw:space": "cosine"},
                    )
                    logger.info(f"Recreated collection: {collection_name}")
                self._collections[cache_key] = collection
            except Exception as e:
                # Collection doesn't exist, create new one with embedding function
                try:
                    self._collections[cache_key] = client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_fn,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Created new collection: {collection_name}")
                except Exception as create_err:
                    # Collection was created by another process, try getting again
                    logger.warning(f"Create failed, trying get: {create_err}")
                    self._collections[cache_key] = client.get_collection(name=collection_name)
                    logger.info(f"Retrieved collection after create conflict: {collection_name}")

        return self._collections[cache_key]

    def get_or_create_collection(self, db_path: str, collection_name: str):
        """Alias for get_collection (for compatibility)."""
        return self.get_collection(db_path, collection_name)

    def invalidate_collection(self, db_path: str, collection_name: str) -> None:
        """Invalidate cached collection (call before deleting).

        Args:
            db_path: Path to the database
            collection_name: Name of the collection to invalidate
        """
        cache_key = f"{db_path}:{collection_name}"
        if cache_key in self._collections:
            del self._collections[cache_key]
            logger.debug(f"Invalidated collection cache: {cache_key}")

    def get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding for text using cache (100-300ms savings per hit).

        Args:
            text: Text to embed

        Returns:
            List of embedding floats (native Python floats, not numpy)
        """
        # Convert numpy float32 to native Python float for ChromaDB compatibility
        return [float(x) for x in _cached_encode_single(text)]

    def get_cached_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using cache.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding lists (native Python floats)
        """
        # Convert numpy float32 to native Python float for ChromaDB compatibility
        return [[float(x) for x in _cached_encode_single(t)] for t in texts]

    @classmethod
    def get_stats(cls) -> dict:
        """Get manager statistics.

        Returns:
            Dictionary with manager stats
        """
        if cls._instance is None:
            return {"initialized": False}

        # Include cache stats
        cache_info = _cached_encode_single.cache_info()

        return {
            "initialized": True,
            "embedding_model": cls.EMBEDDING_MODEL,
            "clients_count": len(cls._instance._clients),
            "collections_count": len(cls._instance._collections),
            "collections": list(cls._instance._collections.keys()),
            "embedding_cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "size": cache_info.currsize,
                "maxsize": cache_info.maxsize,
            }
        }


# Global singleton instance
chroma_manager = ChromaDBManager()
