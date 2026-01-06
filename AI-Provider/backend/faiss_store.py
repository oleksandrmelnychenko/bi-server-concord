"""Lightweight FAISS vector store helper for schema indexing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss  # type: ignore
import numpy as np
from loguru import logger


class FaissVectorStore:
    """Manage a FAISS index and associated metadata."""

    def __init__(
        self,
        index_dir: Path,
        index_type: str = "flat",
        nlist: int = 100,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "metadata.json"
        self.index_type = index_type.lower()
        self.nlist = max(nlist, 1)

        self.index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []

    def load(self) -> bool:
        """Load index + metadata from disk."""
        if not self.index_file.exists() or not self.meta_file.exists():
            return False

        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(f"Loaded FAISS index from {self.index_file}")
            return True
        except Exception as err:
            logger.warning(f"Failed to load FAISS index, will rebuild: {err}")
            self.index = None
            self._metadata = []
            return False

    def rebuild(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None:
        """Rebuild index from embeddings and metadata."""
        if embeddings.size == 0:
            raise ValueError("No embeddings provided to build FAISS index")

        if self.index_type not in {"flat", "ivf"}:
            logger.warning(f"Unknown FAISS index type '{self.index_type}', falling back to flat")
            self.index_type = "flat"

        dim = embeddings.shape[1]
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        else:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            # Train IVF on the provided embeddings
            index.train(embeddings)

        index.add(embeddings)
        faiss.write_index(index, str(self.index_file))

        records = []
        for idx, meta, doc in zip(ids, metadatas, documents):
            records.append({
                "id": idx,
                "metadata": meta,
                "document": doc,
            })

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        self.index = index
        self._metadata = records
        logger.info(f"FAISS index rebuilt with {len(records)} items ({self.index_type})")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the FAISS index."""
        if self.index is None:
            if not self.load():
                logger.error("FAISS index not available; build it first")
                return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, top_k)
        results: List[Dict[str, Any]] = []
        meta_lookup = self._metadata

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(meta_lookup):
                continue
            record = meta_lookup[int(idx)]
            results.append({
                "score": float(score),
                "id": record.get("id"),
                "metadata": record.get("metadata", {}),
                "document": record.get("document"),
            })

        return results
