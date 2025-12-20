"""
Fast Embedder for RAG System
Uses lightweight all-MiniLM-L6-v2 model optimized for CPU
"""
import json
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from datetime import datetime
import numpy as np


class FastEmbedder:
    """Fast CPU-optimized embeddings with all-MiniLM-L6-v2."""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 chroma_dir: str = "chroma_db",
                 collection_name: str = "concorddb_ukrainian"):
        """Initialize FastEmbedder."""
        print("\n" + "="*60)
        print("INITIALIZING FAST EMBEDDER (CPU)")
        print("="*60 + "\n")

        # Load lightweight model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.model.max_seq_length = 256  # Limit for speed
        print(f"OK - Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB
        print(f"\nInitializing ChromaDB: {chroma_dir}")
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name} ({self.collection.count()} docs)")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")

        self.collection_name = collection_name

    def embed_documents(self, documents: List[Dict[str, Any]],
                       batch_size: int = 256,
                       show_progress: bool = True) -> Dict[str, Any]:
        """Embed documents with large batches for CPU efficiency."""
        print("\n" + "="*60)
        print("EMBEDDING DOCUMENTS (CPU BATCH MODE)")
        print("="*60 + "\n")

        stats = {
            "total_documents": len(documents),
            "embedded_documents": 0,
            "skipped_documents": 0,
            "started_at": datetime.now().isoformat()
        }

        print(f"Total documents: {len(documents):,}")
        print(f"Batch size: {batch_size}\n")

        # Resume support
        existing_count = self.collection.count()
        existing_ids = set()
        if existing_count > 0:
            print(f"Found {existing_count:,} existing - enabling resume")
            for offset in range(0, existing_count, 10000):
                chunk = self.collection.get(limit=10000, offset=offset, include=[])
                existing_ids.update(chunk['ids'])

        # Filter out already embedded
        docs_to_embed = [d for d in documents if d["id"] not in existing_ids]
        print(f"Documents to embed: {len(docs_to_embed):,}")

        if not docs_to_embed:
            print("All documents already embedded!")
            return stats

        # Process in large batches
        save_every = 1000
        buf_ids, buf_emb, buf_meta, buf_docs = [], [], [], []

        iterator = range(0, len(docs_to_embed), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", total=len(docs_to_embed)//batch_size + 1)

        for i in iterator:
            batch = docs_to_embed[i:i + batch_size]
            texts = [doc["content"][:1000] for doc in batch]  # Truncate for speed

            # Batch encode (efficient on CPU)
            embeddings = self.model.encode(texts, convert_to_numpy=True,
                                          show_progress_bar=False,
                                          batch_size=batch_size)

            for j, doc in enumerate(batch):
                metadata = {
                    "table": doc["table"],
                    "primary_key": doc["primary_key"],
                    "primary_key_value": str(doc["primary_key_value"]),
                }
                raw_data = doc.get("raw_data", {})
                if "Name" in raw_data and raw_data["Name"]:
                    metadata["name"] = str(raw_data["Name"])[:500]

                buf_ids.append(doc["id"])
                buf_emb.append(embeddings[j].tolist())
                buf_meta.append(metadata)
                buf_docs.append(doc["content"][:5000])

            stats["embedded_documents"] += len(batch)

            # Save periodically
            if len(buf_ids) >= save_every:
                self.collection.add(ids=buf_ids, embeddings=buf_emb,
                                   metadatas=buf_meta, documents=buf_docs)
                buf_ids, buf_emb, buf_meta, buf_docs = [], [], [], []

        # Save remaining
        if buf_ids:
            self.collection.add(ids=buf_ids, embeddings=buf_emb,
                               metadatas=buf_meta, documents=buf_docs)

        stats["completed_at"] = datetime.now().isoformat()

        print("\n" + "="*60)
        print("EMBEDDING COMPLETE")
        print(f"Embedded: {stats['embedded_documents']:,}")
        print(f"Total in collection: {self.collection.count():,}")
        print("="*60 + "\n")

        return stats

    def reset_collection(self):
        """Reset the collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Reset collection: {self.collection_name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fast CPU embedding")
    parser.add_argument("--input", default="data/extracted_documents.json")
    parser.add_argument("--chroma-dir", default="chroma_db")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--tables", help="Comma-separated table names to include")
    parser.add_argument("--limit", type=int, help="Max documents to embed")

    args = parser.parse_args()

    print(f"\nLoading documents from: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents):,} documents")

    # Filter by tables if specified
    if args.tables:
        tables = [t.strip() for t in args.tables.split(",")]
        documents = [d for d in documents if d["table"] in tables]
        print(f"Filtered to {len(documents):,} docs from tables: {tables}")

    # Limit if specified
    if args.limit:
        documents = documents[:args.limit]
        print(f"Limited to {len(documents):,} documents")

    embedder = FastEmbedder(chroma_dir=args.chroma_dir)

    if args.reset:
        embedder.reset_collection()

    embedder.embed_documents(documents, batch_size=args.batch_size)

    print(f"\nCollection: {embedder.collection.count():,} documents")


if __name__ == "__main__":
    main()
