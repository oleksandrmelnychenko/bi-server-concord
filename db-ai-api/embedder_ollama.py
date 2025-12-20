"""
Embedder for RAG System using Ollama
Creates embeddings using Ollama's nomic-embed-text model and stores in ChromaDB
"""
import json
import os
import requests
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from datetime import datetime


class OllamaEmbedder:
    """Create embeddings using Ollama and store in ChromaDB."""

    def __init__(self,
                 model_name: str = "nomic-embed-text",
                 ollama_url: str = "http://localhost:11434",
                 chroma_dir: str = "chroma_db",
                 collection_name: str = "concorddb_ukrainian"):
        """
        Initialize OllamaEmbedder.

        Args:
            model_name: Ollama model name for embeddings
            ollama_url: Ollama API URL
            chroma_dir: Directory for ChromaDB storage
            collection_name: ChromaDB collection name
        """
        print("\n" + "="*60)
        print("INITIALIZING OLLAMA EMBEDDER")
        print("="*60 + "\n")

        self.model_name = model_name
        self.ollama_url = ollama_url
        self.embedding_dim = None

        # Test Ollama connection
        print(f"Testing Ollama connection: {ollama_url}")
        test_embedding = self._get_embedding("test")
        if test_embedding:
            self.embedding_dim = len(test_embedding)
            print(f"OK - Model: {model_name}")
            print(f"  Embedding dimension: {self.embedding_dim}")
        else:
            raise ConnectionError("Failed to connect to Ollama")

        # Initialize ChromaDB
        print(f"\nInitializing ChromaDB: {chroma_dir}")
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
            print(f"  Documents in collection: {self.collection.count()}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")

        self.collection_name = collection_name

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text using Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            return None
        except Exception as e:
            return None

    def _get_embeddings_batch(self, texts: List[str], max_workers: int = 4) -> List[List[float]]:
        """Get embeddings for multiple texts in parallel."""
        embeddings = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._get_embedding, text): idx
                for idx, text in enumerate(texts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception:
                    embeddings[idx] = None

        return embeddings

    def embed_documents(self, documents: List[Dict[str, Any]],
                       batch_size: int = 50,
                       workers: int = 8,
                       show_progress: bool = True) -> Dict[str, Any]:
        """
        Embed documents and store in ChromaDB.

        Args:
            documents: List of documents with 'content' and metadata
            batch_size: Batch size for embedding
            workers: Number of parallel workers for Ollama requests
            show_progress: Show progress bar

        Returns:
            Statistics about embedding process
        """
        print("\n" + "="*60)
        print("EMBEDDING DOCUMENTS WITH OLLAMA")
        print("="*60 + "\n")

        stats = {
            "total_documents": len(documents),
            "embedded_documents": 0,
            "skipped_documents": 0,
            "errors": [],
            "started_at": datetime.now().isoformat()
        }

        print(f"Total documents to embed: {len(documents)}")
        print(f"Batch size: {batch_size}")
        print(f"Parallel workers: {workers}\n")

        # Resume support: get already embedded IDs
        existing_count = self.collection.count()
        existing_ids = set()
        if existing_count > 0:
            print(f"Found {existing_count:,} existing embeddings - enabling resume mode")
            # Get existing IDs in chunks
            for offset in range(0, existing_count, 10000):
                chunk = self.collection.get(limit=10000, offset=offset, include=[])
                existing_ids.update(chunk['ids'])
            print(f"Loaded {len(existing_ids):,} existing IDs to skip")

        # Buffer for incremental saving
        save_every = 500
        buf_ids = []
        buf_emb = []
        buf_meta = []
        buf_docs = []

        # Process in batches
        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches",
                          total=len(documents)//batch_size + 1)

        for i in iterator:
            batch = documents[i:i + batch_size]
            # Skip already embedded docs
            batch = [doc for doc in batch if doc["id"] not in existing_ids]
            if not batch:
                continue

            try:
                texts = [doc["content"] for doc in batch]
                embeddings = self._get_embeddings_batch(texts, max_workers=workers)

                for j, doc in enumerate(batch):
                    if embeddings[j] is None:
                        stats["skipped_documents"] += 1
                        continue

                    metadata = {
                        "table": doc["table"],
                        "primary_key": doc["primary_key"],
                        "primary_key_value": str(doc["primary_key_value"]),
                        "created_at": doc["created_at"]
                    }
                    raw_data = doc.get("raw_data", {})
                    if "Name" in raw_data and raw_data["Name"]:
                        metadata["name"] = str(raw_data["Name"])[:500]
                    if "City" in raw_data and raw_data["City"]:
                        metadata["city"] = str(raw_data["City"])[:100]

                    buf_ids.append(doc["id"])
                    buf_emb.append(embeddings[j])
                    buf_meta.append(metadata)
                    buf_docs.append(doc["content"])

                stats["embedded_documents"] += sum(1 for e in embeddings if e is not None)

                # Save to ChromaDB every save_every docs
                if len(buf_ids) >= save_every:
                    self.collection.add(ids=buf_ids, embeddings=buf_emb,
                                       metadatas=buf_meta, documents=buf_docs)
                    buf_ids, buf_emb, buf_meta, buf_docs = [], [], [], []

            except Exception as e:
                error_msg = f"Error at batch {i}: {str(e)}"
                stats["errors"].append(error_msg)
                print(f"\nError: {error_msg}")
                stats["skipped_documents"] += len(batch)

        # Save remaining
        if buf_ids:
            self.collection.add(ids=buf_ids, embeddings=buf_emb,
                               metadatas=buf_meta, documents=buf_docs)

        stats["completed_at"] = datetime.now().isoformat()
        self.print_stats(stats)

        return stats

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector database."""
        query_embedding = self._get_embedding(query_text)
        if not query_embedding:
            return {"error": "Failed to get query embedding"}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "query": query_text,
            "results": results,
            "n_results": n_results
        }

    def print_stats(self, stats: Dict[str, Any]) -> None:
        """Print embedding statistics."""
        print("\n" + "="*60)
        print("EMBEDDING SUMMARY")
        print("="*60)
        print(f"Total documents: {stats['total_documents']}")
        print(f"Embedded successfully: {stats['embedded_documents']}")
        print(f"Skipped: {stats['skipped_documents']}")
        print(f"Errors: {len(stats['errors'])}")

        if stats["errors"]:
            print("\nErrors encountered:")
            for error in stats["errors"][:5]:
                print(f"  - {error}")

        print("="*60 + "\n")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "embedding_dimension": self.embedding_dim,
            "model": self.model_name
        }

    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        print(f"\nResetting collection: {self.collection_name}")

        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Deleted collection")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {self.collection_name}\n")


def main():
    """Main embedding function."""
    import argparse

    parser = argparse.ArgumentParser(description="Embed documents using Ollama")
    parser.add_argument("--input", default="data/extracted_documents.json",
                       help="Input documents JSON file")
    parser.add_argument("--chroma-dir", default="chroma_db",
                       help="ChromaDB directory")
    parser.add_argument("--collection", default="concorddb_ukrainian",
                       help="Collection name")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for embedding")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--reset", action="store_true",
                       help="Reset collection before embedding")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (1000 documents only)")

    args = parser.parse_args()

    # Load documents
    print(f"\nLoading documents from: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents")

    # Test mode
    if args.test:
        documents = documents[:1000]
        print(f"Test mode: using only {len(documents)} documents")

    # Create embedder
    embedder = OllamaEmbedder(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection
    )

    # Reset if requested
    if args.reset:
        embedder.reset_collection()

    # Embed documents
    stats = embedder.embed_documents(
        documents=documents,
        batch_size=args.batch_size,
        workers=args.workers
    )

    # Print final stats
    print("\nEmbedding complete!")
    collection_stats = embedder.get_collection_stats()
    print(f"\nCollection Statistics:")
    print(f"  Collection: {collection_stats['collection_name']}")
    print(f"  Documents: {collection_stats['document_count']}")
    print(f"  Embedding dimension: {collection_stats['embedding_dimension']}")

    return stats


if __name__ == "__main__":
    main()
