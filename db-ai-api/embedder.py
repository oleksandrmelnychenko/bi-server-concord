"""
Embedder for RAG System
Creates embeddings using multilingual-e5-large and stores in ChromaDB
"""
import json
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from datetime import datetime


class RAGEmbedder:
    """Create and manage embeddings for RAG system."""

    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-large",
                 chroma_dir: str = "chroma_db",
                 collection_name: str = "concorddb_ukrainian"):
        """
        Initialize RAGEmbedder.

        Args:
            model_name: HuggingFace model name for embeddings
            chroma_dir: Directory for ChromaDB storage
            collection_name: ChromaDB collection name
        """
        print("\n" + "="*60)
        print("🚀 INITIALIZING EMBEDDER")
        print("="*60 + "\n")

        # Load embedding model
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded successfully")
        print(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB
        print(f"\n💾 Initializing ChromaDB: {chroma_dir}")
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
            print(f"✓ Loaded existing collection: {collection_name}")
            print(f"  Documents in collection: {self.collection.count()}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Created new collection: {collection_name}")

        self.collection_name = collection_name

    def embed_documents(self, documents: List[Dict[str, Any]],
                       batch_size: int = 32,
                       show_progress: bool = True) -> Dict[str, Any]:
        """
        Embed documents and store in ChromaDB.

        Args:
            documents: List of documents with 'content' and metadata
            batch_size: Batch size for embedding
            show_progress: Show progress bar

        Returns:
            Statistics about embedding process
        """
        print("\n" + "="*60)
        print("🎯 EMBEDDING DOCUMENTS")
        print("="*60 + "\n")

        stats = {
            "total_documents": len(documents),
            "embedded_documents": 0,
            "skipped_documents": 0,
            "errors": [],
            "started_at": datetime.now().isoformat()
        }

        print(f"📊 Total documents to embed: {len(documents)}")
        print(f"📦 Batch size: {batch_size}\n")

        # Prepare data for ChromaDB
        all_ids = []
        all_embeddings = []
        all_metadatas = []
        all_documents = []

        # Process in batches
        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches")

        for i in iterator:
            batch = documents[i:i + batch_size]

            try:
                # Extract texts
                texts = [doc["content"] for doc in batch]

                # Create embeddings
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

                # Prepare for ChromaDB
                for j, doc in enumerate(batch):
                    doc_id = doc["id"]

                    # Metadata (ChromaDB doesn't support nested dicts, so flatten)
                    metadata = {
                        "table": doc["table"],
                        "primary_key": doc["primary_key"],
                        "primary_key_value": str(doc["primary_key_value"]),
                        "created_at": doc["created_at"]
                    }

                    # Add a few key fields from raw_data to metadata
                    raw_data = doc.get("raw_data", {})
                    if "Name" in raw_data and raw_data["Name"]:
                        metadata["name"] = str(raw_data["Name"])[:500]
                    if "City" in raw_data and raw_data["City"]:
                        metadata["city"] = str(raw_data["City"])[:100]

                    all_ids.append(doc_id)
                    all_embeddings.append(embeddings[j].tolist())
                    all_metadatas.append(metadata)
                    all_documents.append(doc["content"])

                stats["embedded_documents"] += len(batch)

            except Exception as e:
                error_msg = f"Error embedding batch at index {i}: {str(e)}"
                stats["errors"].append(error_msg)
                print(f"\n✗ {error_msg}")
                stats["skipped_documents"] += len(batch)

        # Store in ChromaDB
        if all_ids:
            print("\n💾 Storing embeddings in ChromaDB...")

            try:
                # Add in chunks to avoid memory issues
                chunk_size = 1000
                for i in tqdm(range(0, len(all_ids), chunk_size), desc="Storing chunks"):
                    end_idx = min(i + chunk_size, len(all_ids))

                    self.collection.add(
                        ids=all_ids[i:end_idx],
                        embeddings=all_embeddings[i:end_idx],
                        metadatas=all_metadatas[i:end_idx],
                        documents=all_documents[i:end_idx]
                    )

                print(f"✓ Stored {len(all_ids)} embeddings in ChromaDB")

            except Exception as e:
                error_msg = f"Error storing in ChromaDB: {str(e)}"
                stats["errors"].append(error_msg)
                print(f"✗ {error_msg}")

        stats["completed_at"] = datetime.now().isoformat()

        # Print summary
        self.print_stats(stats)

        return stats

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector database.

        Args:
            query_text: Query text
            n_results: Number of results to return

        Returns:
            Query results with documents and metadata
        """
        # Create query embedding
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
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
        print("📊 EMBEDDING SUMMARY")
        print("="*60)
        print(f"Total documents: {stats['total_documents']}")
        print(f"Embedded successfully: {stats['embedded_documents']}")
        print(f"Skipped: {stats['skipped_documents']}")
        print(f"Errors: {len(stats['errors'])}")

        if stats["errors"]:
            print("\n⚠️  Errors encountered:")
            for error in stats["errors"][:5]:  # Show first 5
                print(f"  - {error}")

        print("="*60 + "\n")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "model": self.model._model_config.get("name", "unknown")
        }

    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        print(f"\n⚠️  Resetting collection: {self.collection_name}")

        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"✓ Deleted collection")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Created new collection: {self.collection_name}\n")


def main():
    """Main embedding function."""
    import argparse

    parser = argparse.ArgumentParser(description="Embed documents for RAG system")
    parser.add_argument("--input", default="data/extracted_documents.json",
                       help="Input documents JSON file")
    parser.add_argument("--chroma-dir", default="chroma_db",
                       help="ChromaDB directory")
    parser.add_argument("--collection", default="concorddb_ukrainian",
                       help="Collection name")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding")
    parser.add_argument("--reset", action="store_true",
                       help="Reset collection before embedding")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (100 documents only)")

    args = parser.parse_args()

    # Load documents
    print(f"\n📂 Loading documents from: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"✓ Loaded {len(documents)} documents")

    # Test mode
    if args.test:
        documents = documents[:100]
        print(f"⚠️  Test mode: using only {len(documents)} documents")

    # Create embedder
    embedder = RAGEmbedder(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection
    )

    # Reset if requested
    if args.reset:
        embedder.reset_collection()

    # Embed documents
    stats = embedder.embed_documents(
        documents=documents,
        batch_size=args.batch_size
    )

    # Print final stats
    print("\n✅ Embedding complete!")
    collection_stats = embedder.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"  Collection: {collection_stats['collection_name']}")
    print(f"  Documents: {collection_stats['document_count']}")
    print(f"  Embedding dimension: {collection_stats['embedding_dimension']}")

    return stats


if __name__ == "__main__":
    main()
