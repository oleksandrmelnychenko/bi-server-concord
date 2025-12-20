# -*- coding: utf-8 -*-
"""
Embed ALL documents with multilingual model
Uses paraphrase-multilingual-MiniLM-L12-v2 for Ukrainian support
"""
import json
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from datetime import datetime

# Configuration
INPUT_FILE = "data/all_documents.json"
CHROMA_DIR = "chroma_db_full"
COLLECTION_NAME = "concorddb_full"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 128
SAVE_EVERY = 1000


def main():
    print("\n" + "="*60)
    print("EMBEDDING ALL DOCUMENTS")
    print("="*60 + "\n")

    # Load documents
    print(f"Loading documents from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"Loaded: {len(documents):,} documents\n")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    print(f"OK - Dimension: {model.get_sentence_embedding_dimension()}\n")

    # Initialize ChromaDB
    print(f"Initializing ChromaDB: {CHROMA_DIR}")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    # Check for existing collection
    try:
        existing = client.get_collection(name=COLLECTION_NAME)
        existing_count = existing.count()
        print(f"Found existing collection with {existing_count:,} documents")

        # Get existing IDs for resume
        existing_ids = set()
        for offset in range(0, existing_count, 10000):
            chunk = existing.get(limit=10000, offset=offset, include=[])
            existing_ids.update(chunk['ids'])
        print(f"Loaded {len(existing_ids):,} existing IDs for resume")
        collection = existing
    except:
        print("Creating new collection...")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        existing_ids = set()

    # Filter documents to embed
    docs_to_embed = [d for d in documents if d["id"] not in existing_ids]
    print(f"\nDocuments to embed: {len(docs_to_embed):,}")
    print(f"Already embedded: {len(existing_ids):,}")

    if not docs_to_embed:
        print("\nAll documents already embedded!")
        return

    # Stats
    stats = {
        "total_documents": len(documents),
        "to_embed": len(docs_to_embed),
        "started_at": datetime.now().isoformat()
    }

    # Embedding buffers
    buf_ids, buf_emb, buf_meta, buf_docs = [], [], [], []
    embedded_count = 0

    # Process in batches
    print(f"\nStarting embedding (batch size: {BATCH_SIZE})...\n")

    for i in tqdm(range(0, len(docs_to_embed), BATCH_SIZE), desc="Embedding"):
        batch = docs_to_embed[i:i + BATCH_SIZE]

        # Prepare texts (limit to 2000 chars)
        texts = [doc["content"][:2000] for doc in batch]

        # Encode batch
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE
        )

        # Add to buffers
        for j, doc in enumerate(batch):
            metadata = {
                "table": doc["table"],
                "primary_key": doc["primary_key"],
                "primary_key_value": str(doc["primary_key_value"]),
            }

            # Add name if available
            raw_data = doc.get("raw_data", {})
            if "Name" in raw_data and raw_data["Name"]:
                metadata["name"] = str(raw_data["Name"])[:500]

            # Add region if available
            if "RegionID" in raw_data:
                metadata["region_id"] = str(raw_data["RegionID"])

            buf_ids.append(doc["id"])
            buf_emb.append(embeddings[j].tolist())
            buf_meta.append(metadata)
            buf_docs.append(texts[j][:5000])

        embedded_count += len(batch)

        # Save periodically
        if len(buf_ids) >= SAVE_EVERY:
            collection.add(
                ids=buf_ids,
                embeddings=buf_emb,
                metadatas=buf_meta,
                documents=buf_docs
            )
            buf_ids, buf_emb, buf_meta, buf_docs = [], [], [], []

    # Save remaining
    if buf_ids:
        collection.add(
            ids=buf_ids,
            embeddings=buf_emb,
            metadatas=buf_meta,
            documents=buf_docs
        )

    stats["completed_at"] = datetime.now().isoformat()
    stats["embedded"] = embedded_count

    # Save stats
    with open("data/embedding_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("EMBEDDING COMPLETE")
    print("="*60)
    print(f"Embedded: {embedded_count:,}")
    print(f"Total in collection: {collection.count():,}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Directory: {CHROMA_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
