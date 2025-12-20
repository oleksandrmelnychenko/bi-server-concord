"""Reset and re-embed with multilingual model."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from pathlib import Path
import json
from datetime import datetime

# Paths
db_path = Path(__file__).parent / "chroma_db_examples_v2"
templates_dir = Path(__file__).parent / "training_data" / "templates"

MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "query_examples_multilingual"

print("=" * 60)
print("RESETTING EMBEDDINGS WITH MULTILINGUAL MODEL")
print("=" * 60)

# Delete entire chroma_db_examples_v2 folder and recreate
import shutil
if db_path.exists():
    shutil.rmtree(db_path)
    print(f"Deleted {db_path}")

db_path.mkdir(parents=True, exist_ok=True)

# Initialize
print(f"\nLoading model: {MULTILINGUAL_MODEL}")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MULTILINGUAL_MODEL
)

client = chromadb.PersistentClient(
    path=str(db_path),
    settings=ChromaSettings(anonymized_telemetry=False),
)

print(f"Creating collection: {COLLECTION_NAME}")
collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# Embed all templates
total_embedded = 0

for filepath in templates_dir.glob("*.json"):
    print(f"\nProcessing: {filepath.name}")

    with open(filepath, "r", encoding="utf-8") as f:
        template = json.load(f)

    domain = template.get("domain", "unknown")
    examples = template.get("examples", [])

    documents = []
    metadatas = []
    ids = []

    for ex in examples:
        ex_id = ex.get("id", "")
        if not ex_id:
            continue

        # Create document text
        parts = []
        if ex.get("question_en"):
            parts.append(ex["question_en"])
        if ex.get("question_uk"):
            parts.append(ex["question_uk"])
        for v in ex.get("variations_en", []):
            parts.append(v)
        for v in ex.get("variations_uk", []):
            parts.append(v)

        doc = " | ".join(parts)

        metadata = {
            "id": ex_id,
            "domain": domain,
            "category": ex.get("category", ""),
            "complexity": ex.get("complexity", ""),
            "question_en": ex.get("question_en", ""),
            "question_uk": ex.get("question_uk", ""),
            "sql": ex.get("sql", ""),
            "tables_used": ",".join(ex.get("tables_used", [])),
        }

        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"{domain}_{ex_id}")

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"  Embedded {len(documents)} examples")
        total_embedded += len(documents)

print(f"\n{'='*60}")
print(f"COMPLETE: {total_embedded} examples embedded")
print(f"Collection: {collection.count()} items")
print("=" * 60)
