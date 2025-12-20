"""Clean up old ChromaDB collections."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path

db_path = Path(__file__).parent / "chroma_db_examples_v2"

client = chromadb.PersistentClient(
    path=str(db_path),
    settings=ChromaSettings(anonymized_telemetry=False),
)

print("Existing collections:")
for col in client.list_collections():
    print(f"  - {col.name} ({col.count()} items)")

# Delete old collection
try:
    client.delete_collection("query_examples")
    print("\nDeleted old 'query_examples' collection")
except Exception as e:
    print(f"Could not delete: {e}")

print("\nRemaining collections:")
for col in client.list_collections():
    print(f"  - {col.name} ({col.count()} items)")
