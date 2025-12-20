"""Test RAG system with sample queries."""
import chromadb
from sentence_transformers import SentenceTransformer

# Load model and ChromaDB
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("concorddb_ukrainian")
print(f"Collection has {collection.count():,} documents")

# Test queries
queries = [
    "clients from Kyiv",
    "products with high price",
    "unpaid debts",
    "orders in 2024"
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print("="*50)
    embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - dist
        table = meta.get('table', '?')
        name = meta.get('name', '')
        pk = meta.get('primary_key_value', '')

        print(f"\n{i+1}. [{table}] ID:{pk} Score: {similarity:.3f}")
        if name:
            print(f"   Name: {name}")
        doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
        print(f"   {doc_preview}")

print("\n" + "="*50)
print("RAG System Test Complete!")
print("="*50)
