"""
Simple RAG API Server
Exposes semantic search over ChromaDB embedded documents
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uvicorn


# Module-level embedding cache (saves 100-300ms per cached hit)
@lru_cache(maxsize=1024)
def _get_cached_embedding(query: str) -> Tuple[float, ...]:
    """Cache query embeddings to avoid recomputation."""
    global model
    if model is None:
        raise ValueError("Model not initialized")
    return tuple(model.encode(query).tolist())

# Configuration - Updated to use full collection with all 2.28M documents
CHROMA_DIR = "chroma_db_full"
COLLECTION_NAME = "concorddb_full"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

app = FastAPI(
    title="RAG API - Semantic Search",
    description="Semantic search over ConcordDb documents",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model = None
collection = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    n_results: int = Field(default=5, ge=1, le=50, description="Number of results")
    table_filter: Optional[str] = Field(default=None, description="Filter by table name")


class SearchResult(BaseModel):
    rank: int
    table: str
    primary_key: str
    primary_key_value: str
    name: Optional[str]
    similarity: float
    document: str


class SearchResponse(BaseModel):
    query: str
    n_results: int
    results: List[SearchResult]


@app.on_event("startup")
async def startup():
    """Initialize model and ChromaDB on startup."""
    global model, collection

    print("\n" + "="*60)
    print("STARTING RAG API SERVER")
    print("="*60 + "\n")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    print(f"OK - Dimension: {model.get_sentence_embedding_dimension()}")

    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB: {CHROMA_DIR}")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"OK - Collection: {COLLECTION_NAME}")
    print(f"Documents: {collection.count():,}")

    print("\n" + "="*60)
    print("RAG API READY")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "RAG API - Semantic Search",
        "version": "1.0.0",
        "status": "running",
        "collection": COLLECTION_NAME,
        "documents": collection.count() if collection else 0,
        "embedding_model": EMBEDDING_MODEL
    }


@app.get("/health")
async def health():
    """Health check."""
    if model is None or collection is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {
        "status": "healthy",
        "documents": collection.count()
    }


@app.get("/stats")
async def stats():
    """Get collection statistics."""
    if collection is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Get sample to show tables
    sample = collection.get(limit=100, include=["metadatas"])
    tables = set()
    for meta in sample["metadatas"]:
        tables.add(meta.get("table", "unknown"))

    return {
        "total_documents": collection.count(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "sample_tables": sorted(list(tables))
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Semantic search over documents.

    Returns documents ranked by similarity to the query.
    """
    if model is None or collection is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Create query embedding (cached for repeated queries)
        query_embedding = list(_get_cached_embedding(request.query))

        # Build where filter if table specified
        where_filter = None
        if request.table_filter:
            where_filter = {"table": request.table_filter}

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        search_results = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 - dist  # Convert distance to similarity
            search_results.append(SearchResult(
                rank=i + 1,
                table=meta.get("table", "unknown"),
                primary_key=meta.get("primary_key", "unknown"),
                primary_key_value=meta.get("primary_key_value", ""),
                name=meta.get("name"),
                similarity=round(similarity, 4),
                document=doc[:500] + "..." if len(doc) > 500 else doc
            ))

        return SearchResponse(
            query=request.query,
            n_results=len(search_results),
            results=search_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_get(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=5, ge=1, le=50, description="Number of results"),
    table: Optional[str] = Query(default=None, description="Filter by table")
):
    """GET version of search endpoint."""
    request = SearchRequest(query=q, n_results=n, table_filter=table)
    return await search(request)


@app.get("/tables")
async def list_tables():
    """List all tables in the collection."""
    if collection is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Sample documents to get table list
    sample = collection.get(limit=10000, include=["metadatas"])

    table_counts = {}
    for meta in sample["metadatas"]:
        table = meta.get("table", "unknown")
        table_counts[table] = table_counts.get(table, 0) + 1

    # Sort by count
    sorted_tables = sorted(table_counts.items(), key=lambda x: -x[1])

    return {
        "tables": [
            {"name": name, "document_count": count}
            for name, count in sorted_tables
        ],
        "total_tables": len(sorted_tables)
    }


if __name__ == "__main__":
    print("\nStarting RAG API Server...")
    print("API will be available at: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
