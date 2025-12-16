"""
Enhanced FastAPI with Full RAG + SQL Support
Combines existing Text-to-SQL with new RAG capabilities and Ukrainian support
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import json
from pathlib import Path

# Import existing components
from simple_api import (
    get_db_connection, find_relevant_tables, build_context,
    generate_sql, execute_sql, SCHEMA_CACHE, DB_NAME, OLLAMA_MODEL
)

# Import new RAG components
from hybrid_agent import HybridAgent
from rag_engine import RAGQueryEngine
from embedder import RAGEmbedder
from utils.language_utils import detect_language, has_ukrainian

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="DB AI API - Hybrid SQL + RAG",
    description="Natural language database queries with Ukrainian support via SQL and RAG",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Initialize Components (Lazy Loading)
# ============================================================================

_hybrid_agent = None
_rag_engine = None
_embedder = None


def get_hybrid_agent() -> HybridAgent:
    """Lazy load hybrid agent."""
    global _hybrid_agent
    if _hybrid_agent is None:
        print("Initializing Hybrid Agent...")
        _hybrid_agent = HybridAgent(
            sql_model="qwen2:7b",
            rag_model="qwen2:7b"
        )
    return _hybrid_agent


def get_rag_engine() -> RAGQueryEngine:
    """Lazy load RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        print("Initializing RAG Engine...")
        _rag_engine = RAGQueryEngine(llm_model="qwen2:7b")
    return _rag_engine


def get_embedder() -> RAGEmbedder:
    """Lazy load embedder."""
    global _embedder
    if _embedder is None:
        print("Initializing Embedder...")
        _embedder = RAGEmbedder()
    return _embedder


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    question: str
    execute: bool = True
    max_rows: int = 100
    include_explanation: bool = False


class HybridQueryRequest(BaseModel):
    question: str
    mode: Optional[Literal["sql", "rag", "auto"]] = "auto"
    n_results: int = 5


class RAGQueryRequest(BaseModel):
    question: str
    n_results: int = 5
    return_context: bool = False


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


# ============================================================================
# Endpoints - Original SQL Endpoints (Backward Compatible)
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "DB AI API - Hybrid SQL + RAG with Ukrainian Support",
        "version": "2.0.0",
        "status": "running",
        "database": DB_NAME,
        "sql_model": "sqlcoder:7b",
        "rag_model": "qwen2:7b",
        "tables": len(SCHEMA_CACHE.get('tables', {})),
        "views": len(SCHEMA_CACHE.get('views', {})),
        "features": {
            "text_to_sql": True,
            "rag_search": True,
            "hybrid_queries": True,
            "ukrainian_support": True
        }
    }


@app.get("/health")
async def health():
    try:
        conn = get_db_connection()
        conn.close()
        return {
            "status": "healthy",
            "database": "connected",
            "ollama": "connected",
            "rag_system": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query")
async def query_database(request: QueryRequest):
    """Original Text-to-SQL endpoint (backward compatible)."""
    try:
        # Find relevant tables
        relevant_tables = find_relevant_tables(request.question, top_k=5)

        # Build context
        context = build_context(relevant_tables)

        # Generate SQL
        sql = generate_sql(request.question, context)

        # Build response
        response = {
            "question": request.question,
            "sql": sql,
            "explanation": None
        }

        # Execute if requested
        if request.execute:
            execution_result = execute_sql(sql, request.max_rows)
            response["execution"] = execution_result
        else:
            response["execution"] = {"message": "SQL generated but not executed"}

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema")
async def get_schema():
    """Get schema summary."""
    return {
        "database": SCHEMA_CACHE['database'],
        "tables": len(SCHEMA_CACHE.get('tables', {})),
        "views": len(SCHEMA_CACHE.get('views', {})),
        "total": len(SCHEMA_CACHE.get('tables', {})) + len(SCHEMA_CACHE.get('views', {}))
    }


# ============================================================================
# Endpoints - New Hybrid SQL + RAG Endpoints
# ============================================================================

@app.post("/hybrid/query")
async def hybrid_query(request: HybridQueryRequest):
    """
    Hybrid query endpoint - automatically routes to SQL or RAG.

    - mode="auto": Automatic classification
    - mode="sql": Force SQL generation
    - mode="rag": Force RAG search
    """
    try:
        agent = get_hybrid_agent()

        mode = None if request.mode == "auto" else request.mode

        result = agent.query(
            question=request.question,
            mode=mode
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    """
    Pure RAG query endpoint - semantic search + answer generation.
    """
    try:
        engine = get_rag_engine()

        result = engine.query(
            question=request.question,
            n_results=request.n_results,
            return_context=request.return_context
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/search")
async def rag_search(request: SearchRequest):
    """
    Semantic search in RAG database - returns relevant documents.
    """
    try:
        engine = get_rag_engine()

        result = engine.search(
            query=request.query,
            n_results=request.n_results
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoints - System Management
# ============================================================================

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = {
            "database": {
                "name": DB_NAME,
                "tables": len(SCHEMA_CACHE.get('tables', {})),
                "views": len(SCHEMA_CACHE.get('views', {}))
            },
            "models": {
                "sql_model": "sqlcoder:7b",
                "rag_model": "qwen2:7b",
                "embedding_model": "multilingual-e5-large"
            }
        }

        # Add RAG stats if initialized
        if _rag_engine is not None:
            rag_stats = _rag_engine.get_stats()
            stats["rag"] = rag_stats

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages/detect")
async def detect_query_language(query: str):
    """Detect language of query text."""
    language = detect_language(query)
    has_uk = has_ukrainian(query)

    return {
        "query": query,
        "detected_language": language,
        "has_ukrainian": has_uk
    }


# ============================================================================
# Endpoints - Data Management (Admin)
# ============================================================================

@app.post("/admin/index")
async def trigger_indexing():
    """
    Trigger full data extraction and indexing.
    WARNING: This is a long-running operation!
    """
    try:
        # Import here to avoid loading at startup
        from data_extractor import DataExtractor

        extractor = DataExtractor()

        # Extract data (test mode: 5 tables)
        stats = extractor.extract_all_data(max_tables=5)

        return {
            "message": "Indexing started (test mode: 5 tables)",
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/embed")
async def trigger_embedding():
    """
    Trigger embedding of extracted documents.
    WARNING: This is a long-running operation!
    """
    try:
        embedder = get_embedder()

        # Load documents
        with open("data/extracted_documents.json", "r", encoding="utf-8") as f:
            documents = json.load(f)

        # Embed (test mode: 100 docs)
        stats = embedder.embed_documents(documents[:100])

        return {
            "message": "Embedding complete (test mode: 100 documents)",
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 DB AI API - Hybrid SQL + RAG")
    print("="*60)
    print(f"Database: {DB_NAME}")
    print(f"Tables: {len(SCHEMA_CACHE.get('tables', {}))}")
    print(f"SQL Model: sqlcoder:7b")
    print(f"RAG Model: qwen2:7b")
    print(f"Embedding: multilingual-e5-large")
    print("\n✨ Features:")
    print("  - Text-to-SQL queries")
    print("  - RAG semantic search")
    print("  - Hybrid query routing")
    print("  - Ukrainian language support")
    print("\n🌐 Endpoints:")
    print("  API: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  Health: http://localhost:8000/health")
    print("\n📚 New Endpoints:")
    print("  /hybrid/query - Automatic SQL or RAG routing")
    print("  /rag/query - Pure RAG queries")
    print("  /rag/search - Semantic search")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
