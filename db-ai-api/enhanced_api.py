"""
Enhanced FastAPI with Full RAG + SQL Support
Combines existing Text-to-SQL with new RAG capabilities and Ukrainian support
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import json
import os
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

SQL_MODEL = "qwen2.5:14b"
RAG_MODEL = "qwen2.5:14b"
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def get_hybrid_agent() -> HybridAgent:
    """Lazy load hybrid agent."""
    global _hybrid_agent
    if _hybrid_agent is None:
        print("Initializing Hybrid Agent...")
        _hybrid_agent = HybridAgent(
            sql_model=SQL_MODEL,
            rag_model=RAG_MODEL
        )
    return _hybrid_agent


def get_rag_engine() -> RAGQueryEngine:
    """Lazy load RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        print("Initializing RAG Engine...")
        _rag_engine = RAGQueryEngine(llm_model=RAG_MODEL)
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


class WebQueryRequest(BaseModel):
    question: str
    mode: Optional[Literal["sql", "rag", "auto"]] = "auto"


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
        "sql_model": SQL_MODEL,
        "rag_model": RAG_MODEL,
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


# Legacy /query endpoint removed - use /hybrid/query instead


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


@app.post("/web/query")
async def web_query(request: WebQueryRequest):
    """
    Website query endpoint - returns a stable response shape for UI clients.
    """
    try:
        agent = get_hybrid_agent()
        mode = None if request.mode == "auto" else request.mode

        result = agent.query(
            question=request.question,
            mode=mode
        )

        response: Dict[str, Any] = {
            "question": request.question,
            "mode": result.get("mode"),
            "success": result.get("success", False),
            "answer": result.get("answer"),
            "sql": result.get("sql"),
            "execution": None,
            "error": result.get("error")
        }

        if result.get("mode") == "sql":
            rows = result.get("results") or []
            columns = list(rows[0].keys()) if rows else []
            execution: Dict[str, Any] = {
                "success": result.get("success", False),
                "rows": rows,
                "columns": columns,
                "row_count": result.get("row_count", len(rows))
            }
            if result.get("error"):
                execution["error"] = result.get("error")
            response["execution"] = execution

        return response
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
                "sql_model": SQL_MODEL,
                "rag_model": RAG_MODEL,
                "embedding_model": RAG_EMBEDDING_MODEL
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


@app.get("/regions/stats")
async def get_region_stats():
    """Get client counts and sales statistics by region."""
    try:
        sql = """
        SELECT
            r.Name as RegionCode,
            COUNT(DISTINCT c.ID) as ClientCount,
            COALESCE(SUM(oi.Qty * oi.PricePerItem), 0) as TotalSales,
            COALESCE(SUM(oi.Qty), 0) as TotalQty
        FROM dbo.Region r
        LEFT JOIN dbo.Client c ON c.RegionID = r.ID AND c.Deleted = 0
        LEFT JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID
        LEFT JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID
        LEFT JOIN dbo.OrderItem oi ON oi.OrderID = o.ID
        WHERE r.Deleted = 0
        GROUP BY r.Name
        ORDER BY ClientCount DESC
        """

        result = execute_sql(sql)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to fetch region stats"))

        # Map region codes to full names
        region_names = {
            "KI": "Київ",
            "XM": "Хмельницький",
            "LV": "Львів",
            "OD": "Одеса",
            "XV": "Харків",
            "DP": "Дніпро",
        }

        regions = []
        for row in result.get("rows", []):
            code = row.get("RegionCode", "")
            regions.append({
                "region_code": code,
                "region_name": region_names.get(code, code),
                "client_count": row.get("ClientCount", 0),
                "total_sales": float(row.get("TotalSales", 0)),
                "total_qty": float(row.get("TotalQty", 0)),
            })

        return {
            "success": True,
            "regions": regions,
            "total_regions": len(regions),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/managers/stats")
async def get_manager_stats():
    """Get client counts by manager (User table via Client.MainManagerID)."""
    try:
        sql = """
        SELECT TOP 20
            u.ID as ManagerID,
            u.FirstName,
            u.LastName,
            ur.Name as Role,
            COUNT(c.ID) as ClientCount
        FROM dbo.[User] u
        LEFT JOIN dbo.Client c ON c.MainManagerID = u.ID AND c.Deleted = 0
        LEFT JOIN dbo.UserRole ur ON u.UserRoleID = ur.ID
        WHERE u.Deleted = 0 AND u.IsActive = 1
        GROUP BY u.ID, u.FirstName, u.LastName, ur.Name
        HAVING COUNT(c.ID) > 0
        ORDER BY ClientCount DESC
        """

        result = execute_sql(sql)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to fetch manager stats"))

        managers = []
        for row in result.get("rows", []):
            managers.append({
                "manager_id": row.get("ManagerID"),
                "first_name": row.get("FirstName", ""),
                "last_name": row.get("LastName", ""),
                "role": row.get("Role", ""),
                "client_count": row.get("ClientCount", 0),
            })

        return {
            "success": True,
            "managers": managers,
            "total_managers": len(managers)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    print("DB AI API - Hybrid SQL + RAG")
    print("="*60)
    print(f"Database: {DB_NAME}")
    print(f"Tables: {len(SCHEMA_CACHE.get('tables', {}))}")
    print(f"SQL Model: {SQL_MODEL}")
    print(f"RAG Model: {RAG_MODEL}")
    print(f"Embedding: {RAG_EMBEDDING_MODEL}")
    print("\nFeatures:")
    print("  - Text-to-SQL queries")
    print("  - RAG semantic search")
    print("  - Hybrid query routing")
    print("  - Ukrainian language support")
    print("\nEndpoints:")
    print("  API: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  Health: http://localhost:8000/health")
    print("\nNew Endpoints:")
    print("  /hybrid/query - Automatic SQL or RAG routing")
    print("  /rag/query - Pure RAG queries")
    print("  /rag/search - Semantic search")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
