"""FastAPI service for Text-to-SQL API."""
import asyncio
from asyncio import Semaphore
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from config import settings
from sql_agent import SQLAgent
from schema_extractor import SchemaExtractor
from table_selector import TableSelector

# Semaphore to limit concurrent LLM calls (prevents Ollama contention)
query_semaphore = Semaphore(2)  # Max 2 concurrent LLM generations


# Initialize FastAPI app
app = FastAPI(
    title="DB AI API - Text to SQL",
    description="Natural language to SQL query API powered by local LLM",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Gzip compression for responses > 1KB (20-50x smaller for large result sets)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for natural language query."""

    question: str = Field(..., description="Natural language question")
    execute: bool = Field(
        default=True, description="Whether to execute the generated SQL"
    )
    max_rows: Optional[int] = Field(
        default=None,
        description=f"Maximum rows to return (default: {settings.max_rows_returned})",
    )
    include_explanation: bool = Field(
        default=False, description="Include explanation of the SQL query"
    )


class SQLExecuteRequest(BaseModel):
    """Request model for direct SQL execution."""

    sql: str = Field(..., description="SQL query to execute")
    max_rows: Optional[int] = Field(
        default=None, description="Maximum rows to return"
    )


class QueryResponse(BaseModel):
    """Response model for query results."""

    question: str
    sql: str
    explanation: Optional[str] = None
    generation_attempts: Optional[int] = None
    attempts: Optional[int] = None  # Legacy field from sql_agent
    execution: Optional[Dict[str, Any]] = None


class TableInfo(BaseModel):
    """Table information model."""

    name: str
    type: str
    row_count: int
    column_count: int


class SchemaResponse(BaseModel):
    """Response model for schema endpoint."""

    database: str
    tables: List[TableInfo]
    views: List[TableInfo]
    total_tables: int
    total_views: int


# Global instances (initialized on startup)
agent: Optional[SQLAgent] = None
schema_extractor: Optional[SchemaExtractor] = None
table_selector: Optional[TableSelector] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global agent, schema_extractor, table_selector

    logger.info("Initializing DB AI API...")

    try:
        # Initialize components
        schema_extractor = SchemaExtractor()
        table_selector = TableSelector(schema_extractor)
        agent = SQLAgent(
            schema_extractor=schema_extractor, table_selector=table_selector
        )

        # Index schema on startup
        logger.info("Indexing database schema...")
        table_selector.index_schema()

        logger.info("DB AI API initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DB AI API - Text to SQL",
        "version": "1.0.0",
        "status": "running",
        "database": settings.db_name,
        "model": settings.ollama_model,
        "read_only_mode": settings.read_only_mode,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with component status."""
    components = {}
    all_healthy = True

    # Check database connection
    try:
        with schema_extractor.engine.connect() as conn:
            pass
        components["database"] = {"status": "ok"}
    except Exception as e:
        components["database"] = {"status": "error", "message": str(e)}
        all_healthy = False

    # Check Ollama
    try:
        agent.ollama_client.list()
        components["ollama"] = {"status": "ok"}
    except Exception as e:
        components["ollama"] = {"status": "error", "message": str(e)}
        all_healthy = False

    # Check query examples
    if agent.example_retriever.is_available():
        stats = agent.example_retriever.get_stats()
        components["query_examples"] = {
            "status": "ok",
            "count": stats["total_examples"],
            "domains": stats.get("domains", {})
        }
    else:
        components["query_examples"] = {"status": "unavailable"}

    response = {
        "status": "healthy" if all_healthy else "degraded",
        "components": components,
        "timestamp": datetime.now().isoformat(),
    }

    if not all_healthy:
        raise HTTPException(status_code=503, detail=response)

    return response


@app.get("/query-examples/stats")
async def query_examples_stats():
    """Get query examples statistics."""
    if not agent.example_retriever.is_available():
        raise HTTPException(status_code=503, detail="Query examples not available")
    return agent.example_retriever.get_stats()


@app.get("/query-examples/test")
async def test_query_examples(query: str = Query(..., description="Test query")):
    """Test query example retrieval for a specific query."""
    if not agent.example_retriever.is_available():
        raise HTTPException(status_code=503, detail="Query examples not available")

    examples = agent.example_retriever.find_similar_with_correction(query, top_k=3)
    return {
        "query": query,
        "examples_found": len(examples),
        "examples": examples
    }


@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """
    Generate SQL from natural language and optionally execute it.

    Uses async LLM calls for concurrent request handling.
    Limited to 2 concurrent LLM calls to prevent Ollama contention.
    Includes timeout for better SLA.

    Args:
        request: Query request with natural language question

    Returns:
        QueryResponse with SQL and optional results
    """
    try:
        logger.info(f"Received query: {request.question}")

        # Use semaphore to limit concurrent LLM calls (prevents Ollama thrashing)
        async with query_semaphore:
            # Apply timeout for better SLA
            result = await asyncio.wait_for(
                agent.query_async(
                    question=request.question,
                    execute=request.execute,
                    max_rows=request.max_rows,
                    include_explanation=request.include_explanation,
                ),
                timeout=settings.query_timeout
            )

        return QueryResponse(**result)

    except asyncio.TimeoutError:
        logger.error(f"Query timeout after {settings.query_timeout}s: {request.question}")
        raise HTTPException(status_code=504, detail=f"Query generation timeout ({settings.query_timeout}s)")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute_sql(request: SQLExecuteRequest):
    """
    Execute a SQL query directly (bypassing LLM generation).

    Args:
        request: SQL execution request

    Returns:
        Execution results
    """
    try:
        logger.info(f"Executing SQL: {request.sql[:100]}...")

        result = agent.execute_sql(sql_query=request.sql, max_rows=request.max_rows)

        return result

    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema", response_model=SchemaResponse)
async def get_schema():
    """
    Get database schema information.

    Returns:
        Schema information including tables and views
    """
    try:
        schema = schema_extractor.extract_full_schema()

        tables = [
            TableInfo(
                name=name,
                type="table",
                row_count=info.get("row_count", 0),
                column_count=len(info.get("columns", [])),
            )
            for name, info in schema["tables"].items()
        ]

        views = [
            TableInfo(
                name=name,
                type="view",
                row_count=info.get("row_count", 0),
                column_count=len(info.get("columns", [])),
            )
            for name, info in schema["views"].items()
        ]

        return SchemaResponse(
            database=schema["database"],
            tables=tables,
            views=views,
            total_tables=len(tables),
            total_views=len(views),
        )

    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema/table/{table_name}")
async def get_table_details(table_name: str):
    """
    Get detailed information about a specific table.

    Args:
        table_name: Name of the table

    Returns:
        Detailed table information
    """
    try:
        summary = schema_extractor.get_table_summary(table_name)

        if not summary:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

        # Also get the structured data
        schema = schema_extractor.extract_full_schema()
        table_info = schema["tables"].get(table_name) or schema["views"].get(table_name)

        return {
            "table_name": table_name,
            "summary": summary,
            "details": table_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get table details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schema/refresh")
async def refresh_schema():
    """
    Force refresh of the database schema cache.

    Returns:
        Status message
    """
    try:
        logger.info("Refreshing schema cache...")

        # Refresh schema
        schema = schema_extractor.extract_full_schema(force_refresh=True)

        # Re-index for RAG
        table_selector.index_schema(force_refresh=True)

        return {
            "status": "success",
            "message": "Schema refreshed successfully",
            "tables_indexed": len(schema["tables"]) + len(schema["views"]),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to refresh schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_query(question: str = Body(..., embed=True)):
    """
    Generate SQL and explanation without executing.

    Uses async LLM calls for concurrent request handling.
    Limited to 2 concurrent LLM calls with timeout.

    Args:
        question: Natural language question

    Returns:
        SQL query with explanation
    """
    try:
        # Use semaphore and timeout for consistency with /query
        async with query_semaphore:
            result = await asyncio.wait_for(
                agent.generate_sql_async(question, include_explanation=True),
                timeout=settings.query_timeout
            )

        return {
            "question": question,
            "sql": result["sql"],
            "explanation": result.get("explanation"),
            "relevant_tables": table_selector.find_relevant_tables(question, top_k=5),
        }

    except asyncio.TimeoutError:
        logger.error(f"Explain timeout after {settings.query_timeout}s: {question}")
        raise HTTPException(status_code=504, detail=f"Query generation timeout ({settings.query_timeout}s)")
    except Exception as e:
        logger.error(f"Failed to explain query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables/search")
async def search_tables(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results to return"),
):
    """
    Search for relevant tables using semantic search.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        List of relevant tables
    """
    try:
        results = table_selector.find_relevant_tables(query, top_k=top_k)

        return {
            "query": query,
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"Table search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
