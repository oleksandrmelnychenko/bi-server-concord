"""FastAPI service for AI Provider Text-to-SQL API.

Simplified API with only 2 endpoints:
- POST /query  - Natural language (Ukrainian) to SQL
- GET  /health - Health check
"""
import asyncio
from asyncio import Semaphore
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
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
    title="AI Provider - Text to SQL",
    description="Natural language to SQL query API for Ukrainian text",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Gzip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for natural language query."""
    question: str = Field(..., description="Natural language question (Ukrainian)")
    execute: bool = Field(default=True, description="Execute the generated SQL")
    max_rows: Optional[int] = Field(default=None, description="Max rows to return")


class QueryResponse(BaseModel):
    """Response model for query results."""
    success: bool
    question: str
    sql: str
    row_count: int = 0
    data: Optional[list] = None
    columns: Optional[list] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


# Global instances (initialized on startup)
agent: Optional[SQLAgent] = None
schema_extractor: Optional[SchemaExtractor] = None
table_selector: Optional[TableSelector] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global agent, schema_extractor, table_selector

    logger.info("Initializing AI Provider API...")

    try:
        schema_extractor = SchemaExtractor()
        table_selector = TableSelector(schema_extractor)
        agent = SQLAgent(
            schema_extractor=schema_extractor, table_selector=table_selector
        )

        logger.info("Indexing database schema...")
        table_selector.index_schema()

        logger.info("AI Provider API initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    components = {}
    all_healthy = True

    # Check database
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
        components["ollama"] = {"status": "ok", "model": settings.ollama_model}
    except Exception as e:
        components["ollama"] = {"status": "error", "message": str(e)}
        all_healthy = False

    # Check query examples
    if agent.example_retriever.is_available():
        stats = agent.example_retriever.get_stats()
        components["query_examples"] = {
            "status": "ok",
            "count": stats["total_examples"]
        }
    else:
        components["query_examples"] = {"status": "unavailable"}

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/reload")
async def reload_index():
    """Reload the FAISS index to pick up new training examples."""
    global agent
    from training_data.retriever import QueryExampleRetriever

    try:
        logger.info("Reloading query example retriever...")
        agent.example_retriever = QueryExampleRetriever(db_path=settings.query_examples_db)

        if agent.example_retriever.is_available():
            stats = agent.example_retriever.get_stats()
            logger.info(f"Reloaded: {stats['total_examples']} examples")
            return {"success": True, "message": f"Reloaded {stats['total_examples']} examples"}
        else:
            return {"success": False, "message": "Retriever not available after reload"}
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        return {"success": False, "message": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """
    Generate SQL from natural language (Ukrainian) and execute it.

    Returns standardized response with success flag for chat integration.
    """
    import time
    start_time = time.time()

    try:
        logger.info(f"Query: {request.question}")

        async with query_semaphore:
            result = await asyncio.wait_for(
                agent.query_async(
                    question=request.question,
                    execute=request.execute,
                    max_rows=request.max_rows,
                ),
                timeout=settings.query_timeout
            )

        execution_time = time.time() - start_time

        # Extract execution results
        execution = result.get("execution", {})

        return QueryResponse(
            success=execution.get("success", False) if execution else False,
            question=request.question,
            sql=result.get("sql", ""),
            row_count=execution.get("row_count", 0) if execution else 0,
            data=execution.get("rows") if execution else None,  # sql_agent returns "rows"
            columns=execution.get("columns") if execution else None,
            error=execution.get("error") if execution else None,
            execution_time=round(execution_time, 2),
        )

    except asyncio.TimeoutError:
        logger.error(f"Timeout after {settings.query_timeout}s: {request.question}")
        return QueryResponse(
            success=False,
            question=request.question,
            sql="",
            error=f"Timeout ({settings.query_timeout}s)",
            execution_time=settings.query_timeout,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(
            success=False,
            question=request.question,
            sql="",
            error=str(e),
            execution_time=time.time() - start_time,
        )



class ExecuteRequest(BaseModel):
    """Request model for direct SQL execution."""
    sql: str = Field(..., description="SQL query to execute")
    max_rows: Optional[int] = Field(default=100, description="Max rows to return")


class ExecuteResponse(BaseModel):
    """Response model for SQL execution results."""
    success: bool
    data: Optional[list] = None
    columns: Optional[list] = None
    row_count: int = 0
    error: Optional[str] = None


@app.post("/execute", response_model=ExecuteResponse)
async def execute_sql(request: ExecuteRequest):
    """
    Execute raw SQL query directly.
    Used for simple lookups (e.g., client details) without LLM processing.
    """
    from sqlalchemy import text

    try:
        logger.info(f"Execute SQL: {request.sql[:100]}...")

        with schema_extractor.engine.connect() as conn:
            result = conn.execute(text(request.sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(request.max_rows or 100)]

        return ExecuteResponse(
            success=True,
            data=rows,
            columns=columns,
            row_count=len(rows),
        )

    except Exception as e:
        logger.error(f"Execute failed: {e}")
        return ExecuteResponse(
            success=False,
            error=str(e),
        )


@app.get("/storages")
async def get_storages():
    """
    Get all active storages/warehouses for the storage list panel.
    Returns list of storages with ID and Name.
    """
    from sqlalchemy import text

    try:
        sql = """
        SELECT ID, Name
        FROM dbo.Storage
        WHERE Deleted = 0
        ORDER BY Name
        """

        with schema_extractor.engine.connect() as conn:
            result = conn.execute(text(sql))
            storages = [
                {
                    "id": row[0],
                    "name": row[1]
                }
                for row in result.fetchall()
            ]

        return {"success": True, "storages": storages, "count": len(storages)}

    except Exception as e:
        logger.error(f"Failed to fetch storages: {e}")
        return {"success": False, "error": str(e), "storages": []}


@app.get("/managers")
async def get_managers():
    """
    Get all active managers (users who have created orders).
    Returns list of managers with ID and Name.
    """
    from sqlalchemy import text

    try:
        sql = """
        SELECT DISTINCT
            u.ID,
            ISNULL(u.LastName + ' ' + u.FirstName, 'Manager ' + CAST(u.ID AS VARCHAR)) as Name
        FROM dbo.[User] u
        WHERE u.Deleted = 0
          AND u.ID IN (
              SELECT DISTINCT UserID
              FROM dbo.[Order]
              WHERE Deleted = 0 AND UserID IS NOT NULL
          )
        ORDER BY Name
        """

        with schema_extractor.engine.connect() as conn:
            result = conn.execute(text(sql))
            managers = [
                {
                    "id": row[0],
                    "name": row[1]
                }
                for row in result.fetchall()
            ]

        return {"success": True, "managers": managers, "count": len(managers)}

    except Exception as e:
        logger.error(f"Failed to fetch managers: {e}")
        return {"success": False, "error": str(e), "managers": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
