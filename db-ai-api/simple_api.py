"""Simple FastAPI for Text-to-SQL using pymssql directly."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pymssql
import json
import requests
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings

# Database config
DB_HOST = '78.152.175.67'
DB_PORT = 1433
DB_USER = 'ef_migrator'
DB_PASSWORD = 'Grimm_jow92'
DB_NAME = 'ConcordDb_v5'

# Ollama config
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'sqlcoder:7b'

app = FastAPI(
    title="DB AI API - Text to SQL",
    description="Natural language to SQL powered by local LLM",
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

# Load schema cache
with open('schema_cache.json', 'r') as f:
    SCHEMA_CACHE = json.load(f)

# Initialize ChromaDB
db_path = Path("./vector_db")
chroma_client = chromadb.PersistentClient(
    path=str(db_path),
    settings=ChromaSettings(anonymized_telemetry=False),
)
collection = chroma_client.get_collection(name="tables_ConcordDb_v5")


class QueryRequest(BaseModel):
    question: str
    execute: bool = True
    max_rows: int = 100
    include_explanation: bool = False


class QueryResponse(BaseModel):
    question: str
    sql: str
    explanation: Optional[str] = None
    execution: Optional[Dict[str, Any]] = None


def get_db_connection():
    """Create database connection."""
    return pymssql.connect(
        server=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        timeout=30
    )


def find_relevant_tables(question: str, top_k: int = 10) -> List[Dict]:
    """Find relevant tables using ChromaDB."""
    results = collection.query(query_texts=[question], n_results=top_k)

    relevant_tables = []
    if results and results['documents'] and results['documents'][0]:
        for metadata in results['metadatas'][0]:
            table_name = metadata['table_name']
            # Get full table info from schema cache
            table_info = SCHEMA_CACHE['tables'].get(table_name) or SCHEMA_CACHE['views'].get(table_name)
            if table_info:
                relevant_tables.append(table_info)

    return relevant_tables


def build_context(tables: List[Dict]) -> str:
    """Build context string for LLM."""
    context_parts = []

    for table in tables:
        # Build column list (showing first 10 columns to keep prompt small)
        col_list = ', '.join([f"{c['name']} ({c['type']})" for c in table['columns'][:10]])

        parts = [
            f"Table: {table['full_name']}",
            f"Columns: {col_list}"
        ]

        if table.get('primary_keys'):
            parts.append(f"Primary Keys: {', '.join(table['primary_keys'])}")

        context_parts.append('\n'.join(parts))

    return '\n\n'.join(context_parts)


def generate_sql(question: str, context: str) -> str:
    """Generate SQL using Ollama HTTP API directly."""

    # Build a simple, clear prompt that works well with sqlcoder
    prompt = f"""### Task
Generate a valid T-SQL query for Microsoft SQL Server.

### Question
{question}

### Database Schema
{context}

### Requirements
1. Use SELECT TOP N syntax (NOT LIMIT)
2. Use table aliases like p, c, o
3. Do NOT use NULLS FIRST/LAST (not supported in T-SQL)
4. Use ORDER BY without NULL handling keywords
5. Return ONLY the SQL query

### Answer (SQL only)
SELECT"""

    print(f"\n[DEBUG] Calling Ollama HTTP API")
    print(f"[DEBUG] Model: {OLLAMA_MODEL}")
    print(f"[DEBUG] Prompt length: {len(prompt)} chars")

    try:
        # Call Ollama HTTP API directly
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            },
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        print(f"[DEBUG] HTTP Response status: {response.status_code}")
        print(f"[DEBUG] Response keys: {result.keys()}")

        sql = result.get('response', '').strip()
        print(f"[DEBUG] Raw SQL from LLM (length={len(sql)}): '{sql[:200] if len(sql) > 200 else sql}'")

        # Clean up response - remove special tokens and markdown
        sql = sql.replace('<s>', '').replace('</s>', '').strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()

        # Prepend SELECT if not present (we started the completion with SELECT)
        if sql and not sql.upper().startswith('SELECT'):
            sql = 'SELECT ' + sql

        # Remove any explanatory text before or after the query
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('/*'):
                sql_lines.append(line)

        sql = ' '.join(sql_lines)
        sql = sql.split(';')[0].strip()  # Take first statement

        # Fix common SQL syntax issues for T-SQL
        sql = sql.replace(' LIMIT ', ' TOP ').replace('LIMIT ', 'TOP ')
        sql = sql.replace(' NULLS LAST', '').replace(' NULLS FIRST', '')

        # Fix TOP placement if it's at the end (should be after SELECT)
        import re
        # Pattern: SELECT ... TOP N (wrong placement)
        top_at_end = re.search(r'(SELECT\s+.*?)\s+TOP\s+(\d+)\s*$', sql, re.IGNORECASE)
        if top_at_end:
            # Move TOP to after SELECT
            sql = re.sub(r'(SELECT)\s+(.*?)\s+TOP\s+(\d+)\s*$', r'\1 TOP \3 \2', sql, flags=re.IGNORECASE)

        print(f"[DEBUG] Cleaned SQL: '{sql}'")
        return sql

    except Exception as e:
        print(f"[ERROR] Failed to call Ollama: {e}")
        import traceback
        traceback.print_exc()
        return ''


def execute_sql(sql: str, max_rows: int = 100) -> Dict[str, Any]:
    """Execute SQL query."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(as_dict=True)

        cursor.execute(sql)

        rows = []
        for i, row in enumerate(cursor.fetchall()):
            if i >= max_rows:
                break

            # Convert row to JSON serializable
            clean_row = {}
            for key, value in row.items():
                if value is None:
                    clean_row[key] = None
                elif isinstance(value, (int, float, str, bool)):
                    clean_row[key] = value
                else:
                    clean_row[key] = str(value)
            rows.append(clean_row)

        conn.close()

        return {
            "success": True,
            "rows": rows,
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else []
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.get("/")
async def root():
    return {
        "message": "DB AI API - Text to SQL",
        "version": "1.0.0",
        "status": "running",
        "database": DB_NAME,
        "model": OLLAMA_MODEL,
        "tables": len(SCHEMA_CACHE['tables']),
        "views": len(SCHEMA_CACHE['views'])
    }


@app.get("/health")
async def health():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected", "ollama": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Main endpoint: natural language to SQL."""
    try:
        # Find relevant tables (reduced from 10 to 5 to keep prompt smaller)
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
        "tables": len(SCHEMA_CACHE['tables']),
        "views": len(SCHEMA_CACHE['views']),
        "total": len(SCHEMA_CACHE['tables']) + len(SCHEMA_CACHE['views'])
    }


@app.get("/tables/search")
async def search_tables(query: str, top_k: int = 5):
    """Semantic table search."""
    results = collection.query(query_texts=[query], n_results=top_k)

    tables = []
    if results and results['metadatas'] and results['metadatas'][0]:
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            tables.append({
                "rank": i + 1,
                "table_name": metadata['table_name'],
                "type": metadata['type'],
                "row_count": metadata['row_count'],
                "relevance_score": 1 - distance
            })

    return {"query": query, "results": tables}


if __name__ == "__main__":
    import uvicorn
    print("Starting DB AI API...")
    print(f"Database: {DB_NAME}")
    print(f"Tables: {len(SCHEMA_CACHE['tables'])}")
    print(f"Model: {OLLAMA_MODEL}")
    print("\nAPI will be available at: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
