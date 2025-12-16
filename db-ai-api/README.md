# DB AI API - Text-to-SQL with Local LLM

Transform natural language questions into SQL queries using a local LLM (Ollama). Designed for large MSSQL databases (100+ tables) with RAG-based table selection.

## Features

- **Natural Language to SQL**: Convert plain English questions to SQL queries
- **Local LLM**: Uses Ollama (SQLCoder or CodeLlama) - no API costs
- **RAG-based Table Selection**: Semantic search to find relevant tables from 100+ tables
- **MSSQL Support**: Full Microsoft SQL Server integration
- **Safety First**: Read-only mode, query validation, timeout protection
- **Both Execution Modes**: Auto-execute or return SQL for review
- **FastAPI**: Modern async API with automatic documentation

## Architecture

```
User Question → RAG Table Selection → LLM SQL Generation → Validation → Execution → Results
                    ↓                        ↓
                ChromaDB              Ollama (SQLCoder)
                    ↓
                MSSQL Schema Cache
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Ollama installed and running
- Access to MSSQL database
- ODBC Driver 18 for SQL Server

### 2. Installation

```bash
# Clone/navigate to directory
cd db-ai-api

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your database credentials
```

### 3. Configure Environment

Edit `.env`:

```env
DB_HOST=your-mssql-server.com
DB_PORT=1433
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

OLLAMA_MODEL=sqlcoder:7b  # or codellama:13b
READ_ONLY_MODE=True
```

### 4. Run the API

```bash
# Option 1: Direct Python
python main.py

# Option 2: Docker Compose
docker-compose up -d
```

API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Query Database (Main Endpoint)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me top 10 customers by revenue in 2024",
    "execute": true,
    "include_explanation": true
  }'
```

Response:
```json
{
  "question": "Show me top 10 customers by revenue in 2024",
  "sql": "SELECT TOP 10 c.customer_name, SUM(o.total_amount) as revenue...",
  "explanation": "This query joins customers with orders...",
  "generation_attempts": 1,
  "execution": {
    "success": true,
    "columns": ["customer_name", "revenue"],
    "rows": [...],
    "row_count": 10,
    "execution_time_seconds": 0.42
  }
}
```

### 2. Get Database Schema

```bash
curl http://localhost:8000/schema
```

### 3. Search Tables (Semantic)

```bash
curl "http://localhost:8000/tables/search?query=customer+orders&top_k=5"
```

### 4. Explain Query (No Execution)

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"question": "Find products with low inventory"}'
```

### 5. Execute SQL Directly

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT TOP 10 * FROM Products WHERE stock < 10"
  }'
```

### 6. Refresh Schema Cache

```bash
curl -X POST http://localhost:8000/schema/refresh
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI with interactive API testing.

## Configuration Options

### Security Settings

- `READ_ONLY_MODE=True` - Block all write operations (INSERT, UPDATE, DELETE, DROP)
- `QUERY_TIMEOUT=30` - Maximum query execution time (seconds)
- `MAX_ROWS_RETURNED=1000` - Limit result rows

### RAG Settings

- `TOP_K_TABLES=10` - Number of tables to include in LLM context
- `VECTOR_DB_PATH=./vector_db` - ChromaDB storage location
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` - Embedding model

### LLM Settings

- `OLLAMA_BASE_URL=http://localhost:11434` - Ollama server URL
- `OLLAMA_MODEL=sqlcoder:7b` - Model to use (sqlcoder:7b, codellama:13b)

## Project Structure

```
db-ai-api/
├── api.py                  # FastAPI application
├── config.py               # Configuration management
├── schema_extractor.py     # MSSQL schema introspection
├── table_selector.py       # RAG-based table selection
├── sql_agent.py           # LLM SQL generation & execution
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image
├── docker-compose.yml    # Docker deployment
├── .env.example          # Environment template
├── vector_db/            # ChromaDB storage (auto-created)
├── logs/                 # Application logs (auto-created)
└── schema_cache.json     # Cached schema (auto-created)
```

## How It Works

### 1. Schema Extraction
- Connects to MSSQL and extracts all tables, columns, types, relationships
- Samples 5 rows per table for context
- Caches schema to `schema_cache.json` for performance

### 2. RAG Indexing
- Converts each table into a rich text description
- Embeds using sentence-transformers
- Stores in ChromaDB for semantic search

### 3. Query Processing
- User submits natural language question
- RAG retrieves top-K most relevant tables
- LLM receives question + relevant table schemas
- Generates SQL query (with retry on failure)

### 4. Execution (Optional)
- Validates SQL (syntax, read-only check)
- Executes with timeout protection
- Returns results with metadata

## Ollama Models

### Recommended Models

**SQLCoder 7B** (Best for SQL)
```bash
ollama pull sqlcoder:7b
```

**CodeLlama 13B** (More powerful, slower)
```bash
ollama pull codellama:13b
```

**Mistral 7B** (Fast, general purpose)
```bash
ollama pull mistral:7b
```

### Model Performance on M1 Pro (16GB)

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| SQLCoder 7B | ~2s | Excellent | ~5GB |
| CodeLlama 13B | ~4s | Better | ~8GB |
| Mistral 7B | ~1.5s | Good | ~4GB |

## Example Queries

```python
# Python client example
import requests

API_URL = "http://localhost:8000"

def ask_database(question):
    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": question,
            "execute": True,
            "include_explanation": True
        }
    )
    return response.json()

# Example usage
result = ask_database("What are the top 5 products by sales?")
print(f"SQL: {result['sql']}")
print(f"Results: {result['execution']['rows']}")
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check Ollama is running
ollama list

# Start Ollama service (Mac)
brew services start ollama

# Test connection
curl http://localhost:11434/api/tags
```

### Database Connection Issues

```bash
# Test ODBC driver
odbcinst -q -d

# Test connection
python -c "import pyodbc; pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=...')"
```

### Schema Refresh

If database schema changes:

```bash
curl -X POST http://localhost:8000/schema/refresh
```

Or delete cache:

```bash
rm schema_cache.json
rm -rf vector_db/
```

## Performance Tips

1. **Large databases (500+ tables)**: Reduce `TOP_K_TABLES` to 5-7
2. **Slow queries**: Use smaller model (mistral:7b)
3. **Better accuracy**: Use larger model (codellama:13b)
4. **Cache schema**: Schema extraction runs once and caches

## Security

- **Read-only mode**: Enabled by default, blocks write operations
- **SQL injection protection**: Basic pattern detection
- **Query timeout**: Prevents long-running queries
- **Row limits**: Prevents memory issues from large results

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.
