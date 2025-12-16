# Ukrainian RAG System - Complete Implementation

## Overview

Complete RAG (Retrieval-Augmented Generation) system with full Ukrainian language support for querying MSSQL database using natural language.

## Features

- **Ukrainian Language Support**: Ask questions in Ukrainian, get answers in Ukrainian
- **Hybrid SQL + RAG**: Automatically routes queries to SQL or semantic search
- **Multilingual Embeddings**: Uses `intfloat/multilingual-e5-large` for Cyrillic text
- **Local LLM**: Runs qwen2:7b via Ollama (better Ukrainian support than sqlcoder)
- **Vector Database**: ChromaDB for semantic search
- **FastAPI**: Production-ready REST API
- **CLI Tool**: Command-line management interface

## Architecture

```
User Question (Ukrainian/English)
         ↓
   Hybrid Agent (Classification)
         ↓
    ┌────┴────┐
    ↓         ↓
SQL Mode    RAG Mode
    ↓         ↓
sqlcoder   qwen2:7b
    ↓         ↓
T-SQL      Semantic Search
    ↓         ↓
MSSQL      ChromaDB
```

## Installation

### 1. Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Pull Ollama models
ollama pull sqlcoder:7b  # For SQL generation
ollama pull qwen2:7b     # For RAG answers
```

### 2. Verify Installation

```bash
# Check Ollama
ollama list

# Check database connection
python -c "from db_pool import get_connection; conn = get_connection(); print('✓ DB connected'); conn.close()"
```

## Quick Start

### Option 1: Using CLI (Recommended)

```bash
# Run full pipeline in test mode (5 tables, 100 docs)
python rag_cli.py pipeline --test

# Query the system
python rag_cli.py query "Скільки клієнтів з Києва?"

# Semantic search
python rag_cli.py search "клієнти Київ"

# Show statistics
python rag_cli.py stats
```

### Option 2: Using Python API

```python
from hybrid_agent import HybridAgent

# Initialize agent
agent = HybridAgent()

# Query (automatic SQL or RAG routing)
result = agent.query("Скільки клієнтів з Києва?")

print(result['answer'])
print(f"Mode used: {result['mode']}")  # 'sql' or 'rag'
```

### Option 3: Using REST API

```bash
# Start enhanced API
python enhanced_api.py

# In another terminal, test endpoints:

# Hybrid query (auto mode)
curl -X POST http://localhost:8000/hybrid/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Скільки клієнтів з Києва?"}'

# RAG query
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Розкажи про клієнтів у Києві", "n_results": 5}'

# Semantic search
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "клієнти Київ", "n_results": 5}'
```

## Full Production Setup

### Step 1: Extract Data

Extract all data from database and create Ukrainian documents:

```bash
# Full extraction (all 317 tables)
python data_extractor.py

# Test extraction (5 tables only)
python data_extractor.py --test

# Output: data/extracted_documents.json
```

### Step 2: Create Embeddings

Embed all documents into vector database:

```bash
# Full embedding
python embedder.py

# Test embedding (100 docs)
python embedder.py --test

# Reset and re-embed
python embedder.py --reset

# Output: chroma_db/ directory
```

### Step 3: Start API

```bash
# Start enhanced API (port 8000)
python enhanced_api.py

# Or use the original API (backward compatible)
python simple_api.py
```

### Step 4: Test Queries

```bash
# Ukrainian query (SQL mode)
python rag_cli.py query "Скільки клієнтів з Києва?"

# Ukrainian query (RAG mode)
python rag_cli.py query "Розкажи про клієнтів компанії"

# English query
python rag_cli.py query "How many clients in Kyiv?"

# Force specific mode
python rag_cli.py query "Скільки клієнтів?" --mode sql
```

## API Endpoints

### Enhanced API Endpoints

#### POST /hybrid/query
Hybrid SQL + RAG query with automatic routing

```json
{
  "question": "Скільки клієнтів з Києва?",
  "mode": "auto",  // "auto", "sql", or "rag"
  "n_results": 5
}
```

Response:
```json
{
  "question": "Скільки клієнтів з Києва?",
  "language": "uk",
  "mode": "sql",
  "classification": {
    "mode": "sql",
    "confidence": 0.8,
    "explanation": "Selected SQL mode with 80% confidence"
  },
  "sql": "SELECT COUNT(*) FROM [dbo].[Client] WHERE City = N'Київ' AND Deleted = 0",
  "results": [{"count": 42}],
  "success": true,
  "answer": "Результат: 42"
}
```

#### POST /rag/query
Pure RAG semantic search + answer generation

```json
{
  "question": "Розкажи про клієнтів у Києві",
  "n_results": 5,
  "return_context": false
}
```

Response:
```json
{
  "question": "Розкажи про клієнтів у Києві",
  "language": "uk",
  "answer": "У базі даних є інформація про клієнтів з Києва...",
  "success": true,
  "n_results": 5,
  "model": "qwen2:7b"
}
```

#### POST /rag/search
Semantic search only (no answer generation)

```json
{
  "query": "клієнти Київ",
  "n_results": 5
}
```

Response:
```json
{
  "query": "клієнти Київ",
  "n_results": 5,
  "documents": ["=== Запис з таблиці Клієнт ===\n..."],
  "metadatas": [{"table": "Client", "city": "Київ"}],
  "distances": [0.15, 0.23, ...]
}
```

#### GET /stats
System statistics

#### GET /languages/detect?query=...
Detect query language

### Original API Endpoints (Backward Compatible)

#### POST /query
Original Text-to-SQL endpoint

#### GET /schema
Schema information

#### GET /tables/search?query=...
Table search

## CLI Commands

### Full Pipeline
```bash
# Test mode (5 tables, 100 docs)
python rag_cli.py pipeline --test

# Production (all tables)
python rag_cli.py pipeline

# With reset
python rag_cli.py pipeline --reset
```

### Data Extraction
```bash
# Test extraction
python rag_cli.py extract --test

# Full extraction
python rag_cli.py extract

# Custom output
python rag_cli.py extract --output-dir my_data
```

### Embedding
```bash
# Test embedding
python rag_cli.py embed --test

# Full embedding
python rag_cli.py embed

# Reset collection
python rag_cli.py embed --reset

# Custom settings
python rag_cli.py embed \
  --input data/extracted_documents.json \
  --chroma-dir chroma_db \
  --collection concorddb_ukrainian \
  --batch-size 32
```

### Query
```bash
# Auto mode
python rag_cli.py query "Скільки клієнтів з Києва?"

# Force SQL
python rag_cli.py query "Скільки клієнтів?" --mode sql

# Force RAG
python rag_cli.py query "Розкажи про клієнтів" --mode rag
```

### Search
```bash
# Basic search
python rag_cli.py search "клієнти Київ"

# More results
python rag_cli.py search "замовлення" --n-results 10
```

### Statistics
```bash
python rag_cli.py stats
```

## Configuration Files

### prompts/system_prompt_uk.txt
Ukrainian system prompt for RAG queries. Instructs LLM to:
- Respond ONLY in Ukrainian
- Use proper Ukrainian formatting (dates, numbers, currency)
- Always cite sources (table name, record IDs)
- Base answers ONLY on provided context

### prompts/sql_prompt_uk.txt
Ukrainian prompt for SQL generation. Includes:
- T-SQL syntax rules for MSSQL Server
- Ukrainian → SQL translation examples
- Time expressions (сьогодні, вчора, цього місяця)
- City/country name mappings

### dictionaries/uk_column_mapping.json
Ukrainian term mappings:
- `terms`: Ukrainian word → database tables
- `cities`: City name variations (Київ, Kiev, Kyiv)
- `countries`: Country name variations
- `columns`: Ukrainian column names → English
- `query_patterns`: Query intent patterns (скільки → COUNT)

## File Structure

```
db-ai-api/
├── data/                           # Data storage
│   ├── extracted_documents.json    # Extracted DB data
│   └── extraction_stats.json       # Extraction statistics
├── chroma_db/                      # Vector database
├── prompts/                        # LLM prompts
│   ├── system_prompt_uk.txt        # Ukrainian RAG prompt
│   └── sql_prompt_uk.txt           # Ukrainian SQL prompt
├── dictionaries/                   # Language mappings
│   └── uk_column_mapping.json      # Ukrainian dictionary
├── utils/                          # Utilities
│   └── language_utils.py           # Ukrainian language support
├── data_extractor.py               # Data extraction
├── embedder.py                     # Embedding system
├── rag_engine.py                   # RAG query engine
├── hybrid_agent.py                 # Hybrid SQL+RAG agent
├── enhanced_api.py                 # FastAPI with RAG
├── rag_cli.py                      # CLI management tool
├── simple_api.py                   # Original Text-to-SQL API
├── db_pool.py                      # Database connection pool
├── schema_cache.json               # Database schema (317 tables)
└── requirements.txt                # Python dependencies
```

## Ukrainian Language Features

### Automatic Language Detection
```python
from utils.language_utils import detect_language

lang = detect_language("Скільки клієнтів?")  # Returns 'uk'
lang = detect_language("How many clients?")   # Returns 'en'
```

### Ukrainian Document Formatting
```python
from utils.language_utils import create_ukrainian_document

doc = create_ukrainian_document(
    table_name="Client",
    row_data={"ID": 123, "Name": "ТОВ Горизонт", "City": "Київ"},
    primary_key="ID"
)

# Output:
# === Запис з таблиці Клієнт ===
# ID запису: 123
# Основна інформація:
# - Назва: ТОВ Горизонт
# - Місто: Київ
```

### City/Country Normalization
```python
from utils.language_utils import normalize_ukrainian_value

normalize_ukrainian_value("Kiev")   # → "Київ"
normalize_ukrainian_value("Kyiv")   # → "Київ"
normalize_ukrainian_value("Киев")   # → "Київ"
```

### Number/Currency Formatting
```python
from utils.language_utils import format_currency_ukrainian

format_currency_ukrainian(1250000)  # → "1 250 000 UAH"
```

## Query Classification

The hybrid agent automatically classifies queries:

### SQL Mode (Analytical Queries)
Triggers: скільки, кількість, count, топ, сума, середнє

Examples:
- "Скільки клієнтів з Києва?"
- "Топ 10 найдорожчих товарів"
- "Сума замовлень за місяць"
- "Середня ціна продукту"

### RAG Mode (Informational Queries)
Triggers: що таке, як працює, поясни, розкажи про

Examples:
- "Розкажи про клієнтів компанії"
- "Що таке ProductGroup?"
- "Як працює система знижок?"
- "Порівняй постачальників"

## Performance Optimization

### Data Extraction
- **Sampling**: Large tables are sampled (10K-50K rows) to reduce indexing time
- **Filtering**: System tables and empty tables are skipped
- **Batching**: Data is extracted in manageable chunks

### Embedding
- **Batch Processing**: Documents embedded in batches of 32
- **Chunking**: ChromaDB writes in chunks of 1000 to avoid memory issues
- **Progress Tracking**: tqdm progress bars show real-time status

### Query Performance
- **Lazy Loading**: Components initialized only when needed
- **Connection Pooling**: Database connections reused
- **Cache**: Schema cached to avoid repeated DB queries

## Troubleshooting

### Issue: qwen2:7b not found
```bash
# Pull the model
ollama pull qwen2:7b

# Verify
ollama list | grep qwen2
```

### Issue: ChromaDB collection not found
```bash
# Run pipeline first to create collection
python rag_cli.py pipeline --test
```

### Issue: Database connection failed
```bash
# Check db_pool.py configuration
# Verify: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
```

### Issue: Out of memory during embedding
```bash
# Reduce batch size
python embedder.py --batch-size 16
```

### Issue: Slow query responses
- Check Ollama is running: `ollama list`
- Reduce n_results: `python rag_cli.py query "..." --n-results 3`
- Use test mode for development

## Example Queries

### Ukrainian SQL Queries
```
Скільки клієнтів з Києва?
Топ 10 найдорожчих товарів
Сума замовлень за останній місяць
Покажи клієнтів з Львова
Кількість замовлень сьогодні
```

### Ukrainian RAG Queries
```
Розкажи про клієнтів компанії
Що таке ProductGroup?
Поясни систему знижок
Які є постачальники?
Інформація про організації
```

### English Queries
```
How many clients in Kyiv?
Top 10 most expensive products
Show clients from Lviv
Total orders this month
```

## Production Deployment

### 1. Full Data Extraction
```bash
python data_extractor.py
```
Expected time: 30-60 minutes for 317 tables

### 2. Full Embedding
```bash
python embedder.py
```
Expected time: 1-2 hours depending on document count

### 3. Start API
```bash
# Production mode
python enhanced_api.py

# Or use gunicorn for production
gunicorn enhanced_api:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 4. Monitoring
```bash
# Check API health
curl http://localhost:8000/health

# Check statistics
curl http://localhost:8000/stats
```

## Next Steps

1. **Test the system**: Run `python rag_cli.py pipeline --test`
2. **Try queries**: Run `python rag_cli.py query "Скільки клієнтів?"`
3. **Explore API**: Visit http://localhost:8000/docs
4. **Full extraction**: Run `python rag_cli.py pipeline` (without --test)
5. **Production deployment**: Use gunicorn + nginx

## Support

For issues or questions:
1. Check this documentation
2. Review code comments in source files
3. Test with `--test` flag first
4. Check Ollama and database connectivity

---

**Built with**: Python 3.11, FastAPI, Ollama, ChromaDB, sentence-transformers
**Models**: sqlcoder:7b, qwen2:7b, multilingual-e5-large
**Database**: Microsoft SQL Server (ConcordDb_v5, 317 tables)
