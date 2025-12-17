# RAG System Setup Instructions (Fresh Start)

## Prerequisites
- Python 3.9+
- MSSQL Server with ConcordDb_v5 database
- GPU recommended for faster embedding
- Ollama (for natural language SQL queries)

## Install Ollama (Required for NL Queries)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull the SQL generation model
ollama pull sqlcoder:7b
```

## Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd Concord-BI-Server/db-ai-api
```

## Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Configure Database Connection
Copy `.env.example` to `.env` and update with your credentials:
```bash
cp .env.example .env
```

Edit `.env` file:
```env
DB_HOST=your-server-ip
DB_PORT=1433
DB_NAME=ConcordDb_v5
DB_USER=your-username
DB_PASSWORD=your-password

# Ollama for natural language queries
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=sqlcoder:7b
```

## Step 5: Run Full Pipeline
```bash
# Option 1: Full pipeline (extract + embed)
python3 rag_cli.py pipeline

# Option 2: Step by step
python3 rag_cli.py extract     # Extract data from database
python3 rag_cli.py embed       # Create embeddings
```

## Step 6: Test the System
```bash
# Semantic search
python3 rag_cli.py search "клієнти Київ"

# Natural language query
python3 rag_cli.py query "Скільки клієнтів з Києва?"
```

## Expected Results
- **Extraction**: ~2 million documents from ~200 tables
- **Embedding time**: ~2-4 hours with GPU, ~8-12 hours without
- **ChromaDB size**: ~10-15 GB when complete

## Check Progress
```bash
source venv/bin/activate
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db')
collection = client.get_collection('concorddb_ukrainian')
print(f'Embedded: {collection.count():,} documents')
"
```

## Resume After Interruption
If embedding stops, just run again - it will skip already embedded documents:
```bash
python3 rag_cli.py embed
```

## File Structure
```
db-ai-api/
├── config.py              # Database configuration
├── data_extractor.py      # Extracts data from MSSQL
├── embedder.py            # Creates embeddings (with resume support)
├── rag_engine.py          # Query engine
├── hybrid_agent.py        # SQL + RAG hybrid agent
├── rag_cli.py             # CLI tool
├── requirements.txt       # Python dependencies
├── chroma_db/             # Vector database (created automatically)
└── data/                  # Extracted documents (created automatically)
```
