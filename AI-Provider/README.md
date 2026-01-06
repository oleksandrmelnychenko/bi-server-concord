# AI-Provider

Clean Text-to-SQL and RAG service built on FastAPI, Chroma, and Ollama (`qwen3-coder:30b`) with a lightweight React UI.

## Structure
- `backend/`: FastAPI app (`api.py`, `main.py`), SQL agent, schema extractor, FAISS/Chroma table selector, and query examples module.
- `frontend/`: React + Vite UI for sending questions and visualizing results.
- `backend/prompts/`: Legacy Ukrainian system/SQL prompts (verify encoding and adjust if needed).

## Backend (FastAPI)
1) `cd backend`
2) `python -m venv .venv && .\.venv\Scripts\activate` (PowerShell)  
   `source .venv/bin/activate` (bash)
3) `pip install -r requirements.txt`
4) `cp .env.example .env` and set DB credentials, `OLLAMA_MODEL=qwen3-coder:30b`, and Ollama host.
5) `python main.py`

Notes:
- Default embedding model: `intfloat/multilingual-e5-large` (with E5 query/passages prefixes).
- Default vector store: FAISS (`FAISS_INDEX_PATH=./faiss_index`), exact search via IndexFlatIP; switch to IVF with `FAISS_INDEX_TYPE=ivf` + `FAISS_NLIST`.
- To rebuild FAISS index: `cd backend && python rebuild_faiss_index.py --force` (ensure DB creds are set and the DB is reachable).
- Query examples: build/update FAISS examples index with `python backend/training_data/build_examples_faiss.py --output backend/faiss_examples` (templates live in `backend/training_data/templates/`).
- Chroma is still available: set `VECTOR_STORE_BACKEND=chroma` and `VECTOR_DB_PATH=...`. If reusing the old populated Chroma index, ensure the embedding model matches (old index was likely built with paraphrase-multilingual-MiniLM-L12-v2); otherwise rebuild to avoid similarity drift.
- Logs write to `backend/logs/ai-provider.log`.

## Frontend (React)
1) `cd frontend`
2) `npm install`
3) `cp .env.example .env` and set `VITE_SQL_API=http://localhost:8000`
4) `npm run dev` (or `npm run build && npm run preview`)

## Current APIs
- `POST /query`: generate SQL (via `qwen3-coder:30b`) and optionally execute.
- `POST /execute`: run a SQL statement directly.
- `GET /schema`, `POST /schema/refresh`: schema visibility and refresh.
- `GET /tables/search`: semantic table search.
- `GET /health`: readiness check (DB + Ollama + query examples if available).
