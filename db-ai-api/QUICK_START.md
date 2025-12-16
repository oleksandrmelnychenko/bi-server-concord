# Quick Start Guide - Ukrainian RAG System

## 1. Verify Setup (30 seconds)

```bash
# Check Ollama models
ollama list

# Expected output:
# qwen2:7b       dd314f039b9d    4.4 GB    ...
# sqlcoder:7b    ...             ...       ...

# Check Python dependencies
source venv/bin/activate
python3 -c "import langdetect, sentence_transformers, chromadb; print('✓ All dependencies installed')"

# Test Ukrainian utilities
python3 test_ukrainian.py
```

## 2. Quick Test (2 minutes)

Run the full pipeline in test mode (extracts 5 tables, embeds 100 documents):

```bash
# Run pipeline
python3 rag_cli.py pipeline --test

# Expected output:
# 📦 Extracting 5 tables...
# 🎯 Embedding 100 documents...
# ✅ Pipeline Complete!
```

## 3. Test Queries (1 minute)

### Ukrainian SQL Query
```bash
python3 rag_cli.py query "Скільки клієнтів з Києва?"
```

### Ukrainian RAG Query
```bash
python3 rag_cli.py query "Розкажи про клієнтів компанії" --mode rag
```

### Semantic Search
```bash
python3 rag_cli.py search "клієнти Київ"
```

## 4. Start API (30 seconds)

```bash
# Start enhanced API
python3 enhanced_api.py

# API available at: http://localhost:8000
# Docs: http://localhost:8000/docs
```

## Production Setup (60-90 minutes)

### Full Data Extraction (30-60 minutes)
```bash
python3 rag_cli.py extract
```

### Full Embedding (30-60 minutes)
```bash
python3 rag_cli.py embed --reset
```

### Production API
```bash
pip install gunicorn
gunicorn enhanced_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

**Total setup time**: 5 minutes (test) or 90 minutes (production)
**See RAG_SYSTEM.md for complete documentation**
