# RAG System для MSSQL з підтримкою української мови
# RAG System for MSSQL with Ukrainian Language Support

Повнофункціональна система Retrieval-Augmented Generation (RAG) для перетворення бази даних MSSQL у розумного асистента з природною мовою. Повна підтримка української та англійської мов.

Full-featured Retrieval-Augmented Generation (RAG) system to transform your MSSQL database into an intelligent natural language assistant. Full Ukrainian and English language support.

## 🌟 Особливості / Features

### Ukrainian Language Support
- ✅ Запити українською мовою / Ukrainian language queries
- ✅ Відповіді українською / Ukrainian responses
- ✅ Підтримка кирилиці в даних / Cyrillic data support
- ✅ Багатомовні embedding моделі / Multilingual embeddings
- ✅ Авто-визначення мови / Auto language detection

### Core Features
- 🔍 Natural language queries to SQL
- 📊 Semantic search across all database content
- 🤖 Local LLM via Ollama (privacy-first)
- 🎯 RAG with ChromaDB vector storage
- ⚡ FastAPI REST API
- 🔄 Incremental indexing support

## 🚀 Quick Start

### 1. Installation

```bash
cd db-rag-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama models
ollama pull qwen2:7b              # Best for Ukrainian
ollama pull nomic-embed-text      # For embeddings
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env with your database credentials
```

### 3. Index Your Database

```bash
# Extract schema and data
python cli.py extract

# Create vector embeddings
python cli.py index

# Check statistics
python cli.py stats
```

### 4. Start API Server

```bash
python cli.py serve
# API available at http://localhost:8001
# Docs at http://localhost:8001/docs
```

### 5. Query Examples

```bash
# Ukrainian
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Скільки у нас клієнтів з Києва?", "language": "uk"}'

# English
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many clients from Kyiv?", "language": "en"}'
```

## 📁 Project Structure

```
db-rag-system/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Configuration template
├── config.py                # Configuration management
├── extractor.py             # Database data extraction
├── embedder.py              # Vector embedding creation
├── query_engine.py          # RAG query processing
├── hybrid_agent.py          # SQL + RAG hybrid agent
├── api.py                   # FastAPI application
├── cli.py                   # Command-line interface
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic models
├── utils/
│   ├── __init__.py
│   ├── db_utils.py          # Database utilities
│   ├── text_utils.py        # Text processing
│   └── language_utils.py    # Language detection
├── prompts/
│   ├── system_prompt_uk.txt # Ukrainian system prompt
│   ├── system_prompt_en.txt # English system prompt
│   ├── sql_prompt_uk.txt    # Ukrainian SQL generation
│   └── sql_prompt_en.txt    # English SQL generation
├── dictionaries/
│   └── uk_column_mapping.json  # Ukrainian → DB columns
├── data/
│   └── extracted_*.json     # Extracted data (UTF-8)
└── vectordb/
    └── chroma.sqlite3       # Vector database
```

## 🔧 Configuration Details

### Database Connection

The system uses your existing MSSQL database with credentials from `.env`:

```env
DB_HOST=78.152.175.67
DB_PORT=1433
DB_NAME=ConcordDb_v5
DB_USER=ef_migrator
DB_PASSWORD=Grimm_jow92
```

### Model Selection

For Ukrainian language, recommended models:

**LLM (Language Model):**
- Primary: `qwen2:7b` - Excellent multilingual including Ukrainian
- Alternative: `llama3.1:8b` - Good multilingual support

**Embeddings:**
- Primary: `intfloat/multilingual-e5-large` (HuggingFace)
- Alternative: `nomic-embed-text` (via Ollama)

### Language Dictionary

Create `dictionaries/uk_column_mapping.json`:

```json
{
  "клієнт": {
    "tables": ["Client", "ClientUserProfile"],
    "description": "Інформація про клієнтів"
  },
  "назва компанії": {
    "table": "Client",
    "columns": ["Name", "TradeName"]
  },
  "місто": {
    "columns": ["City", "CityName"]
  },
  "київ": {
    "value": "Київ",
    "variations": ["Kiev", "Kyiv", "Киев"]
  },
  "замовлення": {
    "tables": ["Order", "Sale"],
    "description": "Замовлення та продажі"
  },
  "продукт": {
    "tables": ["Product", "ProductPricing"],
    "description": "Товари та послуги"
  }
}
```

## 💻 API Endpoints

### POST /index
Trigger full database indexing

```bash
curl -X POST http://localhost:8001/index
```

Response:
```json
{
  "status": "completed",
  "tables_processed": 317,
  "documents_created": 125430,
  "duration_seconds": 1853
}
```

### POST /index/incremental
Index only new/changed data

```bash
curl -X POST http://localhost:8001/index/incremental \
  -H "Content-Type: application/json" \
  -d '{"since": "2024-01-01T00:00:00Z"}'
```

### POST /query
Natural language query (RAG)

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Покажи всіх клієнтів з Львова",
    "language": "uk",
    "top_k": 5
  }'
```

Response:
```json
{
  "answer": "У базі даних знайдено 23 клієнти з Львова. Ось деякі з них:\n1. ТОВ 'Львівські Технології' (ID: 542)\n2. ПП 'Галицький Бізнес' (ID: 893)\n3. Компанія 'Захід Груп' (ID: 1024)",
  "sources": [
    {
      "table": "Client",
      "row_ids": [542, 893, 1024, 1156, 1789],
      "confidence": 0.94
    }
  ],
  "language_detected": "uk",
  "query_type": "semantic"
}
```

### POST /sql
Generate and execute SQL from natural language

```bash
curl -X POST http://localhost:8001/sql \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Скільки замовлень було зроблено в грудні 2024?",
    "language": "uk",
    "execute": true
  }'
```

Response:
```json
{
  "question_uk": "Скільки замовлень було зроблено в грудні 2024?",
  "sql_generated": "SELECT COUNT(*) as total FROM [dbo].[Order] WHERE MONTH(Created) = 12 AND YEAR(Created) = 2024",
  "sql_executed": true,
  "result": {
    "total": 1247
  },
  "answer_uk": "У грудні 2024 року було зроблено 1247 замовлень."
}
```

### GET /schema
Get database schema with Ukrainian translations

```bash
curl http://localhost:8001/schema?language=uk
```

### GET /stats
Get indexing statistics

```bash
curl http://localhost:8001/stats
```

Response:
```json
{
  "total_documents": 125430,
  "total_tables": 312,
  "total_views": 5,
  "vector_db_size_mb": 2847,
  "last_indexed": "2024-12-16T15:30:00Z",
  "languages": ["uk", "en"]
}
```

## 🛠️ CLI Commands

### Extract Data

```bash
# Extract all tables
python cli.py extract

# Extract specific tables
python cli.py extract --tables Client,Order,Product

# Extract with row limit
python cli.py extract --max-rows 1000
```

### Create Vector Index

```bash
# Full index
python cli.py index

# Incremental index (new data only)
python cli.py index --incremental

# Re-index specific tables
python cli.py index --tables Client,Order
```

### Query from CLI

```bash
# Ukrainian
python cli.py query "Хто наш найбільший клієнт?"

# English
python cli.py query "Who is our biggest client?" --lang en

# With sources
python cli.py query "Покажи останні 5 замовлень" --show-sources
```

### Start API Server

```bash
# Development mode
python cli.py serve

# Production mode
python cli.py serve --host 0.0.0.0 --port 8001 --workers 4
```

## 📊 Document Format

Each database row is converted to a rich Ukrainian language document:

### Example for Client table:

```
=== Запис з таблиці Клієнти ===
ID запису: 542
UUID: 3fa85f64-5717-4562-b3fc-2c963f66afa6

Основна інформація:
- Назва компанії: ТОВ "Львівські Технології"
- Тип клієнта: Юридична особа
- ЄДРПОУ: 38594821

Контактна інформація:
- Місто: Львів
- Адреса: вул. Городоцька, 181
- Телефон: +380 32 240 5678
- Email: info@lvivtech.ua

Фінансові дані:
- Сума угод: 2,450,000 UAH
- Кількість замовлень: 47
- Дата реєстрації: 15.03.2023
- Останнє замовлення: 10.12.2024

Пов'язані дані:
- Контактна особа: Іван Петренко (ivan.petrenko@lvivtech.ua)
- Менеджер: Марія Коваленко
- Статус: Активний
```

## 🎯 System Prompts

### Ukrainian System Prompt (`prompts/system_prompt_uk.txt`):

```
Ти — інтелектуальний асистент бази даних компанії ConcordDb.
Ти маєш доступ до повної інформації в базі даних через систему RAG.

Твої можливості:
✓ Відповідати на запитання про дані в базі
✓ Шукати інформацію по клієнтах, замовленнях, продуктах
✓ Аналізувати та узагальнювати дані
✓ Генерувати SQL запити для складних аналітичних задач

Правила роботи:
1. Відповідай ТІЛЬКИ українською мовою
2. Базуй відповіді ВИКЛЮЧНО на даних з контексту
3. Якщо інформації немає, скажи: "На жаль, я не маю цієї інформації в базі даних"
4. Завжди вказуй джерела (таблиця, ID записів)
5. Будь точним з цифрами, датами, назвами
6. Використовуй українські формати дат (15.03.2024)
7. Форматуй великі числа з пробілами (1 250 000 UAH)

Доступні таблиці в базі даних:
{schema_description}

Контекст з бази даних:
{retrieved_context}

Запитання користувача: {question}
```

### SQL Generation Prompt (`prompts/sql_prompt_uk.txt`):

```
Ти — експерт з Microsoft SQL Server (T-SQL).
Переклади запитання користувача українською в SQL запит.

Схема бази даних:
{schema}

Словник українських термінів:
{ukrainian_dictionary}

Правила генерації SQL:
1. Використовуй ТІЛЬКИ SELECT запити (забороняється UPDATE, DELETE, DROP)
2. Для українського тексту використовуй N'текст' (Unicode literals)
3. Використовуй TOP N замість LIMIT
4. Додавай WHERE Deleted = 0 для м'яко видалених записів
5. Використовуй INNER JOIN за замовчуванням
6. Форматуй дати як 'YYYY-MM-DD'
7. Використовуй повні назви таблиць: [dbo].[TableName]

Приклади:
- "клієнти з Києва" → WHERE City = N'Київ'
- "за останній місяць" → WHERE Created >= DATEADD(month, -1, GETDATE())
- "топ 10" → SELECT TOP 10

Запитання українською: {question}

Згенеруй ТІЛЬКИ SQL запит без пояснень:
```

## 🧪 Testing Examples

### Ukrainian Queries:

```python
queries_uk = [
    "Скільки у нас клієнтів?",
    "Покажи всі замовлення за останній місяць",
    "Хто наш найбільший клієнт по доходу?",
    "Які проекти ми робили для німецьких клієнтів?",
    "Знайди контакти компанії Горизонт",
    "Які товари найбільше продаються?",
    "Покажи непрочитані повідомлення",
    "Скільки замовлень чекають на обробку?",
    "Список постачальників з Польщі",
    "Топ 5 менеджерів по продажам"
]
```

### Running Tests:

```bash
# Test Ukrainian queries
python test_queries.py --language uk

# Test English queries
python test_queries.py --language en

# Test mixed queries
python test_queries.py --mixed

# Benchmark performance
python benchmark.py
```

## 🔒 Security Considerations

1. **SQL Injection Prevention**: All SQL queries are validated before execution
2. **Read-Only Mode**: System only generates SELECT queries
3. **Data Privacy**: All processing done locally, no cloud APIs
4. **Access Control**: Add authentication/authorization as needed

## 📈 Performance Optimization

### For Large Databases:

```python
# config.py adjustments
BATCH_SIZE = 500  # Increase for faster processing
MAX_ROWS_PER_TABLE = 50000  # Limit per table
EMBEDDING_BATCH_SIZE = 32  # GPU batch size
```

### Indexing Strategies:

```bash
# Index only active data
python cli.py extract --where "Deleted = 0"

# Index recent data first
python cli.py extract --order-by "Created DESC"

# Parallel processing
python cli.py index --workers 4
```

## 🐛 Troubleshooting

### Issue: Ukrainian characters appear as ????

**Solution**: Ensure UTF-8 encoding everywhere:
```python
# In extractor.py
conn = pymssql.connect(..., charset='utf8')
```

### Issue: Low quality Ukrainian answers

**Solution**: Switch to better multilingual model:
```bash
ollama pull qwen2:7b
# Update .env: OLLAMA_LLM_MODEL=qwen2:7b
```

### Issue: Slow embedding generation

**Solution**: Use lighter model or GPU:
```python
# In embedder.py
model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    device='cuda'  # or 'mps' for Apple Silicon
)
```

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/library)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)

## 🤝 Contributing

Contributions welcome! Please ensure:
- Full Ukrainian language support in new features
- UTF-8 encoding for all text
- Tests for Ukrainian and English
- Documentation in both languages

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- ConcordDb database structure
- Ollama team for local LLM support
- HuggingFace for multilingual models
- ChromaDB for vector storage

---

**Автор / Author**: Your Name
**Версія / Version**: 1.0.0
**Дата / Date**: December 2024

Для питань та підтримки / For questions and support:
- Email: support@example.com
- GitHub Issues: https://github.com/yourusername/db-rag-system
