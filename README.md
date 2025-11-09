# Concord BI Server - Product Recommendation API

**Production-ready V3.2 recommendation API for B2B product recommendations with Redis caching**

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Access to MSSQL database (ConcordDb_v5)
- Redis (for caching)

### 2. Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create `.env` file from template:
```bash
cp .env.example .env
```

Update `.env` with your values:
```env
# Database Configuration
DB_HOST=your-database-host
DB_PORT=1433
DB_NAME=ConcordDb_v5
DB_USER=your-username
DB_PASSWORD=your-password

# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600

# Worker Configuration (optional)
AS_OF_DATE=2024-07-01
CRON_SCHEDULE=0 2 * * 0

# API Configuration
API_WORKERS=4
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Run API
```bash
# Local development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production (with workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Test API
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 410376,
    "top_n": 50,
    "as_of_date": "2024-07-01",
    "use_cache": true,
    "include_discovery": true
  }'
```

---

## API Endpoints

### POST /recommend
Get personalized product recommendations with discovery.

**Request:**
```json
{
  "customer_id": 410376,
  "top_n": 50,
  "as_of_date": "2024-07-01",
  "use_cache": true,
  "include_discovery": true
}
```

**Response:**
```json
{
  "customer_id": 410376,
  "recommendations": [
    {
      "product_id": 25432060,
      "score": 81.25,
      "rank": 1,
      "segment": "HEAVY",
      "source": "repurchase"
    },
    {
      "product_id": 12345678,
      "score": 0.42,
      "rank": 21,
      "segment": "HEAVY",
      "source": "discovery"
    }
  ],
  "count": 25,
  "discovery_count": 5,
  "precision_estimate": 0.754,
  "latency_ms": 0.63,
  "cached": true,
  "timestamp": "2024-11-09T10:30:00Z"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "redis_connected": true,
  "model_version": "improved_hybrid_v3_75.4pct_pooled"
}
```

### GET /metrics
Performance metrics endpoint.

**Response:**
```json
{
  "total_requests": 1234,
  "cache_hit_rate": 0.85,
  "error_rate": 0.01,
  "avg_latency_ms": 45.2
}
```

---

## V3.2 Algorithm

Hybrid recommendation system with two components:

### 1. Repurchase Recommendations
Scores products customer has purchased before using:
- **Frequency** (40% weight): How many times purchased
- **Recency** (60% weight): How recently purchased

### 2. Discovery Recommendations
Finds NEW products customer hasn't bought using collaborative filtering:
- Finds similar customers (Jaccard similarity on purchase history)
- Recommends products those similar customers bought
- Filters out products customer already owns

### Strict 20+5 Mix
- **20 repurchase products**: Best candidates for re-ordering
- **5 discovery products**: New products to explore
- All customers get discovery (including heavy users)
- Product diversity: Max 3 products per product group

---

## Docker Deployment

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for complete Docker setup with:
- FastAPI service (port 8000)
- Redis cache (port 6379)
- Weekly background worker (pre-computes recommendations)

Quick start:
```bash
docker-compose up -d
```

---

## Performance

**With Redis Cache**:
- Cached requests: 0.50-0.63ms latency
- Cache hit rate: 85%+

**Without Cache (Real-time)**:
- Average latency: 3-12 seconds
- Depends on customer size and database load

**Cache Performance Improvement**: 19,800x faster (cached vs uncached)

---

## Architecture

```
┌─────────────────┐
│  Client Request │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   FastAPI       │
│   (port 8000)   │
└────────┬────────┘
         │
         ├──────> Redis Cache (check cache)
         │        └─> Hit: Return in <1ms
         │        └─> Miss: Generate below
         │
         v
┌─────────────────┐
│ V3.2 Algorithm  │
│ - Repurchase    │
│ - Discovery     │
│ - Diversity     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  MSSQL Database │
│  ConcordDb_v5   │
└─────────────────┘
```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | MSSQL server host | Required |
| `DB_PORT` | MSSQL server port | 1433 |
| `DB_NAME` | Database name | ConcordDb_v5 |
| `DB_USER` | Database username | Required |
| `DB_PASSWORD` | Database password | Required |
| `REDIS_HOST` | Redis host | 127.0.0.1 |
| `REDIS_PORT` | Redis port | 6379 |
| `REDIS_DB` | Redis database number | 0 |
| `CACHE_TTL` | Cache TTL in seconds | 3600 |
| `AS_OF_DATE` | Point-in-time date (YYYY-MM-DD) | Today |
| `API_WORKERS` | Number of uvicorn workers | 4 |
| `API_HOST` | API host | 0.0.0.0 |
| `API_PORT` | API port | 8000 |

---

## License

Proprietary - Concord Enterprise
