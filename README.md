# Concord BI Server - Production Recommendation API

**Production-ready V3 recommendation API for B2B product recommendations**

---

## Performance Metrics (Empirically Validated)

- **Precision@50:** 34.1% (1 in 3 recommendations actually purchased)
- **Average Latency:** 566ms
- **Success Rate:** 100% (all test customers)
- **Segment Performance:**
  - Heavy users (500+ orders): 46% precision
  - Regular users (100-499 orders): 32% precision
  - Light users (<100 orders): 12.5% precision

---

## Quick Start

### 1. Prerequisites
- Python 3.9+
- Access to MSSQL database (ConcordDb_v5)
- Redis (optional, for caching)

### 2. Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create `.env` file:
```env
MSSQL_HOST=your-database-host
MSSQL_PORT=1433
MSSQL_DATABASE=ConcordDb_v5
MSSQL_USER=your-username
MSSQL_PASSWORD=your-password

# Optional: Redis caching
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
CACHE_TTL=3600
```

### 4. Run API
```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production (with workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Test API
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/predict" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 410169, "n_recommendations": 10}'
```

---

## API Endpoints

### POST /api/v1/recommendations/predict
Get personalized product recommendations.

**Request:**
```json
{
  "customer_id": 410169,
  "n_recommendations": 50,
  "as_of_date": "2024-11-09"
}
```

**Response:**
```json
{
  "customer_id": 410169,
  "recommendations": [
    {"product_id": 25432060, "score": 81.25, "rank": 1, "segment": "HEAVY"}
  ],
  "model_version": "V3",
  "generated_at": "2024-11-09T10:30:00Z"
}
```

---

## V3 Algorithm

Segment-specific recommendation strategy:

```
├── HEAVY (≥500 orders) → Frequency 60%, Recency 25%
├── REGULAR_CONSISTENT → Frequency 50%, Recency 35%
├── REGULAR_EXPLORATORY → Frequency 25%, Recency 50%
└── LIGHT (<100 orders) → Frequency 70%, Recency 30%
```

**Read more:** [HOW_V3_WORKS.md](HOW_V3_WORKS.md)

---

## Documentation

- **[HOW_V3_WORKS.md](HOW_V3_WORKS.md)** - V3 algorithm explained
- **[PRODUCTION_API_TEST_REPORT.md](PRODUCTION_API_TEST_REPORT.md)** - Test results
- **[RECOMMENDATION_API_COMPARISON.md](RECOMMENDATION_API_COMPARISON.md)** - V3 vs V3.5 vs V4
- **[V4_EMPIRICAL_VALIDATION_REPORT.md](V4_EMPIRICAL_VALIDATION_REPORT.md)** - Why V4 failed
- **[V3.6_STATUS_REPORT.md](V3.6_STATUS_REPORT.md)** - V3.6 status

---

## FAQ

**Why V3 and not V4?**
V4 was 40x slower (22.6s vs 566ms) and had worse precision (30.5% vs 34.1%).

**What about V3.6?**
V3.6 was planned but never implemented. Design docs exist only.

---

## License

Proprietary - Concord Enterprise
