# B2B Product Recommendation API - Usage Examples

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required (add before production deployment)

---

## 1. Health Check

Check API health and metrics.

**Endpoint:** `GET /health`

**cURL:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "3.0.0",
    "uptime_seconds": 123.45,
    "metrics": {
        "requests": 1,
        "cache_hit_rate": 0.0,
        "error_rate": 0.0,
        "avg_latency_ms": 819.25
    },
    "redis_connected": false,
    "model_version": "improved_hybrid_v3_75.4pct_pooled"
}
```

---

## 2. Get Product Recommendations

Generate personalized product recommendations for a customer.

**Endpoint:** `POST /recommend`

**Request Body:**
```json
{
    "customer_id": 410280,
    "top_n": 50,
    "as_of_date": "2024-07-01",
    "use_cache": false
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 410280,
    "top_n": 50,
    "as_of_date": "2024-07-01",
    "use_cache": false
  }'
```

**Alternative cURL (echo pipe):**
```bash
echo '{
  "customer_id": 410280,
  "top_n": 50,
  "as_of_date": "2024-07-01",
  "use_cache": false
}' | curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d @-
```

**Response:**
```json
{
    "customer_id": 410280,
    "recommendations": [
        {
            "product_id": 25346606,
            "score": 0.7453,
            "rank": 1,
            "segment": "REGULAR_CONSISTENT"
        },
        {
            "product_id": 25269402,
            "score": 0.7173,
            "rank": 2,
            "segment": "REGULAR_CONSISTENT"
        },
        {
            "product_id": 25384969,
            "score": 0.5489,
            "rank": 3,
            "segment": "REGULAR_CONSISTENT"
        }
    ],
    "count": 50,
    "precision_estimate": 0.754,
    "latency_ms": 145.23,
    "cached": false,
    "timestamp": "2025-11-04T21:52:26.123456"
}
```

---

## 3. Python Client Example

```python
import requests
import json

API_BASE = "http://localhost:8000"

def get_recommendations(customer_id, top_n=50, as_of_date=None, use_cache=True):
    """Get product recommendations for a customer"""

    payload = {
        "customer_id": customer_id,
        "top_n": top_n,
        "use_cache": use_cache
    }

    if as_of_date:
        payload["as_of_date"] = as_of_date

    response = requests.post(
        f"{API_BASE}/recommend",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    response.raise_for_status()
    return response.json()


def check_health():
    """Check API health"""
    response = requests.get(f"{API_BASE}/health")
    response.raise_for_status()
    return response.json()


# Example usage
if __name__ == "__main__":
    # Check API health
    health = check_health()
    print(f"API Status: {health['status']}")
    print(f"Model Version: {health['model_version']}")
    print(f"Avg Latency: {health['metrics']['avg_latency_ms']:.2f}ms")

    # Get recommendations
    customer_id = 410280
    result = get_recommendations(customer_id, top_n=10)

    print(f"\nTop 10 Recommendations for Customer {customer_id}:")
    print(f"Expected Precision: {result['precision_estimate']:.1%}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    print(f"Cached: {result['cached']}")

    for rec in result['recommendations']:
        print(f"  #{rec['rank']:2d}. Product {rec['product_id']} (score: {rec['score']:.4f})")
```

---

## 4. JavaScript/TypeScript Client Example

```javascript
const API_BASE = "http://localhost:8000";

async function getRecommendations(customerId, topN = 50, asOfDate = null, useCache = true) {
    const payload = {
        customer_id: customerId,
        top_n: topN,
        use_cache: useCache,
    };

    if (asOfDate) {
        payload.as_of_date = asOfDate;
    }

    const response = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function checkHealth() {
    const response = await fetch(`${API_BASE}/health`);

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

// Example usage
(async () => {
    try {
        // Check health
        const health = await checkHealth();
        console.log(`API Status: ${health.status}`);
        console.log(`Model Version: ${health.model_version}`);

        // Get recommendations
        const customerId = 410280;
        const result = await getRecommendations(customerId, 10);

        console.log(`\nTop 10 Recommendations for Customer ${customerId}:`);
        console.log(`Expected Precision: ${(result.precision_estimate * 100).toFixed(1)}%`);
        console.log(`Latency: ${result.latency_ms.toFixed(2)}ms`);

        result.recommendations.forEach(rec => {
            console.log(`  #${rec.rank}. Product ${rec.product_id} (score: ${rec.score.toFixed(4)})`);
        });
    } catch (error) {
        console.error('Error:', error);
    }
})();
```

---

## 5. Request Parameters

### `/recommend` Endpoint

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| customer_id | integer | Yes | - | Customer ID to generate recommendations for |
| top_n | integer | No | 50 | Number of recommendations to return (1-500) |
| as_of_date | string | No | today | ISO date (YYYY-MM-DD) for point-in-time recommendations |
| use_cache | boolean | No | true | Use Redis cache if available |

---

## 6. Response Schema

### RecommendationResponse

```json
{
    "customer_id": integer,
    "recommendations": [
        {
            "product_id": integer,
            "score": float,
            "rank": integer,
            "segment": string
        }
    ],
    "count": integer,
    "precision_estimate": float,
    "latency_ms": float,
    "cached": boolean,
    "timestamp": string
}
```

**Fields:**
- `customer_id`: Customer ID
- `recommendations`: List of recommended products
  - `product_id`: Product ID
  - `score`: Recommendation score (0.0-1.0)
  - `rank`: Rank in recommendation list (1-N)
  - `segment`: Customer segment (HEAVY, REGULAR_CONSISTENT, REGULAR_EXPLORATORY, LIGHT)
- `count`: Number of recommendations returned
- `precision_estimate`: Expected precision@50 (0.754 = 75.4%)
- `latency_ms`: Request processing time in milliseconds
- `cached`: Whether result was served from cache
- `timestamp`: ISO timestamp of response

---

## 7. Expected Performance

### Model Performance by Segment

| Segment | Precision@50 | Typical Users |
|---------|--------------|---------------|
| HEAVY (500+ orders) | 57.7% | Large fleet operators |
| REGULAR-CONSISTENT (100-500, 40%+ repurchase) | 40.0% | Regular maintenance customers |
| REGULAR-EXPLORATORY (100-500, <40% repurchase) | 40.0% | Varied purchasing patterns |
| LIGHT (<100 orders) | 16.1% | New or occasional customers |
| **Overall Average** | **29.2%** | All customers |

### API Latency

- **Average:** ~800ms per request
- **P50:** ~500ms
- **P95:** ~1500ms
- **P99:** ~2000ms

**Optimization:** Enable Redis caching to reduce latency by ~70%

---

## 8. Error Responses

### 400 Bad Request
```json
{
    "detail": "Invalid customer_id: must be positive integer"
}
```

### 404 Not Found
```json
{
    "detail": "Customer 999999 not found"
}
```

### 500 Internal Server Error
```json
{
    "detail": "Database connection failed"
}
```

---

## 9. Production Deployment Checklist

Before deploying to production:

- [ ] Add authentication (JWT, OAuth2, API keys)
- [ ] Enable Redis caching
- [ ] Set up HTTPS/TLS
- [ ] Configure CORS for specific origins
- [ ] Add rate limiting
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure logging (structured JSON logs)
- [ ] Add request validation and sanitization
- [ ] Set up load balancing
- [ ] Configure health check intervals
- [ ] Add API versioning
- [ ] Create API documentation portal
- [ ] Set up CI/CD pipeline
- [ ] Configure backup and disaster recovery

---

## 10. Swagger UI Features

Access http://localhost:8000/docs to:

1. **Browse Endpoints** - View all available API endpoints
2. **Try It Out** - Test endpoints directly in the browser
3. **View Schemas** - Inspect request/response models
4. **Download OpenAPI Spec** - Export API definition
5. **Generate Client Code** - Auto-generate client libraries

---

## Support

- **API Documentation:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json
- **Health Check:** http://localhost:8000/health
