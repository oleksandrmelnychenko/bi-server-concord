# API Endpoints Documentation

**Version:** 3.0.0
**Service:** B2B Product Recommendation API
**Base URL:** `http://localhost:8000` (development) or your production domain
**Model Performance:** 75.4% precision@50 (Improved V3 with Connection Pooling)

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core Endpoints](#core-endpoints)
   - [POST /recommend](#post-recommend)
   - [GET /forecast/{product_id}](#get-forecastproduct_id)
4. [Supporting Endpoints](#supporting-endpoints)
5. [Error Handling](#error-handling)
6. [Performance Characteristics](#performance-characteristics)
7. [Caching Behavior](#caching-behavior)
8. [Usage Examples](#usage-examples)

---

## Overview

This API provides enterprise-grade B2B product recommendations and sales forecasting capabilities. The service uses advanced machine learning models with connection pooling to support 20+ concurrent users with <400ms P99 latency.

### Key Features

- **Product Recommendations**: Hybrid collaborative filtering with discovery (V3.1)
- **Sales Forecasting**: Customer-based forecasting with Bayesian statistics
- **Redis Caching**: Sub-100ms response times for hot data
- **Connection Pooling**: 20 base connections, max 30 concurrent
- **Monitoring**: Comprehensive health checks and metrics endpoints

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| P50 Latency | <50ms | Median response time |
| P99 Latency | <100ms | 99th percentile |
| Concurrent Users | 20+ | With connection pooling |
| Recommendation Precision | 75.4% | @50 recommendations |
| Cache Hit Rate | >80% | For production traffic |

---

## Authentication

**Current Version:** No authentication required (development mode)

**Production Recommendation:** Implement API key authentication or OAuth2 before deploying to production.

```python
# Future implementation example
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

---

## Core Endpoints

### POST /recommend

Generate personalized product recommendations for a B2B customer using hybrid collaborative filtering.

#### Endpoint Details

- **URL:** `/recommend`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Response Format:** JSON

#### Request Schema

**RecommendationRequest** (Pydantic Model)

```json
{
  "customer_id": 410376,
  "top_n": 50,
  "as_of_date": "2024-07-01",
  "use_cache": true,
  "include_discovery": true
}
```

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `customer_id` | integer | Yes | - | - | Customer ID to generate recommendations for |
| `top_n` | integer | No | 50 | 1-500 | Number of recommendations to return |
| `as_of_date` | string | No | null | YYYY-MM-DD | Point-in-time recommendations (ISO date) |
| `use_cache` | boolean | No | true | - | Whether to use Redis cache for faster responses |
| `include_discovery` | boolean | No | true | - | Include collaborative filtering for new product discovery (V3.1) |

#### Response Schema

**RecommendationResponse** (Pydantic Model)

```json
{
  "customer_id": 410376,
  "recommendations": [
    {
      "product_id": 25367399,
      "product_name": "Premium Widget",
      "score": 0.95,
      "source": "hybrid",
      "rank": 1,
      "last_purchased": "2024-06-15",
      "purchase_frequency": 12,
      "avg_quantity": 45.5,
      "avg_revenue": 1275.50
    }
  ],
  "count": 50,
  "discovery_count": 12,
  "precision_estimate": 0.754,
  "latency_ms": 45.23,
  "cached": true,
  "timestamp": "2024-11-11T15:30:00.123456"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | integer | Customer ID from request |
| `recommendations` | array | List of recommended products with metadata |
| `recommendations[].product_id` | integer | Unique product identifier |
| `recommendations[].product_name` | string | Product display name |
| `recommendations[].score` | float | Recommendation confidence score (0-1) |
| `recommendations[].source` | string | Algorithm source: 'repurchase', 'discovery', or 'hybrid' |
| `recommendations[].rank` | integer | Ranking position (1-N) |
| `recommendations[].last_purchased` | string | Last purchase date (ISO format) |
| `recommendations[].purchase_frequency` | integer | Number of times purchased historically |
| `recommendations[].avg_quantity` | float | Average quantity per order |
| `recommendations[].avg_revenue` | float | Average revenue per order |
| `count` | integer | Total number of recommendations returned |
| `discovery_count` | integer | Number of NEW products (not previously purchased) |
| `precision_estimate` | float | Expected accuracy based on validation (75.4%) |
| `latency_ms` | float | Response time in milliseconds |
| `cached` | boolean | Whether served from cache |
| `timestamp` | string | Response generation timestamp (ISO format) |

#### Recommendation Precision by Customer Segment

| Segment | Order Count | Precision@50 | Use Case |
|---------|-------------|--------------|----------|
| Heavy Users | 500+ orders | 89.2% | High-volume B2B customers |
| Regular Users | 100-500 orders | 88.2% | Medium-volume customers |
| Light Users | <100 orders | 54.1% | New or low-volume customers |
| **Overall** | All segments | **75.4%** | Average across all customers |

#### Discovery Mode (V3.1)

When `include_discovery=true`, the API uses collaborative filtering to recommend:

1. **Repurchase Products**: Items the customer has bought before
2. **Discovery Products**: NEW products based on similar customers (Jaccard similarity)

**Blending Strategy:**
- Heavy users: 80% repurchase, 20% discovery
- Regular users: 70% repurchase, 30% discovery
- Light users: 40% repurchase, 60% discovery

#### Query Parameters

None (all parameters are in request body)

#### Error Responses

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | Bad Request | Invalid date format or parameter validation failed |
| 404 | Not Found | Customer ID not found in database |
| 500 | Internal Server Error | Database connection error or model failure |
| 503 | Service Unavailable | Redis cache unavailable (falls back to direct computation) |

**Error Response Format:**

```json
{
  "detail": "Invalid date format: 2024-13-01. Use YYYY-MM-DD"
}
```

#### Caching Behavior

The `/recommend` endpoint uses a **two-tier caching strategy**:

1. **Worker Cache** (Primary): Pre-computed weekly recommendations
   - Key Format: `recommendations:customer:{customer_id}:{as_of_date}`
   - TTL: 7 days
   - Generated by weekly batch job
   - Fastest retrieval (<10ms)

2. **Ad-hoc Cache** (Fallback): On-demand computation results
   - Key Format: `recs:v1:{customer_id}:{top_n}:{date}`
   - TTL: 1 hour (configurable via `CACHE_TTL` env var)
   - Generated on first request
   - Fast retrieval (~50ms)

Cache can be disabled by setting `use_cache=false` in request.

#### Performance Characteristics

| Scenario | Latency | Notes |
|----------|---------|-------|
| Worker Cache Hit | <10ms | Pre-computed recommendations |
| Ad-hoc Cache Hit | ~50ms | Previously computed on-demand |
| Cache Miss (Cold Start) | ~2s | First computation includes discovery |
| Cache Miss (No Discovery) | ~500ms | Repurchase-only mode |
| Concurrent Requests (20+) | <400ms | P99 with connection pooling |

#### Usage Examples

**Basic Recommendation Request**

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 410376,
    "top_n": 50
  }'
```

**Point-in-Time Recommendations (Historical)**

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 410376,
    "top_n": 50,
    "as_of_date": "2024-07-01",
    "use_cache": true
  }'
```

**Repurchase-Only Mode (No Discovery)**

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 410376,
    "top_n": 25,
    "include_discovery": false
  }'
```

**Top 100 Recommendations (No Cache)**

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 410376,
    "top_n": 100,
    "use_cache": false
  }'
```

---

### GET /forecast/{product_id}

Generate 3-month sales forecast for a specific product using customer-based predictive analytics.

#### Endpoint Details

- **URL:** `/forecast/{product_id}`
- **Method:** `GET`
- **Response Format:** JSON

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `product_id` | integer | Yes | Unique product identifier |

#### Query Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `as_of_date` | string | No | Today | YYYY-MM-DD | Reference date for forecast (ISO format) |
| `forecast_weeks` | integer | No | 12 | 4-26 | Number of weeks to forecast (~3 months default) |
| `use_cache` | boolean | No | true | - | Use Redis cache for performance |

#### Response Schema

**ProductForecastResponse** (Pydantic Model)

```json
{
  "product_id": 25367399,
  "product_name": "Premium Widget",
  "forecast_period_weeks": 12,
  "historical_weeks": 52,
  "summary": {
    "total_predicted_quantity": 2450.0,
    "total_predicted_revenue": 85750.00,
    "total_predicted_orders": 87.0,
    "active_customers": 34,
    "at_risk_customers": 5
  },
  "weekly_data": [
    {
      "week_start": "2024-10-01",
      "week_end": "2024-10-07",
      "quantity": 245.0,
      "revenue": 8575.00,
      "orders": 8.0,
      "data_type": "actual",
      "confidence_lower": null,
      "confidence_upper": null,
      "expected_customers": []
    },
    {
      "week_start": "2024-11-11",
      "week_end": "2024-11-17",
      "quantity": 245.0,
      "revenue": 8575.00,
      "orders": 8.0,
      "data_type": "predicted",
      "confidence_lower": 180.0,
      "confidence_upper": 310.0,
      "expected_customers": [
        {
          "customer_id": 412138,
          "customer_name": "Acme Corp",
          "probability": 0.85,
          "expected_quantity": 45.0,
          "expected_date": "2024-11-14",
          "days_since_last_order": 17,
          "avg_reorder_cycle": 21.0
        }
      ]
    }
  ],
  "top_customers_by_volume": [
    {
      "customer_id": 410376,
      "customer_name": "BigCorp Inc",
      "predicted_quantity": 480.0,
      "contribution_pct": 19.6
    }
  ],
  "at_risk_customers": [
    {
      "customer_id": 410999,
      "customer_name": "LateOrders Inc",
      "last_order": "2024-09-15",
      "expected_reorder": "2024-10-20",
      "days_overdue": 21,
      "churn_probability": 0.65,
      "action": "proactive_outreach_recommended"
    }
  ],
  "model_metadata": {
    "model_type": "customer_based_aggregate",
    "training_customers": 34,
    "forecast_accuracy_estimate": 0.78,
    "seasonality_detected": true,
    "model_version": "1.0.0",
    "statistical_methods": [
      "bayesian_inference",
      "mann_kendall_trend",
      "fft_seasonality",
      "survival_analysis"
    ]
  }
}
```

#### Response Field Descriptions

**Top-Level Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | integer | Product identifier from request |
| `product_name` | string\|null | Product display name (if available) |
| `forecast_period_weeks` | integer | Number of future weeks forecasted |
| `historical_weeks` | integer | Number of historical weeks included in response |

**Summary Object (ForecastSummary):**

| Field | Type | Description |
|-------|------|-------------|
| `total_predicted_quantity` | float | Total units forecasted for entire period |
| `total_predicted_revenue` | float | Total revenue forecasted (USD) |
| `total_predicted_orders` | float | Expected number of orders |
| `active_customers` | integer | Customers predicted to order during period |
| `at_risk_customers` | integer | Customers at risk of churning |

**Weekly Data Array (WeeklyForecast):**

Each object contains:

| Field | Type | Description |
|-------|------|-------------|
| `week_start` | string | Week start date (ISO format, Monday) |
| `week_end` | string | Week end date (ISO format, Sunday) |
| `quantity` | float | Units (actual or predicted) |
| `revenue` | float | Revenue in USD (actual or predicted) |
| `orders` | float | Number of orders (actual or predicted) |
| `data_type` | string | 'actual' (historical) or 'predicted' (forecast) |
| `confidence_lower` | float\|null | 95% confidence interval lower bound (predictions only) |
| `confidence_upper` | float\|null | 95% confidence interval upper bound (predictions only) |
| `expected_customers` | array | Customers likely to order this week (probability ≥ 15%) |

**Expected Customer Object (ExpectedCustomer):**

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | integer | Customer identifier |
| `customer_name` | string | Customer display name |
| `probability` | float | Probability of ordering (0-1) |
| `expected_quantity` | float | Expected order quantity |
| `expected_date` | string | Predicted order date (ISO format) |
| `days_since_last_order` | integer | Days since last purchase |
| `avg_reorder_cycle` | float | Average days between orders |

**Top Customer Object (TopCustomer):**

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | integer | Customer identifier |
| `customer_name` | string | Customer display name |
| `predicted_quantity` | float | Total predicted quantity for period |
| `contribution_pct` | float | Percentage of total forecasted volume |

**At-Risk Customer Object (AtRiskCustomer):**

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | integer | Customer identifier |
| `customer_name` | string | Customer display name |
| `last_order` | string | Last order date (ISO format) |
| `expected_reorder` | string | Expected reorder date based on historical cycle |
| `days_overdue` | integer | Days past expected reorder date (0 if not overdue) |
| `churn_probability` | float | Probability of customer churn (0-1) |
| `action` | string | Recommended action (see table below) |

**At-Risk Action Types:**

| Action | Criteria | Recommendation |
|--------|----------|----------------|
| `urgent_outreach_required` | >30 days overdue, churn prob >0.7 | Immediate sales contact required |
| `proactive_outreach_recommended` | 15-30 days overdue, churn prob 0.4-0.7 | Schedule follow-up call |
| `monitor_closely` | <15 days overdue, churn prob <0.4 | Continue monitoring |

**Model Metadata Object (ModelMetadata):**

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | Always "customer_based_aggregate" |
| `training_customers` | integer | Number of customers used for training |
| `forecast_accuracy_estimate` | float | Estimated accuracy (0-1, typically 0.75-0.85) |
| `seasonality_detected` | boolean | Whether seasonal patterns were detected (FFT analysis) |
| `model_version` | string | Model version identifier |
| `statistical_methods` | array[string] | List of statistical methods applied |

**Statistical Methods:**

- `bayesian_inference`: Bayesian forecasting for uncertainty quantification
- `mann_kendall_trend`: Trend detection for demand patterns
- `fft_seasonality`: Fast Fourier Transform for seasonal decomposition
- `survival_analysis`: Customer churn prediction

#### Error Responses

| Status Code | Error Type | Description | Response Model |
|-------------|------------|-------------|----------------|
| 404 | Not Found | No predictable customers for product | ForecastErrorResponse |
| 400 | Bad Request | Invalid query parameters | ForecastErrorResponse |
| 500 | Internal Server Error | Model failure or database error | ForecastErrorResponse |

**ForecastErrorResponse Schema:**

```json
{
  "error": "No predictable customers found for product 25367399",
  "detail": "Product requires at least 3 active customers with regular ordering patterns",
  "product_id": 25367399
}
```

#### Caching Behavior

**Cache Strategy:**

- **Key Format**: `forecast:{product_id}:{as_of_date}:{forecast_weeks}`
- **TTL**: 1 hour (configurable via `CACHE_TTL` environment variable)
- **Cache Backend**: Redis
- **Invalidation**: Automatic expiration, manual clear via `/cache/clear-all`

**Cache Hit Performance:**
- Cache Hit: ~10-20ms
- Cache Miss: ~1-2s (for 20-40 customer products)

**When Cache is Bypassed:**
- `use_cache=false` in query parameter
- Redis unavailable (automatic fallback to direct computation)
- First request for specific product/date/weeks combination

#### Performance Characteristics

| Product Size | Active Customers | Latency (Cache Miss) | Latency (Cache Hit) |
|--------------|------------------|---------------------|---------------------|
| Small | 5-10 customers | ~500ms | ~10ms |
| Medium | 10-40 customers | ~1-2s | ~15ms |
| Large | 40+ customers | ~2-5s | ~20ms |

**Performance Notes:**

- Products with more historical data generate more accurate forecasts
- Minimum 3 active customers required for forecast generation
- Confidence intervals widen for products with volatile demand
- Seasonal detection improves accuracy for products with 12+ months of data

#### Usage Examples

**Basic Forecast (12 weeks)**

```bash
curl -X GET "http://localhost:8000/forecast/25367399"
```

**6-Month Forecast with Specific Date**

```bash
curl -X GET "http://localhost:8000/forecast/25367399?forecast_weeks=26&as_of_date=2024-07-01"
```

**4-Week Forecast (No Cache)**

```bash
curl -X GET "http://localhost:8000/forecast/25367399?forecast_weeks=4&use_cache=false"
```

**Historical Point-in-Time Forecast**

```bash
curl -X GET "http://localhost:8000/forecast/25367399?as_of_date=2024-01-01&forecast_weeks=12"
```

**Example with jq for Pretty Output**

```bash
curl -X GET "http://localhost:8000/forecast/25367399?forecast_weeks=12" | jq '.'
```

**Extract Only Summary Statistics**

```bash
curl -X GET "http://localhost:8000/forecast/25367399" | jq '.summary'
```

**Get At-Risk Customers Only**

```bash
curl -X GET "http://localhost:8000/forecast/25367399" | jq '.at_risk_customers'
```

---

## Supporting Endpoints

### GET /

Root endpoint providing basic API information.

**Response:**

```json
{
  "service": "B2B Product Recommendation API",
  "version": "3.0.0",
  "status": "operational",
  "model_performance": "75.4% precision@50 (Improved V3 + Connection Pooling)",
  "docs": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

**Usage:**

```bash
curl http://localhost:8000/
```

---

### GET /health

Health check endpoint for monitoring and load balancers.

**Response Schema (HealthResponse):**

```json
{
  "status": "healthy",
  "version": "3.0.0",
  "uptime_seconds": 86400.5,
  "metrics": {
    "requests": 15234,
    "cache_hit_rate": 0.856,
    "error_rate": 0.002,
    "avg_latency_ms": 47.3
  },
  "redis_connected": true,
  "model_version": "improved_hybrid_v3_75.4pct_pooled"
}
```

**Usage:**

```bash
curl http://localhost:8000/health
```

---

### GET /metrics

Detailed performance metrics for monitoring dashboards.

**Response Schema (MetricsResponse):**

```json
{
  "total_requests": 15234,
  "cache_hit_rate": 0.856,
  "error_rate": 0.002,
  "avg_latency_ms": 47.3,
  "p99_target_ms": 100,
  "p50_target_ms": 50
}
```

**Usage:**

```bash
curl http://localhost:8000/metrics
```

---

### GET /weekly-recommendations/{customer_id}

Get pre-computed weekly recommendations (faster alternative to `/recommend`).

**Path Parameters:**
- `customer_id` (integer): Customer ID

**Response:**

```json
{
  "customer_id": 410376,
  "week": "2025_W45",
  "recommendations": [...],
  "count": 25,
  "discovery_count": 8,
  "cached": true,
  "latency_ms": 8.5,
  "timestamp": "2024-11-11T15:30:00"
}
```

**Performance:**
- Cache Hit: <10ms
- Cache Miss: ~2s (fallback to on-demand generation)

**Usage:**

```bash
curl http://localhost:8000/weekly-recommendations/410376
```

---

### DELETE /cache/{customer_id}

Clear all cached recommendations for a specific customer.

**Path Parameters:**
- `customer_id` (integer): Customer ID

**Response:**

```json
{
  "status": "success",
  "deleted_keys": 12
}
```

**Usage:**

```bash
curl -X DELETE http://localhost:8000/cache/410376
```

---

### POST /cache/clear-all

Clear all cached recommendations (use with caution in production).

**Response:**

```json
{
  "status": "success",
  "deleted_keys": 1523
}
```

**Usage:**

```bash
curl -X POST http://localhost:8000/cache/clear-all
```

---

## Error Handling

### Standard Error Response Format

All error responses follow this structure:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Status Code | Meaning | Common Causes |
|-------------|---------|---------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters, malformed date |
| 404 | Not Found | Customer/Product not found, insufficient data |
| 500 | Internal Server Error | Database error, model failure |
| 503 | Service Unavailable | Redis unavailable (falls back gracefully) |

### Common Error Scenarios

**Invalid Date Format:**

```bash
# Request
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 410376, "as_of_date": "2024-13-01"}'

# Response (400)
{
  "detail": "Invalid date format: 2024-13-01. Use YYYY-MM-DD"
}
```

**Product Not Found:**

```bash
# Request
curl "http://localhost:8000/forecast/99999999"

# Response (404)
{
  "error": "No predictable customers found for product 99999999",
  "detail": "Product requires at least 3 active customers with regular ordering patterns",
  "product_id": 99999999
}
```

**Redis Unavailable (Graceful Degradation):**

```bash
# Service continues without cache
# Response includes: "cached": false
# Latency increases but request succeeds
```

---

## Performance Characteristics

### Latency Breakdown

**Recommendation Endpoint (`/recommend`):**

| Scenario | P50 | P99 | Max |
|----------|-----|-----|-----|
| Worker Cache Hit | 5ms | 10ms | 15ms |
| Ad-hoc Cache Hit | 30ms | 50ms | 100ms |
| Cache Miss (with discovery) | 1.5s | 2.5s | 5s |
| Cache Miss (no discovery) | 300ms | 500ms | 1s |

**Forecast Endpoint (`/forecast/{product_id}`):**

| Product Size | P50 | P99 | Max |
|--------------|-----|-----|-----|
| Cache Hit | 10ms | 20ms | 30ms |
| Small (5-10 customers) | 400ms | 600ms | 1s |
| Medium (10-40 customers) | 1s | 2s | 3s |
| Large (40+ customers) | 2s | 5s | 10s |

### Throughput

- **Connection Pool**: 20 base connections, 30 max
- **Concurrent Requests**: 20+ simultaneous users
- **Request Queue**: Automatic queuing for >30 concurrent requests
- **Rate Limiting**: None (configure in production via reverse proxy)

### Resource Usage

| Resource | Typical | Peak | Notes |
|----------|---------|------|-------|
| Memory | 500MB | 1GB | Per worker process |
| CPU | 20% | 80% | During cold start |
| Database Connections | 20 | 30 | Connection pool |
| Redis Connections | 1 | 5 | Shared across workers |

---

## Caching Behavior

### Cache Hierarchy

```
Request → Worker Cache (7 days)
              ↓ miss
          Ad-hoc Cache (1 hour)
              ↓ miss
          Direct Computation
              ↓
          Store in Ad-hoc Cache
```

### Cache Keys

**Recommendations:**
- Worker Cache: `recommendations:customer:{customer_id}:{as_of_date}`
- Ad-hoc Cache: `recs:v1:{customer_id}:{top_n}:{date}`

**Forecasts:**
- `forecast:{product_id}:{as_of_date}:{forecast_weeks}`

### Cache TTL Configuration

Environment variables:

```bash
export CACHE_TTL=3600        # 1 hour (default)
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_DB=0
```

### Cache Invalidation

**Automatic:**
- TTL expiration (1 hour for ad-hoc, 7 days for worker cache)

**Manual:**
- `DELETE /cache/{customer_id}` - Clear specific customer
- `POST /cache/clear-all` - Clear all caches

**Recommendation:**
- Invalidate cache after bulk data imports
- Use `use_cache=false` for testing new algorithms

---

## Usage Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Get recommendations
def get_recommendations(customer_id, top_n=50):
    response = requests.post(
        f"{BASE_URL}/recommend",
        json={
            "customer_id": customer_id,
            "top_n": top_n,
            "include_discovery": True
        }
    )
    return response.json()

# Get forecast
def get_forecast(product_id, weeks=12):
    response = requests.get(
        f"{BASE_URL}/forecast/{product_id}",
        params={"forecast_weeks": weeks}
    )
    return response.json()

# Example usage
recs = get_recommendations(410376, top_n=25)
print(f"Got {recs['count']} recommendations")
print(f"Discovery products: {recs['discovery_count']}")
print(f"Latency: {recs['latency_ms']}ms")

forecast = get_forecast(25367399, weeks=12)
print(f"Predicted quantity: {forecast['summary']['total_predicted_quantity']}")
print(f"At-risk customers: {forecast['summary']['at_risk_customers']}")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

// Get recommendations
async function getRecommendations(customerId, topN = 50) {
  const response = await axios.post(`${BASE_URL}/recommend`, {
    customer_id: customerId,
    top_n: topN,
    include_discovery: true
  });
  return response.data;
}

// Get forecast
async function getForecast(productId, weeks = 12) {
  const response = await axios.get(`${BASE_URL}/forecast/${productId}`, {
    params: { forecast_weeks: weeks }
  });
  return response.data;
}

// Example usage
(async () => {
  const recs = await getRecommendations(410376, 25);
  console.log(`Got ${recs.count} recommendations`);
  console.log(`Latency: ${recs.latency_ms}ms`);

  const forecast = await getForecast(25367399, 12);
  console.log(`Total predicted: ${forecast.summary.total_predicted_quantity}`);
})();
```

### cURL Advanced Examples

**Batch Processing Multiple Customers:**

```bash
#!/bin/bash
for customer_id in 410376 410999 412138; do
  echo "Processing customer $customer_id..."
  curl -X POST "http://localhost:8000/recommend" \
    -H "Content-Type: application/json" \
    -d "{\"customer_id\": $customer_id, \"top_n\": 25}" \
    | jq '.count, .latency_ms'
done
```

**Monitor Health in Loop:**

```bash
#!/bin/bash
while true; do
  curl -s http://localhost:8000/health | jq '.metrics'
  sleep 5
done
```

**Parallel Forecasts:**

```bash
#!/bin/bash
products=(25367399 25367400 25367401)
for product in "${products[@]}"; do
  curl -X GET "http://localhost:8000/forecast/$product?forecast_weeks=8" \
    > "forecast_${product}.json" &
done
wait
echo "All forecasts complete"
```

---

## Interactive API Documentation

### Swagger UI

Access interactive API documentation at:

```
http://localhost:8000/docs
```

Features:
- Try-it-out functionality for all endpoints
- Request/response schema validation
- Example payloads
- Authentication testing

### ReDoc

Alternative documentation interface:

```
http://localhost:8000/redoc
```

Features:
- Clean, readable format
- Downloadable OpenAPI spec
- Code generation examples

---

## Production Deployment Checklist

- [ ] Configure authentication (API keys or OAuth2)
- [ ] Set up rate limiting (via nginx/API Gateway)
- [ ] Configure Redis in cluster mode for HA
- [ ] Set appropriate `CACHE_TTL` values
- [ ] Enable CORS for specific domains only
- [ ] Set up monitoring alerts (Prometheus/Grafana)
- [ ] Configure database connection pool limits
- [ ] Enable request logging and audit trails
- [ ] Set up backup for Redis cache
- [ ] Configure health check endpoints in load balancer

---

**Last Updated:** 2024-11-11
**API Version:** 3.0.0
**Model Version:** improved_hybrid_v3_75.4pct_pooled
