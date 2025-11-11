# Forecast Worker Documentation

## Overview

The Forecast Worker is a background service that pre-computes product demand forecasts for all forecastable products in the system. It processes thousands of products in parallel, stores results in Redis cache with a 7-day TTL, and provides the foundation for real-time forecast API responses.

**Location**: `/scripts/forecasting/forecast_worker.py`

**Purpose**: Batch-generate 12-week demand forecasts for all eligible products and cache results for fast API access.

**Performance**: ~10,000 products in 8-10 minutes using 10 parallel workers.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Product Queue Processing](#product-queue-processing)
3. [Batch Forecast Generation](#batch-forecast-generation)
4. [Redis Caching Strategy](#redis-caching-strategy)
5. [Error Recovery](#error-recovery)
6. [Scheduling Logic](#scheduling-logic)
7. [Configuration](#configuration)
8. [Operational Guidelines](#operational-guidelines)
9. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FORECAST WORKER                          │
│                                                                 │
│  ┌────────────────┐    ┌──────────────────────────────────┐   │
│  │  Main Process  │───▶│  ForecastWorker Controller       │   │
│  └────────────────┘    └──────────────────────────────────┘   │
│                                    │                            │
│                                    ▼                            │
│                        ┌───────────────────────┐               │
│                        │ Query Database for    │               │
│                        │ Forecastable Products │               │
│                        └───────────────────────┘               │
│                                    │                            │
│                                    ▼                            │
│                        ┌───────────────────────┐               │
│                        │ Create Work Queue     │               │
│                        │ (Product IDs + Index) │               │
│                        └───────────────────────┘               │
│                                    │                            │
│                                    ▼                            │
│            ┌───────────────────────────────────────┐           │
│            │   Multiprocessing Pool (10 workers)   │           │
│            └───────────────────────────────────────┘           │
│                    │       │       │       │                   │
│           ┌────────┴───────┴───────┴───────┴────────┐          │
│           ▼        ▼       ▼       ▼       ▼        ▼          │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│    │ Worker 1 │ │ Worker 2 │ │ Worker N │ │ Worker   │        │
│    │          │ │          │ │   ...    │ │    10    │        │
│    └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│         │            │            │            │               │
│         └────────────┴────────────┴────────────┘               │
│                          │                                     │
└──────────────────────────┼─────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
    ┌─────────────┐              ┌──────────────────┐
    │   MS SQL    │              │   Redis Cache    │
    │  Database   │              │                  │
    │             │              │  forecast:{pid}  │
    │ - Orders    │              │  - Forecast JSON │
    │ - Products  │              │  - 7-day TTL     │
    │ - Customers │              │  - Metadata      │
    └─────────────┘              └──────────────────┘
```

### Component Responsibilities

1. **Main Process**
   - Initialize worker configuration
   - Connect to Redis and validate connectivity
   - Query database for eligible products
   - Create multiprocessing pool
   - Aggregate results and generate statistics

2. **Worker Processes** (N=10 default)
   - Each worker is independent with own DB/Redis connections
   - Fetches historical order data for assigned product
   - Invokes ForecastEngine to generate 12-week forecast
   - Stores forecast in Redis with TTL
   - Returns success/failure status

3. **Database**
   - Provides historical order data (2019-present)
   - Filtered by customer count and order thresholds
   - Read-only access during processing

4. **Redis Cache**
   - Stores forecast results as JSON
   - Key pattern: `forecast:{product_id}`
   - 7-day TTL (configurable)
   - Stores metadata about last run

---

## Product Queue Processing

### Eligibility Criteria

Products must meet ALL criteria to be forecasted:

```sql
SELECT oi.ProductID
FROM dbo.OrderItem oi
INNER JOIN dbo.[Order] o ON oi.OrderID = o.ID
INNER JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
WHERE o.Created >= '2019-01-01'
  AND o.Created < :as_of_date
  AND oi.ProductID IS NOT NULL
GROUP BY oi.ProductID
HAVING COUNT(DISTINCT ca.ClientID) >= :min_customers  -- Default: 2
   AND COUNT(DISTINCT o.ID) >= :min_orders            -- Default: 3
ORDER BY COUNT(DISTINCT o.ID) DESC
```

**Requirements:**
- **Historical Range**: Orders from 2019-01-01 to AS_OF_DATE
- **Minimum Customers**: At least 2 unique customers (configurable)
- **Minimum Orders**: At least 3 total orders (configurable)
- **Active Product**: ProductID must not be NULL

**Sorting**: Products are ordered by total order count (descending), prioritizing high-volume products.

### Queue Construction

```python
# Prepare arguments for parallel processing
args_list = [
    (product_id, index, total_count, as_of_date, cache_ttl)
    for index, product_id in enumerate(products)
]
```

Each work item contains:
- `product_id`: The product to forecast
- `index`: Position in queue (for progress tracking)
- `total_count`: Total products in queue
- `as_of_date`: Forecast start date
- `cache_ttl`: Redis TTL in seconds

### Parallel Processing Flow

```python
# Process in parallel using multiprocessing Pool
with Pool(processes=self.num_workers) as pool:
    results = pool.map(_process_product_worker, args_list)
```

**Key Design Points:**

1. **Multiprocessing vs Threading**: Uses `multiprocessing.Pool` (not threading) to bypass Python GIL and achieve true parallelism for CPU-bound forecast calculations.

2. **Process Isolation**: Each worker process:
   - Creates independent DB connection
   - Creates independent Redis client
   - Cannot share thread locks (connections recreated per process)
   - Closes connections after processing

3. **Work Distribution**: Pool automatically distributes work items to available workers using round-robin or dynamic scheduling.

---

## Batch Forecast Generation

### Worker Process Flow

```python
def _process_product_worker(args):
    product_id, index, total_count, as_of_date, cache_ttl = args

    # 1. Create isolated connections
    conn = get_connection()
    redis_client = redis.Redis(...)

    # 2. Initialize forecast engine
    engine = ForecastEngine(conn=conn, forecast_weeks=12)

    # 3. Generate forecast with caching
    forecast = engine.generate_forecast_cached(
        product_id=product_id,
        redis_client=redis_client,
        as_of_date=as_of_date,
        cache_ttl=cache_ttl
    )

    # 4. Log progress every 100 products
    if (index + 1) % 100 == 0:
        logger.info(f"Progress: {index + 1}/{total_count}...")

    # 5. Return result
    return {
        'product_id': product_id,
        'status': 'success',
        'quantity': forecast.summary['total_predicted_quantity'],
        'customers': forecast.summary['active_customers'],
        'confidence': forecast.model_metadata['forecast_accuracy_estimate']
    }
```

### Forecast Generation Stages

1. **Data Retrieval**
   - Query historical orders for product
   - Aggregate by week
   - Calculate customer purchase patterns

2. **Model Selection**
   - Statistical model (moving averages, seasonality)
   - ML model (if sufficient data)
   - Fallback to baseline if insufficient data

3. **Prediction**
   - Generate 12-week forward forecast
   - Calculate confidence intervals
   - Identify active vs. churned customers

4. **Result Formatting**
   - Weekly predictions with quantities
   - Customer-level breakdown
   - Metadata (model type, accuracy estimate)

5. **Caching**
   - Serialize forecast to JSON
   - Store in Redis with product-specific key
   - Set TTL (default: 7 days)

### Result Structure

```json
{
  "product_id": 12345,
  "status": "success",
  "quantity": 1450.5,
  "customers": 47,
  "confidence": 0.78,
  "elapsed": 0.23
}
```

**Possible Status Values:**
- `success`: Forecast generated and cached
- `no_forecast`: Insufficient data for forecasting
- `error`: Exception occurred during processing

---

## Redis Caching Strategy

### Cache Key Structure

```
forecast:{product_id}
```

**Example**: `forecast:12345`

### Data Format

Each cached forecast contains:

```json
{
  "product_id": 12345,
  "as_of_date": "2025-11-11",
  "forecast_weeks": 12,
  "predictions": [
    {
      "week_start": "2025-11-11",
      "week_end": "2025-11-17",
      "predicted_quantity": 120.5,
      "confidence_lower": 100.2,
      "confidence_upper": 140.8,
      "active_customers": 15
    }
    // ... 11 more weeks
  ],
  "summary": {
    "total_predicted_quantity": 1450.5,
    "active_customers": 47,
    "avg_weekly_quantity": 120.9
  },
  "model_metadata": {
    "model_type": "statistical",
    "forecast_accuracy_estimate": 0.78,
    "historical_weeks": 52,
    "generated_at": "2025-11-11T06:00:00Z"
  }
}
```

### TTL Strategy

**Default TTL**: 7 days (604,800 seconds)

**Rationale**:
- Demand patterns typically stable over 1 week
- Weekly worker run refreshes all caches
- Balances freshness vs. compute cost

**Environment Override**:
```bash
export FORECAST_CACHE_TTL=604800  # 7 days
```

### Metadata Storage

Worker stores run metadata in Redis:

```python
metadata = {
    'last_run': '2025-11-11T06:00:00',
    'as_of_date': '2025-11-11',
    'total_products': 10247,
    'successful': 9850,
    'failed': 12,
    'no_forecast': 385,
    'elapsed_seconds': 542
}

redis_client.setex(
    'forecast:metadata:last_run',
    cache_ttl,
    json.dumps(metadata)
)
```

**Purpose**: Enables monitoring dashboards to track worker health and cache freshness.

### Cache Warming

The worker performs **full cache warming** on each run:

1. Identifies all forecastable products
2. Processes ALL products (even if cached)
3. Overwrites existing cache entries
4. Extends TTL to full 7 days

**No incremental updates**: Simpler, more predictable cache state.

---

## Error Recovery

### Error Handling at Worker Level

```python
try:
    # Generate forecast
    forecast = engine.generate_forecast_cached(...)
    return {'status': 'success', ...}

except Exception as e:
    logger.error(f"Error processing product {product_id}: {e}")
    return {
        'product_id': product_id,
        'status': 'error',
        'error': str(e)
    }
```

**Key Points:**
- Each worker catches its own exceptions
- Failed products don't block other workers
- Errors logged with product ID and traceback
- Failed products return error status

### Resource Cleanup

```python
try:
    # Process product
    ...
finally:
    # Always close connections
    conn.close()
    redis_client.close()
```

**Guaranteed Cleanup**: Connections closed even if error occurs.

### Aggregated Error Reporting

After processing completes, main process reports:

```
Total Products: 10,247
Successful: 9,850 (96.1%)
No Forecast: 385 (3.8%)
Failed: 12 (0.1%)
```

**Exit Codes:**
- `0`: All products processed successfully
- `1`: Some products failed OR worker-level exception
- `130`: User interrupted (SIGINT)

### Common Error Scenarios

| Error Type | Cause | Recovery |
|------------|-------|----------|
| **Database Connection** | Network/timeout | Worker retries on next product; main job continues |
| **Redis Connection** | Redis unavailable | Worker fails; main job exits with error |
| **Insufficient Data** | Product below threshold | Returns `no_forecast` status; not counted as failure |
| **Model Exception** | Bug in ForecastEngine | Logged as error; product skipped; job continues |
| **Memory Error** | Too many workers | Reduce `FORECAST_WORKERS` env var |

### Retry Strategy

**No automatic retries** at worker level:
- Each product processed exactly once per run
- Failed products must wait for next scheduled run
- Rationale: Batch job runs frequently (daily/weekly); transient errors resolve naturally

**Manual Retry**: Re-run worker with same AS_OF_DATE.

---

## Scheduling Logic

### Recommended Schedule

**Option 1: Daily (Best for Production)**
```cron
# Run at 6:00 AM every day
0 6 * * * cd /path/to/Concord-BI-Server && python3 scripts/forecasting/forecast_worker.py
```

**Option 2: Weekly (Resource-Constrained)**
```cron
# Run at 6:00 AM every Monday
0 6 * * 1 cd /path/to/Concord-BI-Server && python3 scripts/forecasting/forecast_worker.py
```

**Option 3: Docker Scheduled Service**
```yaml
# docker-compose.yml
services:
  forecast-worker:
    image: concord-bi-server:latest
    command: python3 scripts/forecasting/forecast_worker.py
    environment:
      - AS_OF_DATE=2025-11-11
      - FORECAST_WORKERS=10
      - REDIS_HOST=redis
    depends_on:
      - redis
      - database
```

### Scheduling Considerations

**Time of Day:**
- Run during low-traffic periods (early morning)
- Avoid peak business hours
- Consider database backup windows

**Frequency:**
- Daily: Freshest forecasts, higher compute cost
- Weekly: Lower cost, acceptable staleness for most use cases
- 7-day cache TTL matches weekly schedule

**Concurrency:**
- Only one worker instance should run at a time
- Use cron locks or Kubernetes Jobs (restartPolicy: Never)

### AS_OF_DATE Configuration

**Default**: Current date (`datetime.now()`)

**Override for Historical Runs:**
```bash
export AS_OF_DATE=2025-10-01
python3 scripts/forecasting/forecast_worker.py
```

**Use Cases:**
- Backtesting forecasts against historical data
- Simulating forecasts at specific dates
- Debugging forecast accuracy

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AS_OF_DATE` | Current date | Date to generate forecasts from (YYYY-MM-DD) |
| `REDIS_HOST` | `127.0.0.1` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `FORECAST_CACHE_TTL` | `604800` (7 days) | Cache TTL in seconds |
| `FORECAST_WORKERS` | `10` | Number of parallel worker processes |
| `FORECAST_MIN_CUSTOMERS` | `2` | Minimum customers required for forecast |
| `FORECAST_MIN_ORDERS` | `3` | Minimum orders required for forecast |

### Performance Tuning

**Worker Count:**
```bash
# High-performance server (16+ cores)
export FORECAST_WORKERS=20

# Standard server (4-8 cores)
export FORECAST_WORKERS=10

# Low-resource (2 cores)
export FORECAST_WORKERS=4
```

**Cache TTL Tuning:**
```bash
# High-volatility business (daily refresh)
export FORECAST_CACHE_TTL=86400  # 1 day

# Standard (weekly refresh)
export FORECAST_CACHE_TTL=604800  # 7 days

# Long-term caching (monthly)
export FORECAST_CACHE_TTL=2592000  # 30 days
```

### Database Configuration

Worker inherits DB connection from `api.db_pool`:

```python
# api/db_pool.py configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
```

---

## Operational Guidelines

### Pre-Deployment Checklist

- [ ] Redis server running and accessible
- [ ] Database credentials configured
- [ ] Environment variables set
- [ ] Test run with `--limit 100` flag
- [ ] Monitor resource usage (CPU, memory, DB connections)
- [ ] Schedule cron job or Kubernetes CronJob
- [ ] Set up alerting for failures

### Running the Worker

**Standalone Execution:**
```bash
cd /path/to/Concord-BI-Server
python3 scripts/forecasting/forecast_worker.py
```

**Docker Execution:**
```bash
docker-compose run forecast-worker
```

**Test Run (First 100 Products):**
```bash
# Modify code temporarily to limit products
# Or filter in SQL query with TOP 100
```

### Monitoring

**Check Last Run Metadata:**
```bash
redis-cli GET forecast:metadata:last_run
```

**Output:**
```json
{
  "last_run": "2025-11-11T06:15:23",
  "as_of_date": "2025-11-11",
  "total_products": 10247,
  "successful": 9850,
  "failed": 12,
  "no_forecast": 385,
  "elapsed_seconds": 542
}
```

**Check Individual Forecast:**
```bash
redis-cli GET forecast:12345
```

**Monitor Worker Logs:**
```bash
# Systemd service
journalctl -u forecast-worker -f

# Docker
docker logs -f forecast-worker

# Cron (redirect output)
0 6 * * * /path/to/worker.sh >> /var/log/forecast_worker.log 2>&1
```

### Performance Benchmarks

**Expected Performance (10 workers):**
- 10,000 products: 8-10 minutes
- Average time per product: 0.05-0.06 seconds
- Throughput: ~18-20 products/second
- Redis writes: ~1.5 MB/second

**Resource Usage:**
- CPU: 400-800% (4-8 cores fully utilized)
- Memory: 2-4 GB total (200-400 MB per worker)
- Database connections: 10 concurrent
- Redis connections: 10 concurrent

---

## Monitoring & Troubleshooting

### Health Checks

**Verify Worker Ran Successfully:**
```python
import redis
import json

r = redis.Redis(host='127.0.0.1', decode_responses=True)
metadata = json.loads(r.get('forecast:metadata:last_run'))

if metadata['failed'] > 0:
    print(f"⚠️ Warning: {metadata['failed']} products failed")
else:
    print(f"✅ Success: {metadata['successful']} products forecasted")
```

### Common Issues

**Issue 1: Redis Connection Failed**
```
ERROR - Failed to connect to Redis: [Errno 61] Connection refused
```

**Solution:**
```bash
# Check Redis is running
redis-cli ping

# Verify Redis host/port
echo $REDIS_HOST
echo $REDIS_PORT

# Restart Redis
sudo systemctl restart redis
```

---

**Issue 2: Database Connection Timeout**
```
ERROR - Error processing product 12345: Query timeout
```

**Solution:**
```bash
# Increase DB timeout
export DB_QUERY_TIMEOUT=60

# Reduce worker count to lower DB load
export FORECAST_WORKERS=5
```

---

**Issue 3: Out of Memory**
```
MemoryError: Unable to allocate memory
```

**Solution:**
```bash
# Reduce worker count
export FORECAST_WORKERS=4

# Check memory usage
free -h
```

---

**Issue 4: Slow Performance**
```
Duration: 45.2 minutes (expected: 8-10 min)
```

**Solution:**
```bash
# Check DB query performance
# - Add indexes on OrderItem.ProductID
# - Add indexes on Order.Created

# Increase workers (if CPU available)
export FORECAST_WORKERS=15

# Check for resource contention
top
iostat
```

### Alerting

**Key Metrics to Alert On:**

1. **Worker Failed**: Exit code != 0
2. **High Failure Rate**: >5% products failed
3. **Long Runtime**: >20 minutes for 10K products
4. **Cache Expiry**: Metadata older than 8 days
5. **Redis Unavailable**: Connection errors

**Example Alert (Prometheus/Alertmanager):**
```yaml
- alert: ForecastWorkerFailed
  expr: forecast_worker_exit_code != 0
  for: 5m
  annotations:
    summary: "Forecast worker failed"
    description: "Worker exited with code {{ $value }}"
```

---

## Summary

The Forecast Worker is a robust, scalable batch processing system that:

- **Processes** 10,000+ products in <10 minutes using parallel multiprocessing
- **Generates** 12-week demand forecasts using statistical and ML models
- **Caches** results in Redis with 7-day TTL for instant API responses
- **Recovers** gracefully from errors without blocking other products
- **Schedules** easily via cron, Docker, or Kubernetes CronJobs
- **Monitors** health via Redis metadata and structured logging

**Next Steps:**
- Set up scheduled runs (cron/K8s)
- Configure monitoring and alerting
- Tune worker count based on infrastructure
- Integrate with forecast API endpoints
