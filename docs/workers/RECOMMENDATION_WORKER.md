# Weekly Recommendation Worker Documentation

## Overview

The Weekly Recommendation Worker is a scheduled background service that pre-generates personalized product recommendations for all active customers. It processes customers in parallel batches, generates 25 recommendations per customer, and stores results in Redis with a weekly cache key for fast API retrieval.

**Location**: `/scripts/weekly_recommendation_worker.py`

**Purpose**: Batch-generate weekly product recommendations for active customers, combining purchased products with discovery items.

**Performance**: ~400 customers/minute using 4 parallel workers.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Weekly Recommendation Generation](#weekly-recommendation-generation)
3. [Batch Customer Processing](#batch-customer-processing)
4. [Cache Warming Strategy](#cache-warming-strategy)
5. [Scheduling](#scheduling)
6. [Configuration](#configuration)
7. [Operational Guidelines](#operational-guidelines)
8. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                 WEEKLY RECOMMENDATION WORKER                      │
│                                                                   │
│  ┌─────────────────┐    ┌───────────────────────────────────┐   │
│  │   Main Process  │───▶│  WeeklyRecommendationWorker       │   │
│  │   (CLI Entry)   │    │  Controller                       │   │
│  └─────────────────┘    └───────────────────────────────────┘   │
│                                     │                             │
│                                     ▼                             │
│                         ┌───────────────────────┐                │
│                         │ Query Active Clients  │                │
│                         │ (Orders in Last 90d)  │                │
│                         └───────────────────────┘                │
│                                     │                             │
│                                     ▼                             │
│                         ┌───────────────────────┐                │
│                         │ Create Customer Queue │                │
│                         │ (Customer IDs List)   │                │
│                         └───────────────────────┘                │
│                                     │                             │
│                                     ▼                             │
│            ┌────────────────────────────────────────┐            │
│            │  ThreadPoolExecutor (4 workers)        │            │
│            │  (Concurrent Futures)                  │            │
│            └────────────────────────────────────────┘            │
│                   │        │        │        │                   │
│          ┌────────┴────────┴────────┴────────┴────────┐          │
│          ▼        ▼        ▼        ▼        ▼        ▼          │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│    │Thread 1 │ │Thread 2 │ │Thread 3 │ │Thread 4 │            │
│    │         │ │         │ │         │ │         │            │
│    └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
│         │           │           │           │                    │
│         └───────────┴───────────┴───────────┘                    │
│                          │                                       │
│                          ▼                                       │
│         ┌─────────────────────────────────────┐                 │
│         │  ImprovedHybridRecommenderV31       │                 │
│         │  - Purchase History Analysis        │                 │
│         │  - Collaborative Filtering          │                 │
│         │  - Product Similarity               │                 │
│         │  - Discovery Recommendations        │                 │
│         └─────────────────────────────────────┘                 │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │
           ┌───────────────┴────────────────┐
           │                                │
           ▼                                ▼
    ┌─────────────┐              ┌──────────────────────┐
    │   MS SQL    │              │    Redis Cache       │
    │  Database   │              │                      │
    │             │              │  Weekly Key:         │
    │ - Orders    │              │  recs:2025-W46       │
    │ - Products  │              │                      │
    │ - Customers │              │  Per Customer:       │
    │ - Purchase  │              │  {week}:{cust_id}    │
    │   History   │              │  - 25 products       │
    └─────────────┘              │  - 8-day TTL         │
                                 │                      │
                                 │  Backup in DB:       │
                                 │  WeeklyRecommendation│
                                 └──────────────────────┘
```

### Component Responsibilities

1. **Main Process**
   - Parse CLI arguments (dry-run, limit, workers)
   - Initialize WeeklyRecommendationWorker
   - Query database for active customers
   - Create ThreadPoolExecutor with N workers
   - Track statistics and print summary

2. **Worker Threads** (N=4 default)
   - Each thread processes individual customers
   - Creates DB connection per customer (from pool)
   - Invokes ImprovedHybridRecommenderV31
   - Generates 25 recommendations (mix of repeat + discovery)
   - Stores in Redis with weekly key
   - Optionally backs up to database

3. **Database**
   - Provides customer data and purchase history
   - Defines "active" customers (orders in last 90 days)
   - Stores backup recommendations (WeeklyRecommendation table)

4. **Redis Cache**
   - Stores recommendations with weekly key pattern
   - 8-day TTL (slightly longer than 7 days for buffer)
   - Fast retrieval for API requests

---

## Weekly Recommendation Generation

### Customer Eligibility

Customers must meet ALL criteria to receive recommendations:

```sql
SELECT DISTINCT c.ID
FROM dbo.Client c
INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
WHERE o.Created >= DATEADD(day, -90, GETDATE())  -- Orders in last 90 days
      AND c.IsActive = 1
      AND c.IsBlocked = 0
      AND c.Deleted = 0
ORDER BY c.ID
```

**Requirements:**
- **Recent Activity**: At least 1 order in last 90 days
- **Active Account**: IsActive = 1
- **Not Blocked**: IsBlocked = 0
- **Not Deleted**: Deleted = 0

**Typical Volume**: 5,000-10,000 active customers.

### Recommendation Algorithm

The worker uses `ImprovedHybridRecommenderV31`, which combines:

1. **Repeat Purchase Predictions**
   - Analyzes customer's historical purchases
   - Identifies products likely to be reordered
   - Considers purchase frequency and recency

2. **Collaborative Filtering**
   - Finds similar customers based on purchase patterns
   - Recommends products popular among similar customers
   - Uses customer-product interaction matrix

3. **Product Similarity**
   - Content-based filtering on product attributes
   - Recommends products similar to past purchases
   - Category, brand, price range matching

4. **Discovery Recommendations**
   - Introduces new products customer hasn't purchased
   - Trending products in customer's categories
   - Seasonal or promotional items

### Recommendation Structure

Each customer receives **25 recommendations**:

```python
recommendations = [
    {
        "product_id": 12345,
        "product_name": "Premium Widget",
        "score": 0.92,
        "source": "repeat",  # or 'discovery', 'collaborative', 'hybrid'
        "category": "Widgets",
        "predicted_quantity": 50,
        "last_purchased": "2025-10-15",
        "purchase_frequency_days": 30,
        "rank": 1
    },
    # ... 24 more products
]
```

**Fields:**
- `product_id`: Unique product identifier
- `product_name`: Display name
- `score`: Recommendation confidence (0-1)
- `source`: Algorithm that recommended this product
- `category`: Product category
- `predicted_quantity`: Suggested order quantity
- `last_purchased`: Date customer last ordered (if applicable)
- `purchase_frequency_days`: Average days between orders
- `rank`: Position in recommendation list (1-25)

### Discovery Mix

**Typical Distribution:**
- **15-20 products**: Repeat purchases (high confidence)
- **5-10 products**: Discovery items (new opportunities)

Discovery percentage varies by customer:
- New customers: Higher discovery (60-70%)
- Loyal customers: Lower discovery (20-30%)
- Dormant customers: Balanced mix (40-50%)

---

## Batch Customer Processing

### Processing Flow

```python
def run(self, limit: int = None):
    # 1. Get active customers
    clients = self.get_active_clients(limit=limit)

    # 2. Process in parallel
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        # Submit all tasks
        future_to_customer = {
            executor.submit(self.process_client, customer_id): customer_id
            for customer_id in clients
        }

        # Process results as they complete
        for future in as_completed(future_to_customer):
            customer_id, success, num_recs, num_discovery, error = future.result()

            # Update statistics
            if success:
                self.stats['successful'] += 1
                self.stats['total_recommendations'] += num_recs
                self.stats['total_discovery'] += num_discovery
            else:
                self.stats['failed'] += 1

            # Progress logging every 10 clients
            if processed % 10 == 0:
                logger.info(f"Progress: {processed}/{total} ({pct}%)")
```

### Individual Customer Processing

```python
def process_client(self, customer_id: int):
    conn = get_connection()
    try:
        # 1. Initialize recommender
        recommender = ImprovedHybridRecommenderV31(conn=conn, use_cache=True)

        # 2. Generate recommendations
        recommendations = recommender.get_recommendations(
            customer_id=customer_id,
            as_of_date=datetime.now().strftime('%Y-%m-%d'),
            top_n=25,
            include_discovery=True
        )

        # 3. Count discovery items
        num_discovery = sum(1 for r in recommendations
                           if r.get('source') in ['discovery', 'hybrid'])

        # 4. Store in Redis
        if not self.dry_run:
            self.cache.store_recommendations(
                customer_id=customer_id,
                recommendations=recommendations,
                week_key=self.week_key
            )

        return (customer_id, True, len(recommendations), num_discovery, "")

    except Exception as e:
        logger.error(f"Customer {customer_id} failed: {e}")
        return (customer_id, False, 0, 0, str(e))
    finally:
        conn.close()
```

### Parallelization Strategy

**ThreadPoolExecutor vs Multiprocessing:**

Worker uses **threading** (not multiprocessing) because:

1. **I/O Bound**: Most time spent waiting on DB queries, not CPU
2. **Connection Pooling**: Can share DB connection pool across threads
3. **Simpler State**: Easier to share Redis client and statistics
4. **Lower Overhead**: Thread creation faster than process forking

**Optimal Worker Count:**
- **4 threads** (default): Balances throughput and DB load
- Can increase to 8-10 for high-performance servers
- Limited by DB connection pool size

### Progress Tracking

Worker logs progress every 10 customers:

```
Progress: 10/5000 (0.2%) | Success: 10 | Failed: 0 | Rate: 2.5 clients/sec | ETA: 33.0 min
Progress: 20/5000 (0.4%) | Success: 20 | Failed: 0 | Rate: 2.7 clients/sec | ETA: 30.8 min
...
Progress: 5000/5000 (100.0%) | Success: 4987 | Failed: 13 | Rate: 2.6 clients/sec | ETA: 0.0 min
```

**Metrics:**
- **Progress**: Current/Total (Percentage)
- **Success**: Count of successfully processed customers
- **Failed**: Count of failed customers
- **Rate**: Throughput (clients/second)
- **ETA**: Estimated time to completion (minutes)

---

## Cache Warming Strategy

### Weekly Cache Key

Recommendations are stored with a **weekly key** derived from ISO week number:

```python
@staticmethod
def get_week_key() -> str:
    """
    Generate weekly cache key (e.g., '2025-W46')

    Returns ISO year and week number
    """
    now = datetime.now()
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"
```

**Example Keys:**
- `2025-W46`: Week 46 of 2025
- `2025-W52`: Week 52 of 2025
- `2026-W01`: Week 1 of 2026

**Benefits:**
- All customers processed in same week share same key
- Easy to invalidate entire week's cache
- API can validate cache freshness by comparing week keys

### Redis Storage Pattern

**Key Structure:**
```
{week_key}:{customer_id}
```

**Examples:**
- `2025-W46:12345`: Recommendations for customer 12345 in week 46
- `2025-W46:67890`: Recommendations for customer 67890 in week 46

**Value Structure:**
```json
{
  "customer_id": 12345,
  "week": "2025-W46",
  "generated_at": "2025-11-11T06:30:15Z",
  "recommendations": [
    {
      "product_id": 789,
      "product_name": "Widget Pro",
      "score": 0.95,
      "source": "repeat",
      "rank": 1
    }
    // ... 24 more
  ]
}
```

### TTL Configuration

**Default TTL**: 8 days (691,200 seconds)

**Rationale:**
- Weekly refresh cycle (7 days)
- 1-day buffer for late worker runs
- Prevents stale data if worker fails

**Redis Command:**
```python
redis_client.setex(
    f"{week_key}:{customer_id}",
    691200,  # 8 days
    json.dumps(recommendations)
)
```

### Cache Warming Schedule

**Full Cache Warming** every week:

1. **Monday 6:00 AM**: Worker runs
2. **All active customers processed**: ~5,000 customers
3. **All old week keys expire**: Redis auto-removes after TTL
4. **New week key populated**: All customers have fresh recommendations

**Incremental Updates** (Not Implemented):
- Worker currently regenerates ALL recommendations
- No incremental updates for individual customers
- Simplifies cache invalidation logic

### Database Backup

Worker optionally stores recommendations in database:

```python
class WeeklyRecommendationCache:
    def store_recommendations(self, customer_id, recommendations, week_key):
        # 1. Store in Redis
        redis_client.setex(...)

        # 2. Backup to database (optional)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO WeeklyRecommendation (CustomerID, Week, ProductID, Rank, Score, Source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ...)
        conn.commit()
```

**Benefits:**
- Audit trail for recommendation history
- Fallback if Redis fails
- Analytics on recommendation effectiveness

---

## Scheduling

### Recommended Schedule

**Option 1: Weekly (Standard)**
```cron
# Every Monday at 6:00 AM
0 6 * * 1 cd /path/to/Concord-BI-Server && python3 scripts/weekly_recommendation_worker.py
```

**Option 2: Bi-Weekly (Low-Change Businesses)**
```cron
# Every other Monday at 6:00 AM
0 6 * * 1 [ $(expr $(date +\%W) \% 2) -eq 0 ] && cd /path/to/Concord-BI-Server && python3 scripts/weekly_recommendation_worker.py
```

**Option 3: Daily (High-Velocity E-commerce)**
```cron
# Every day at 6:00 AM
0 6 * * * cd /path/to/Concord-BI-Server && python3 scripts/weekly_recommendation_worker.py
```

**Option 4: Kubernetes CronJob**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-recommendation-worker
spec:
  schedule: "0 6 * * 1"  # Monday 6 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: worker
            image: concord-bi-server:latest
            command: ["python3", "scripts/weekly_recommendation_worker.py"]
            env:
            - name: REDIS_HOST
              value: "redis-service"
          restartPolicy: OnFailure
```

### Scheduling Considerations

**Time of Day:**
- **Early Morning (6:00 AM)**: Before business hours
- **Off-Peak**: Lower DB load, faster queries
- **Before API Traffic**: Cache warm when users arrive

**Frequency Trade-offs:**

| Frequency | Pros | Cons |
|-----------|------|------|
| **Daily** | Freshest recommendations, responds to recent orders | Higher compute cost, more DB load |
| **Weekly** | Lower cost, acceptable staleness, matches business cycle | Recommendations can be 6 days old |
| **Bi-Weekly** | Lowest cost, suitable for stable catalogs | Stale for dynamic inventory |

**Recommendation**: **Weekly** for most businesses, **Daily** for high-velocity e-commerce.

### Execution Time Estimates

**Expected Duration:**

| Customers | Workers | Time |
|-----------|---------|------|
| 1,000 | 4 | ~6 minutes |
| 5,000 | 4 | ~30 minutes |
| 10,000 | 4 | ~60 minutes |
| 5,000 | 8 | ~15 minutes |
| 10,000 | 8 | ~30 minutes |

**Formula**: `Time ≈ (Customers / Workers) / 2.5 clients/sec`

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `127.0.0.1` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `DB_HOST` | - | MS SQL Server hostname |
| `DB_PORT` | `1433` | MS SQL Server port |
| `DB_NAME` | - | Database name |
| `DB_USER` | - | Database username |
| `DB_PASSWORD` | - | Database password |

### CLI Arguments

```bash
python3 scripts/weekly_recommendation_worker.py [OPTIONS]
```

**Options:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dry-run` | flag | False | Test mode, doesn't store in Redis |
| `--limit N` | int | None | Process only first N clients |
| `--workers W` | int | 4 | Number of parallel workers |

**Examples:**

```bash
# Standard run
python3 scripts/weekly_recommendation_worker.py

# Test with first 100 customers
python3 scripts/weekly_recommendation_worker.py --limit 100

# Dry run (no Redis writes)
python3 scripts/weekly_recommendation_worker.py --dry-run --limit 10

# High-performance (8 workers)
python3 scripts/weekly_recommendation_worker.py --workers 8
```

### Performance Tuning

**Worker Count:**
```bash
# Standard server (4-8 cores, 100 DB connections)
--workers 4

# High-performance (16+ cores, 200+ DB connections)
--workers 8

# Low-resource (2 cores, 50 DB connections)
--workers 2
```

**Limiting Customer Scope:**
```bash
# Test with subset
--limit 100

# Process specific segment (modify SQL in code)
# WHERE c.ID >= 10000 AND c.ID < 20000
```

---

## Operational Guidelines

### Pre-Deployment Checklist

- [ ] Redis server running and accessible
- [ ] Database connection pool configured (min 10 connections)
- [ ] Environment variables set
- [ ] Test run with `--dry-run --limit 10`
- [ ] Verify recommendations generated correctly
- [ ] Schedule cron job or Kubernetes CronJob
- [ ] Set up monitoring and alerting

### Running the Worker

**Standard Execution:**
```bash
cd /path/to/Concord-BI-Server
python3 scripts/weekly_recommendation_worker.py
```

**Docker Execution:**
```bash
docker run --rm \
  -e REDIS_HOST=redis \
  -e DB_HOST=db \
  concord-bi-server:latest \
  python3 scripts/weekly_recommendation_worker.py
```

**Test Run:**
```bash
# Dry run with 10 customers
python3 scripts/weekly_recommendation_worker.py --dry-run --limit 10
```

### Expected Output

```
================================================================================
WEEKLY RECOMMENDATION WORKER
================================================================================
Week: 2025-W46
Recommendations per client: 25
Workers: 4
Dry run: False
================================================================================
INFO - Fetching active clients (orders in last 90 days)...
INFO - Found 5247 active clients

Processing 5247 clients with 4 workers...
INFO - Progress: 10/5247 (0.2%) | Success: 10 | Failed: 0 | Rate: 2.5 clients/sec | ETA: 35.0 min
INFO - Progress: 20/5247 (0.4%) | Success: 20 | Failed: 0 | Rate: 2.6 clients/sec | ETA: 33.5 min
...
INFO - Progress: 5247/5247 (100.0%) | Success: 5234 | Failed: 13 | Rate: 2.7 clients/sec | ETA: 0.0 min

================================================================================
JOB SUMMARY
================================================================================
Week: 2025-W46
Total clients: 5247
Processed: 5247
Successful: 5234
Failed: 13
Success rate: 99.8%
Total recommendations: 130850
Total discovery recommendations: 32715
Avg discovery per client: 6.2
Duration: 32.4 minutes
Rate: 2.70 clients/second
================================================================================
✅ JOB COMPLETED SUCCESSFULLY
```

---

## Monitoring & Troubleshooting

### Health Checks

**Verify Recommendations Generated:**
```bash
# Check Redis
redis-cli GET "2025-W46:12345"

# Check current week key
python3 -c "from scripts.redis_helper import WeeklyRecommendationCache; print(WeeklyRecommendationCache.get_week_key())"
```

**Count Cached Customers:**
```bash
redis-cli KEYS "2025-W46:*" | wc -l
```

### Common Issues

**Issue 1: No Recommendations Generated**
```
WARNING - No active clients found. Exiting.
```

**Solution:**
```sql
-- Verify active clients exist
SELECT COUNT(DISTINCT c.ID)
FROM dbo.Client c
INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
WHERE o.Created >= DATEADD(day, -90, GETDATE())
      AND c.IsActive = 1;
```

---

**Issue 2: High Failure Rate**
```
Failed: 500 (10.0%)
```

**Solution:**
```bash
# Check logs for common errors
grep "ERROR" /var/log/recommendation_worker.log

# Common causes:
# - DB connection pool exhausted
# - Redis connection timeout
# - Missing product data
```

---

**Issue 3: Slow Performance**
```
Duration: 120.0 minutes (expected: 30 min)
```

**Solution:**
```bash
# Increase workers
--workers 8

# Check DB query performance
# - Add indexes on Order.Created
# - Add indexes on ClientAgreement.ClientID

# Check Redis latency
redis-cli --latency
```

### Alerting

**Key Metrics:**

1. **Job Failed**: Exit code != 0
2. **High Failure Rate**: >5% customers failed
3. **Long Runtime**: >2x expected duration
4. **Low Success Count**: <95% of active customers
5. **Cache Miss**: Week key doesn't exist in Redis

---

## Summary

The Weekly Recommendation Worker provides:

- **Personalized recommendations** for 5,000+ active customers
- **25 products per customer** (mix of repeat + discovery)
- **Weekly cache warming** with 8-day TTL
- **Parallel processing** using 4 threads (2.5 customers/second)
- **Redis storage** with weekly key pattern
- **Database backup** for audit trail
- **Flexible scheduling** (weekly, daily, or on-demand)

**Next Steps:**
- Schedule weekly runs (Monday 6 AM)
- Configure monitoring and alerting
- Tune worker count based on infrastructure
- Integrate with recommendation API endpoints
