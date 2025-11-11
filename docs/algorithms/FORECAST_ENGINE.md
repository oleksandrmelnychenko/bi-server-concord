# Forecast Engine - Production-Grade Orchestrator

## Overview

The **Forecast Engine** is the main orchestrator that coordinates the complete multi-layer forecasting pipeline. It acts as the central intelligence layer that connects pattern analysis, Bayesian prediction, and aggregation into a cohesive production-grade system.

**Location**: `/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts/forecasting/core/forecast_engine.py`

## Architecture Philosophy

The Forecast Engine implements a **pipeline orchestration pattern** with the following design principles:

1. **Layer Separation**: Each forecasting layer (pattern, prediction, aggregation) is encapsulated in dedicated components
2. **Connection Pooling**: Database connections are managed externally and passed in, enabling efficient resource usage
3. **Caching Strategy**: Redis-backed caching with configurable TTL for performance optimization
4. **Error Resilience**: Comprehensive error handling at each pipeline stage with fallback mechanisms
5. **Enrichment Pipeline**: Progressive data enrichment from raw predictions to business-ready insights

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FORECAST ENGINE PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   INPUT      │
                              │ Product ID   │
                              │ As-Of Date   │
                              └──────┬───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  STAGE 1: Customer Discovery   │
                    │  _get_product_customers()      │
                    │  Query: ClientAgreement +      │
                    │         Order + OrderItem      │
                    └────────────┬───────────────────┘
                                 │
                                 │ List[customer_id]
                                 ▼
                    ┌────────────────────────────────┐
                    │  STAGE 2: Pattern Analysis     │
                    │  FOR EACH customer:            │
                    │    pattern_analyzer.analyze()  │
                    │    → CustomerProductPattern    │
                    │  Filter: min_orders >= 2       │
                    └────────────┬───────────────────┘
                                 │
                                 │ List[Pattern]
                                 ▼
                    ┌────────────────────────────────┐
                    │  STAGE 3: Prediction           │
                    │  FOR EACH pattern:             │
                    │    predictor.predict_next()    │
                    │    → CustomerPrediction        │
                    │  Filter: confidence >= 0.3     │
                    └────────────┬───────────────────┘
                                 │
                                 │ List[Prediction]
                                 ▼
                    ┌────────────────────────────────┐
                    │  STAGE 4: Product Metadata     │
                    │  _get_product_info()           │
                    │  → {name, unit_price}          │
                    └────────────┬───────────────────┘
                                 │
                                 │ Product Info
                                 ▼
                    ┌────────────────────────────────┐
                    │  STAGE 5: Aggregation          │
                    │  aggregator.aggregate()        │
                    │  Group by week, sum quantities │
                    │  Calculate business metrics    │
                    │  → ProductForecast             │
                    └────────────┬───────────────────┘
                                 │
                                 │ ProductForecast (raw)
                                 ▼
                    ┌────────────────────────────────┐
                    │  STAGE 6: Customer Enrichment  │
                    │  _enrich_with_customer_names() │
                    │  Add customer names to:        │
                    │    - weekly_data               │
                    │    - top_customers             │
                    │    - at_risk_customers         │
                    └────────────┬───────────────────┘
                                 │
                                 ▼
                              ┌──────────────┐
                              │   OUTPUT     │
                              │ ProductForecast │
                              │ (enriched)   │
                              └──────────────┘
```

---

## Core Components

### 1. ForecastEngine Class

**Initialization**:
```python
def __init__(self, conn, forecast_weeks: int = 12):
    """
    Args:
        conn: pymssql database connection (from connection pool)
        forecast_weeks: Number of weeks to forecast (default 12 = ~3 months)

    Initializes:
        - PatternAnalyzer: Analyzes historical ordering patterns
        - CustomerPredictor: Predicts next order dates using Bayesian inference
        - ProductAggregator: Aggregates customer predictions into product-level forecast
    """
```

**Key Attributes**:
- `self.conn`: Database connection (managed by external pool)
- `self.forecast_weeks`: Forecast horizon (default: 12 weeks / 3 months)
- `self.pattern_analyzer`: Instance of PatternAnalyzer
- `self.predictor`: Instance of CustomerPredictor
- `self.aggregator`: Instance of ProductAggregator

---

## Main Pipeline Method

### `generate_forecast()`

**Signature**:
```python
def generate_forecast(
    product_id: int,
    as_of_date: Optional[str] = None,
    min_orders: int = 2,
    min_confidence: float = 0.3
) -> Optional[ProductForecast]
```

**Parameters**:
- `product_id`: Product ID to forecast
- `as_of_date`: Reference date in ISO format (YYYY-MM-DD), defaults to today
- `min_orders`: Minimum historical orders required for pattern analysis (default: 2)
- `min_confidence`: Minimum prediction confidence threshold (default: 0.3 = 30%)

**Returns**:
- `ProductForecast` object with complete forecast data
- `None` if no customers meet the prediction criteria

**Pipeline Stages**:

#### Stage 1: Customer Discovery
```python
customers = self._get_product_customers(product_id, as_of_date)
```
- Queries database for all customers who have ordered the product
- Joins: `ClientAgreement` → `Order` → `OrderItem`
- Filter: Orders created before `as_of_date`
- Returns: List of distinct customer IDs

**SQL Query**:
```sql
SELECT DISTINCT ca.ClientID
FROM dbo.ClientAgreement ca
INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
WHERE oi.ProductID = %s
      AND o.Created < %s
ORDER BY ca.ClientID
```

#### Stage 2: Pattern Analysis Loop
```python
for customer_id in customers:
    pattern = self.pattern_analyzer.analyze_customer_product(
        customer_id=customer_id,
        product_id=product_id,
        as_of_date=as_of_date
    )
```

**Process**:
1. For each customer, analyze historical ordering pattern
2. Calculate statistics: mean interval, std dev, CV, recency
3. Filter: Skip if `pattern.total_orders < min_orders`
4. Accumulate valid patterns for prediction

**Pattern Filtering Example**:
```python
if pattern.total_orders < min_orders:  # default: 2
    logger.debug(f"Customer {customer_id} has only {pattern.total_orders} orders")
    continue  # Skip this customer
```

#### Stage 3: Prediction Loop
```python
prediction = self.predictor.predict_next_order(
    pattern=pattern,
    as_of_date=as_of_date
)
```

**Process**:
1. Use Bayesian inference to predict next order date
2. Calculate prediction confidence based on pattern regularity
3. Estimate expected quantity based on historical average
4. Filter: Skip if `prediction.prediction_confidence < min_confidence`

**Confidence Filtering Example**:
```python
if prediction.prediction_confidence < min_confidence:  # default: 0.3
    logger.debug(f"Customer {customer_id} confidence too low")
    continue  # Skip this prediction
```

#### Stage 4: Product Metadata Retrieval
```python
product_info = self._get_product_info(product_id)
```

**Process**:
1. Fetch product name from `Product` table
2. Get most recent unit price from `OrderItem`
3. Fallback: Use default price ($35.00) if no data found

**SQL Query**:
```sql
SELECT TOP 1
    p.Name as product_name,
    oi.PricePerItem as unit_price
FROM dbo.Product p
LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID
WHERE p.ID = %s
      AND oi.PricePerItem IS NOT NULL
ORDER BY oi.Created DESC
```

**Return Structure**:
```python
{
    'product_name': 'Organic Honey 500g',
    'unit_price': 35.0
}
```

#### Stage 5: Aggregation
```python
forecast = self.aggregator.aggregate_forecast(
    product_id=product_id,
    predictions=predictions,
    as_of_date=as_of_date,
    conn=self.conn,
    product_name=product_info.get('product_name'),
    unit_price=product_info.get('unit_price')
)
```

**Process**:
1. Group predictions into weekly buckets
2. Calculate per-week totals: orders, quantities, revenue
3. Compute summary statistics: total demand, active customers
4. Identify top customers by volume
5. Flag at-risk customers (low confidence)

#### Stage 6: Customer Enrichment
```python
forecast = self._enrich_with_customer_names(forecast)
```

**Process**:
1. Collect all customer IDs from forecast data structures
2. Batch query customer names from database
3. Enrich three data structures:
   - `weekly_data.expected_customers[]`
   - `top_customers_by_volume[]`
   - `at_risk_customers[]`

**Enrichment Query**:
```sql
SELECT
    ID as customer_id,
    Name as customer_name
FROM dbo.Client
WHERE ID IN (123, 456, 789, ...)
```

---

## Connection Pooling Strategy

### Design Philosophy

The Forecast Engine **does not manage its own database connections**. Instead:

1. **External Pool Management**: Connections are created and managed by a connection pool (e.g., `DBUtils.PooledDB`)
2. **Dependency Injection**: Connection is passed into `__init__(self, conn, ...)`
3. **Cursor Management**: Engine creates/closes cursors but doesn't manage connection lifecycle
4. **Thread Safety**: Multiple engine instances can use connections from the same pool

### Connection Usage Pattern

```python
# GOOD: Connection pool pattern
from DBUtils.PooledDB import PooledDB

# Create connection pool (application startup)
pool = PooledDB(
    creator=pymssql,
    maxconnections=10,
    host='server',
    database='db',
    user='user',
    password='pass'
)

# Get connection from pool
conn = pool.connection()

# Create engine with pooled connection
engine = ForecastEngine(conn, forecast_weeks=12)

# Use engine
forecast = engine.generate_forecast(product_id=123)

# Return connection to pool
conn.close()
```

### Cursor Lifecycle

The engine manages cursors explicitly:

```python
# Pattern used throughout engine
cursor = self.conn.cursor()  # or cursor(as_dict=True)
cursor.execute(query, params)
results = cursor.fetchall()
cursor.close()  # Always close cursor
```

**Why This Matters**:
- Prevents cursor leaks and memory issues
- Ensures database resources are freed promptly
- Allows connection reuse across multiple forecast generations

---

## Redis Caching Mechanism

### Cache-Enabled Method

```python
def generate_forecast_cached(
    product_id: int,
    redis_client,
    as_of_date: Optional[str] = None,
    cache_ttl: int = 3600
) -> Optional[ProductForecast]
```

### Caching Strategy

**Cache Key Format**:
```
forecast:product:{product_id}:{as_of_date}
```

**Examples**:
- `forecast:product:123:2025-11-11`
- `forecast:product:456:2025-11-10`

**Cache Flow**:

```
┌──────────────────────────────────────────────────────────┐
│              REDIS CACHING FLOW                          │
└──────────────────────────────────────────────────────────┘

    ┌─────────────────────────┐
    │ generate_forecast_cached│
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Generate Cache Key      │
    │ forecast:product:123:   │
    │   2025-11-11            │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ redis_client.get(key)   │
    └───────────┬─────────────┘
                │
        ┌───────┴────────┐
        │                │
    CACHE HIT       CACHE MISS
        │                │
        ▼                ▼
    ┌────────┐      ┌──────────────────┐
    │ Deserialize │  │ generate_forecast│
    │ JSON → Obj  │  │  (full pipeline) │
    └────┬───┘      └────────┬─────────┘
         │                   │
         │                   ▼
         │         ┌──────────────────┐
         │         │ Serialize Obj →  │
         │         │ JSON             │
         │         └────────┬─────────┘
         │                  │
         │                  ▼
         │         ┌──────────────────┐
         │         │ redis_client.    │
         │         │ setex(key, ttl,  │
         │         │       json)      │
         │         └────────┬─────────┘
         │                  │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Return Forecast  │
         └──────────────────┘
```

### Serialization Methods

**To JSON**:
```python
def _forecast_to_dict(self, forecast: ProductForecast) -> Dict:
    return {
        'product_id': forecast.product_id,
        'forecast_period_weeks': forecast.forecast_period_weeks,
        'summary': forecast.summary,
        'weekly_forecasts': forecast.weekly_forecasts,
        'top_customers_by_volume': forecast.top_customers_by_volume,
        'at_risk_customers': forecast.at_risk_customers,
        'model_metadata': forecast.model_metadata
    }
```

**From JSON**:
```python
def _dict_to_forecast(self, data: Dict) -> ProductForecast:
    return ProductForecast(
        product_id=data['product_id'],
        forecast_period_weeks=data['forecast_period_weeks'],
        summary=data['summary'],
        weekly_forecasts=data['weekly_forecasts'],
        top_customers_by_volume=data['top_customers_by_volume'],
        at_risk_customers=data['at_risk_customers'],
        model_metadata=data['model_metadata']
    )
```

### Cache TTL Strategy

**Default**: 3600 seconds (1 hour)

**Recommended TTL Values**:
- **Ad-hoc requests**: 3600s (1 hour) - User-initiated forecasts
- **Nightly batch**: 86400s (24 hours) - Daily forecast generation
- **Real-time dashboard**: 1800s (30 minutes) - Frequently updated views
- **Historical analysis**: 604800s (7 days) - Static historical forecasts

### Cache Invalidation

**Current Strategy**: Time-based expiration (TTL)

**Future Enhancements**:
- Event-based invalidation when new orders are placed
- Cascade invalidation when customer patterns change significantly
- Version tagging for algorithm updates

---

## Error Handling

### Multi-Layer Error Resilience

#### 1. Top-Level Pipeline Protection

```python
try:
    # Full pipeline execution
    forecast = self.generate_forecast(...)
except Exception as e:
    logger.error(f"Error generating forecast for product {product_id}: {e}")
    raise  # Re-raise for caller to handle
```

**Strategy**: Catch all exceptions, log with context, re-raise for upstream handling

#### 2. Customer-Level Fault Isolation

```python
for customer_id in customers:
    pattern = self.pattern_analyzer.analyze_customer_product(...)
    if pattern is None:
        continue  # Skip this customer, continue with others
```

**Strategy**: If one customer's pattern analysis fails, don't fail entire forecast

#### 3. Cache Error Tolerance

```python
try:
    cached = redis_client.get(cache_key)
    # ... use cached data
except Exception as e:
    logger.warning(f"Cache read error: {e}")
    # Continue without cache - generate fresh forecast
```

**Strategy**: Cache failures are logged but don't block forecast generation

```python
try:
    redis_client.setex(cache_key, cache_ttl, json.dumps(forecast_dict))
except Exception as e:
    logger.warning(f"Cache write error: {e}")
    # Continue - forecast is still returned to caller
```

#### 4. Graceful Data Fallbacks

**Missing Product Info**:
```python
if row:
    return {
        'product_name': row['product_name'],
        'unit_price': float(row['unit_price'])
    }
else:
    return {
        'product_name': f"Product {product_id}",  # Fallback
        'unit_price': 35.0  # Default price
    }
```

**Missing Customer Names**:
```python
cust['customer_name'] = customer_names.get(
    cust_id,
    f"Customer {cust_id}"  # Fallback if name not found
)
```

#### 5. Empty Result Handling

```python
if not customers:
    logger.warning(f"No customers found for product {product_id}")
    return None  # Explicit None return, not exception

if not predictions:
    logger.warning(f"No predictable customers for product {product_id}")
    return None  # Explicit None return
```

**Strategy**: Distinguish between errors (raise exception) and empty results (return None)

---

## Customer Enrichment Process

### Purpose

Transform raw numerical data into business-ready insights by adding human-readable customer names.

### Data Structures Enriched

#### 1. Weekly Data - Expected Customers

**Before Enrichment**:
```python
{
    'week_number': 1,
    'expected_customers': [
        {
            'customer_id': 123,
            'predicted_quantity': 100,
            'confidence': 0.85
        }
    ]
}
```

**After Enrichment**:
```python
{
    'week_number': 1,
    'expected_customers': [
        {
            'customer_id': 123,
            'customer_name': 'ABC Restaurant',  # ← Added
            'predicted_quantity': 100,
            'confidence': 0.85
        }
    ]
}
```

#### 2. Top Customers by Volume

**Before Enrichment**:
```python
[
    {
        'customer_id': 456,
        'total_quantity': 500,
        'total_orders': 10
    }
]
```

**After Enrichment**:
```python
[
    {
        'customer_id': 456,
        'customer_name': 'XYZ Cafe',  # ← Added
        'total_quantity': 500,
        'total_orders': 10
    }
]
```

#### 3. At-Risk Customers

**Before Enrichment**:
```python
[
    {
        'customer_id': 789,
        'avg_confidence': 0.35,
        'risk_reason': 'Irregular ordering pattern'
    }
]
```

**After Enrichment**:
```python
[
    {
        'customer_id': 789,
        'customer_name': 'LMN Bakery',  # ← Added
        'avg_confidence': 0.35,
        'risk_reason': 'Irregular ordering pattern'
    }
]
```

### Enrichment Algorithm

**Step 1**: Collect all unique customer IDs
```python
customer_ids = set()

# From weekly data
for week in forecast.weekly_data:
    for cust in week.get('expected_customers', []):
        customer_ids.add(cust['customer_id'])

# From top customers
for cust in forecast.top_customers_by_volume:
    customer_ids.add(cust['customer_id'])

# From at-risk customers
for cust in forecast.at_risk_customers:
    customer_ids.add(cust['customer_id'])
```

**Step 2**: Batch query customer names (single database query)
```python
customer_names = self._get_customer_names(list(customer_ids))
# Returns: {123: 'ABC Restaurant', 456: 'XYZ Cafe', ...}
```

**Step 3**: Enrich each data structure in place
```python
# Enrich weekly data
for week in forecast.weekly_data:
    for cust in week.get('expected_customers', []):
        cust['customer_name'] = customer_names.get(
            cust['customer_id'],
            f"Customer {cust['customer_id']}"  # Fallback
        )

# Enrich top customers
for cust in forecast.top_customers_by_volume:
    cust['customer_name'] = customer_names.get(...)

# Enrich at-risk customers
for cust in forecast.at_risk_customers:
    cust['customer_name'] = customer_names.get(...)
```

### Batch Query Optimization

**Parameterized IN Clause**:
```python
placeholders = ','.join(['%s'] * len(customer_ids))
query = f"""
    SELECT ID as customer_id, Name as customer_name
    FROM dbo.Client
    WHERE ID IN ({placeholders})
"""
cursor.execute(query, customer_ids)
```

**Why This Matters**:
- **Single Query**: Fetch all names at once instead of N separate queries
- **Performance**: Dramatically reduces database round-trips
- **Example**: 50 customers = 1 query instead of 50 queries

---

## Input/Output Examples

### Example 1: Successful Forecast

**Input**:
```python
engine = ForecastEngine(conn, forecast_weeks=12)
forecast = engine.generate_forecast(
    product_id=123,
    as_of_date='2025-11-11',
    min_orders=2,
    min_confidence=0.3
)
```

**Pipeline Execution Log**:
```
INFO: ForecastEngine initialized for 12 weeks
INFO: Generating forecast for product 123 as of 2025-11-11
INFO: Found 45 customers for product 123
INFO: Pattern analysis: 42/45 customers, Predictions: 28 (confidence >= 0.3)
INFO: Forecast complete for product 123: 1250 units, 28 orders
```

**Output** (ProductForecast object):
```python
ProductForecast(
    product_id=123,
    forecast_period_weeks=12,

    summary={
        'total_predicted_quantity': 1250,
        'total_predicted_orders': 28,
        'total_predicted_revenue': 43750.0,  # 1250 * $35
        'avg_confidence': 0.67,
        'unique_customers': 28,
        'forecast_start_date': '2025-11-11',
        'forecast_end_date': '2026-02-02'
    },

    weekly_forecasts=[
        {
            'week_number': 1,
            'week_start_date': '2025-11-11',
            'week_end_date': '2025-11-17',
            'predicted_quantity': 120,
            'predicted_orders': 3,
            'predicted_revenue': 4200.0,
            'avg_confidence': 0.75
        },
        # ... weeks 2-12
    ],

    top_customers_by_volume=[
        {
            'customer_id': 456,
            'customer_name': 'ABC Restaurant',
            'total_quantity': 300,
            'total_orders': 6,
            'avg_confidence': 0.85
        },
        # ... more customers
    ],

    at_risk_customers=[
        {
            'customer_id': 789,
            'customer_name': 'XYZ Cafe',
            'avg_confidence': 0.32,
            'risk_reason': 'Low prediction confidence'
        }
    ],

    model_metadata={
        'as_of_date': '2025-11-11',
        'generated_at': '2025-11-11T10:30:00Z',
        'model_version': 'v1.0',
        'min_orders_threshold': 2,
        'min_confidence_threshold': 0.3
    }
)
```

### Example 2: No Predictable Customers

**Input**:
```python
forecast = engine.generate_forecast(
    product_id=999,  # Rarely ordered product
    as_of_date='2025-11-11',
    min_orders=3,  # Higher threshold
    min_confidence=0.5  # Higher confidence threshold
)
```

**Pipeline Execution Log**:
```
INFO: Generating forecast for product 999 as of 2025-11-11
INFO: Found 5 customers for product 999
INFO: Pattern analysis: 3/5 customers, Predictions: 0 (confidence >= 0.5)
WARNING: No predictable customers for product 999
```

**Output**:
```python
None  # No forecast generated
```

### Example 3: Cached Forecast

**First Call** (cache miss):
```python
engine = ForecastEngine(conn, forecast_weeks=12)
forecast = engine.generate_forecast_cached(
    product_id=123,
    redis_client=redis_client,
    as_of_date='2025-11-11',
    cache_ttl=3600
)
```

**Log**:
```
INFO: Forecast cache MISS for product 123
INFO: Generating forecast for product 123 as of 2025-11-11
INFO: Found 45 customers for product 123
INFO: Pattern analysis: 42/45 customers, Predictions: 28
INFO: Forecast cached for product 123 (TTL: 3600s)
```

**Second Call** (cache hit):
```python
forecast = engine.generate_forecast_cached(
    product_id=123,
    redis_client=redis_client,
    as_of_date='2025-11-11'
)
```

**Log**:
```
INFO: Forecast cache HIT for product 123
```

**Performance Impact**:
- First call: ~2-5 seconds (full pipeline)
- Second call: ~50-100ms (Redis retrieval + JSON deserialization)
- **Speedup**: 20-50x faster

---

## Performance Characteristics

### Computational Complexity

**Per-Product Forecast**:
- Customer discovery: O(1) - Single database query
- Pattern analysis: O(N) - N = number of customers
- Prediction: O(N) - Linear with customers
- Aggregation: O(N) - Group and sum
- Enrichment: O(1) - Single batch query for names

**Overall**: O(N) where N = number of customers who ordered the product

### Database Query Count (Without Cache)

**Minimum Queries**: 3
1. Customer discovery query
2. Product info query
3. Customer names batch query

**Plus**: N pattern queries (one per customer)
- Total: 3 + N queries

**With Pattern Analyzer Caching**: Reduces to 3 queries if patterns are cached

### Memory Usage

**Per Customer**:
- Pattern: ~500 bytes (statistics + metadata)
- Prediction: ~300 bytes (date + confidence + quantity)

**Example**: 100 customers × 800 bytes = ~80 KB per forecast

**ProductForecast Object**: ~10-20 KB (weekly data + customer lists)

### Time Estimates

**Typical Product (50 customers)**:
- Without cache: 1-3 seconds
- With cache (hit): 50-100 ms

**Large Product (500 customers)**:
- Without cache: 10-20 seconds
- With cache (hit): 100-200 ms

**Batch Processing (1000 products)**:
- Without cache: 30-60 minutes
- With aggressive caching: 5-10 minutes

---

## Integration Patterns

### Pattern 1: Single Product Forecast (API Endpoint)

```python
from contextlib import closing
from DBUtils.PooledDB import PooledDB
import redis

# Initialize connection pool (once at startup)
db_pool = PooledDB(creator=pymssql, ...)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# API endpoint handler
def get_product_forecast(product_id: int):
    with closing(db_pool.connection()) as conn:
        engine = ForecastEngine(conn, forecast_weeks=12)
        forecast = engine.generate_forecast_cached(
            product_id=product_id,
            redis_client=redis_client,
            cache_ttl=3600  # 1 hour
        )

        if forecast is None:
            return {"error": "No predictable customers"}, 404

        return forecast.to_dict(), 200
```

### Pattern 2: Batch Forecast Generation (Nightly Job)

```python
def generate_all_forecasts(product_ids: List[int]):
    results = []

    with closing(db_pool.connection()) as conn:
        engine = ForecastEngine(conn, forecast_weeks=12)

        for product_id in product_ids:
            try:
                forecast = engine.generate_forecast_cached(
                    product_id=product_id,
                    redis_client=redis_client,
                    cache_ttl=86400  # 24 hours
                )

                if forecast:
                    results.append({
                        'product_id': product_id,
                        'status': 'success',
                        'total_quantity': forecast.summary['total_predicted_quantity']
                    })
                else:
                    results.append({
                        'product_id': product_id,
                        'status': 'no_predictions'
                    })

            except Exception as e:
                logger.error(f"Failed to forecast product {product_id}: {e}")
                results.append({
                    'product_id': product_id,
                    'status': 'error',
                    'error': str(e)
                })

    return results
```

### Pattern 3: Historical Backtest

```python
def backtest_forecast(product_id: int, as_of_dates: List[str]):
    """Generate forecasts for multiple historical dates"""
    backtests = []

    with closing(db_pool.connection()) as conn:
        engine = ForecastEngine(conn, forecast_weeks=12)

        for as_of_date in as_of_dates:
            forecast = engine.generate_forecast(
                product_id=product_id,
                as_of_date=as_of_date,
                min_orders=2,
                min_confidence=0.3
            )

            if forecast:
                backtests.append({
                    'as_of_date': as_of_date,
                    'predicted_quantity': forecast.summary['total_predicted_quantity'],
                    'predicted_orders': forecast.summary['total_predicted_orders']
                })

    return backtests
```

---

## Configuration Best Practices

### Forecast Horizon Selection

**Short-term (4-8 weeks)**:
- Use case: Immediate inventory planning
- Higher confidence predictions
- Better for products with regular patterns

**Medium-term (12-16 weeks)**:
- Use case: Quarterly planning
- Balance between accuracy and planning horizon
- Default recommendation

**Long-term (24-52 weeks)**:
- Use case: Strategic planning
- Lower confidence, higher uncertainty
- Best for stable, predictable products

### Filtering Thresholds

**min_orders**:
- `1`: Very lenient, includes new customers (high noise)
- `2`: Default, requires repeat behavior (balanced)
- `3-5`: Conservative, established patterns only (high confidence)

**min_confidence**:
- `0.2`: Very lenient (includes irregular customers)
- `0.3`: Default, reasonable balance
- `0.5`: Conservative (only regular customers)
- `0.7+`: Very conservative (highly regular only)

### Cache TTL Strategy

**Production Recommendations**:
```python
# Real-time API
cache_ttl = 1800  # 30 minutes

# Dashboard
cache_ttl = 3600  # 1 hour

# Nightly batch
cache_ttl = 86400  # 24 hours

# Historical analysis
cache_ttl = 604800  # 7 days
```

---

## Logging and Monitoring

### Key Metrics to Track

1. **Pipeline Metrics**:
   - Total customers discovered
   - Patterns analyzed vs. total customers
   - Predictions made vs. patterns analyzed
   - Average confidence score

2. **Performance Metrics**:
   - Forecast generation time
   - Cache hit rate
   - Database query count
   - Memory usage per forecast

3. **Business Metrics**:
   - Total predicted quantity
   - Total predicted orders
   - Number of at-risk customers
   - Revenue forecast

### Log Levels Used

**INFO**: Normal pipeline progression
```python
logger.info(f"Generating forecast for product {product_id}")
logger.info(f"Found {len(customers)} customers")
logger.info(f"Forecast complete: {total_quantity} units")
```

**WARNING**: Recoverable issues
```python
logger.warning(f"No customers found for product {product_id}")
logger.warning(f"Cache read error: {e}")
```

**DEBUG**: Detailed filtering decisions
```python
logger.debug(f"Customer {customer_id} confidence too low")
logger.debug(f"Skipping customer with {total_orders} orders")
```

**ERROR**: Unrecoverable errors
```python
logger.error(f"Error generating forecast: {e}")
```

---

## Future Enhancements

### 1. Parallel Processing
- Use `concurrent.futures` to analyze customers in parallel
- Potential 3-5x speedup for products with many customers

### 2. Advanced Caching
- Multi-level cache (L1: in-memory, L2: Redis)
- Partial cache invalidation (customer-specific)
- Cache warming strategies

### 3. Dynamic Threshold Optimization
- Auto-tune `min_orders` and `min_confidence` per product
- Machine learning-based threshold selection

### 4. Forecast Ensembles
- Combine multiple prediction algorithms
- Weighted averaging based on historical accuracy

### 5. What-If Analysis
- Scenario planning (e.g., "What if demand increases 20%?")
- Sensitivity analysis on confidence thresholds

---

## Summary

The Forecast Engine is the production-grade orchestrator that:

1. **Coordinates** three specialized forecasting components into a cohesive pipeline
2. **Manages** database connections efficiently through external pooling
3. **Optimizes** performance with Redis caching and batch queries
4. **Handles** errors gracefully at multiple levels with clear fallbacks
5. **Enriches** raw predictions with business-ready metadata
6. **Scales** to handle thousands of products with configurable thresholds

It transforms individual customer ordering patterns into actionable product-level demand forecasts, enabling data-driven inventory management and business planning.
