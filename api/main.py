#!/usr/bin/env python3

import os
import sys
import time
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v33 import ImprovedHybridRecommenderV33
from scripts.redis_helper import WeeklyRecommendationCache
from scripts.forecasting import ForecastEngine
from api.db_pool import get_connection, close_pool
from api.models.forecast_schemas import ProductForecastResponse, ForecastErrorResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')  # Use 127.0.0.1 instead of localhost
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
CACHE_TTL = int(os.getenv('CACHE_TTL', 604800))  # 7 days default

TARGET_P99_MS = 100
TARGET_P50_MS = 50

redis_client = None
weekly_cache = None

metrics = {
    'requests': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'errors': 0,
    'total_latency_ms': 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, weekly_cache

    logger.info("="*60)
    logger.info("Starting Production Recommendation API")
    logger.info("="*60)

    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2
        )
        redis_client.ping()
        logger.info(f"✓ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.warning(f"⚠ Redis not available: {e}. Running without cache.")
        redis_client = None

    try:
        weekly_cache = WeeklyRecommendationCache()
        logger.info(f"✓ Weekly recommendation cache initialized")
    except Exception as e:
        logger.warning(f"⚠ Weekly cache not available: {e}. Weekly endpoint will use fallback.")
        weekly_cache = None

    logger.info("✓ Database connection pool initialized (20 connections, max 30)")
    logger.info("✓ Improved Hybrid Recommender V3 (75.4% precision@50)")
    logger.info("✓ API ready to serve concurrent requests")
    logger.info("="*60)

    yield

    logger.info("Shutting down API...")
    if redis_client:
        redis_client.close()
    if weekly_cache:
        weekly_cache.close()
    close_pool()
    logger.info("✓ Database connection pool closed")

app = FastAPI(
    title="B2B Product Recommendation API",
    description="Enterprise-grade recommendation service with 75.4% precision@50 (Improved V3 + Connection Pooling)",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    customer_id: int = Field(..., description="Customer ID to generate recommendations for")
    top_n: int = Field(50, ge=1, le=500, description="Number of recommendations to return")
    as_of_date: Optional[str] = Field(None, description="ISO date for point-in-time recommendations (YYYY-MM-DD)")
    use_cache: bool = Field(True, description="Whether to use Redis cache")
    include_discovery: bool = Field(True, description="Whether to include collaborative filtering for new product discovery (V3.1)")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": 410376,
                "top_n": 50,
                "as_of_date": "2024-07-01",
                "use_cache": True,
                "include_discovery": True
            }
        }

class RecommendationResponse(BaseModel):
    customer_id: int
    recommendations: List[Dict[str, Any]]
    count: int
    discovery_count: int = Field(0, description="Number of recommendations that are NEW products (not previously purchased)")
    precision_estimate: float = Field(0.754, description="Expected precision@50 based on validation (Heavy: 89.2%, Regular: 88.2%, Light: 54.1%)")
    latency_ms: float
    cached: bool
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    metrics: Dict[str, Any]
    redis_connected: bool
    model_version: str = "improved_hybrid_v3_75.4pct_pooled"

class MetricsResponse(BaseModel):
    total_requests: int
    cache_hit_rate: float
    error_rate: float
    avg_latency_ms: float
    p99_target_ms: int
    p50_target_ms: int

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(process_time_ms, 2))
    return response

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "B2B Product Recommendation API",
        "version": "3.0.0",
        "status": "operational",
        "model_performance": "75.4% precision@50 (Improved V3 + Connection Pooling)",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0

    return HealthResponse(
        status="healthy",
        version="3.0.0",
        uptime_seconds=uptime,
        metrics={
            "requests": metrics['requests'],
            "cache_hit_rate": metrics['cache_hits'] / max(metrics['requests'], 1),
            "error_rate": metrics['errors'] / max(metrics['requests'], 1),
            "avg_latency_ms": metrics['total_latency_ms'] / max(metrics['requests'], 1)
        },
        redis_connected=redis_client is not None,
        model_version="improved_hybrid_v3_75.4pct_pooled"
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    total_requests = max(metrics['requests'], 1)

    return MetricsResponse(
        total_requests=metrics['requests'],
        cache_hit_rate=metrics['cache_hits'] / total_requests,
        error_rate=metrics['errors'] / total_requests,
        avg_latency_ms=metrics['total_latency_ms'] / total_requests,
        p99_target_ms=TARGET_P99_MS,
        p50_target_ms=TARGET_P50_MS
    )

# Unified cache key pattern: recommendations:customer:{customer_id}:{as_of_date}
CACHE_KEY_PREFIX = "recommendations:customer"


def get_cache_key(customer_id: int, as_of_date: Optional[str] = None) -> str:
    """Get unified cache key for a customer's recommendations."""
    if as_of_date is None:
        as_of_date = datetime.now().strftime('%Y-%m-%d')
    return f"{CACHE_KEY_PREFIX}:{customer_id}:{as_of_date}"


def get_from_cache(customer_id: int, as_of_date: Optional[str] = None) -> Optional[List[Dict]]:
    """Retrieve recommendations from cache."""
    if not redis_client:
        return None

    try:
        cache_key = get_cache_key(customer_id, as_of_date)
        cached = redis_client.get(cache_key)
        if cached:
            metrics['cache_hits'] += 1
            logger.debug(f"Cache HIT: {cache_key}")
            return json.loads(cached)
        else:
            metrics['cache_misses'] += 1
            logger.debug(f"Cache MISS: {cache_key}")
            return None
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def set_in_cache(customer_id: int, recommendations: List[Dict], as_of_date: Optional[str] = None, ttl: int = CACHE_TTL):
    """Store recommendations in cache."""
    if not redis_client:
        return

    try:
        cache_key = get_cache_key(customer_id, as_of_date)
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(recommendations)
        )
        logger.debug(f"Cached: {cache_key} (TTL: {ttl}s)")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


def fetch_client_analytics(conn, customer_id: int) -> Dict[str, Any]:
    """Fetch client analytics data for charts and proof metrics."""
    cursor = conn.cursor()
    result = {
        "client_name": None,
        "segment": "UNKNOWN",
        "charts": {
            "purchase_history": [],
            "top_categories": [],
            "recommendation_sources": []
        },
        "proof": {
            "total_orders": 0,
            "avg_order_value": 0,
            "last_order_date": None,
            "days_since_last_order": None,
            "loyalty_score": 0,
            "total_products_purchased": 0,
            "total_spent": 0,
            "model_confidence": 0.754
        }
    }

    try:
        # Get client name
        cursor.execute("""
            SELECT Name, FullName FROM dbo.Client WHERE ID = ? AND Deleted = 0
        """, (customer_id,))
        row = cursor.fetchone()
        if row:
            result["client_name"] = row[1] if row[1] else row[0]

        # Get agreement IDs for this customer
        cursor.execute("""
            SELECT ID FROM dbo.ClientAgreement WHERE ClientID = ? AND Deleted = 0
        """, (customer_id,))
        agreement_ids = [r[0] for r in cursor.fetchall()]

        if not agreement_ids:
            return result

        agreement_list = ','.join(str(a) for a in agreement_ids)

        # Purchase history (last 12 months for chart tabs)
        cursor.execute(f"""
            SELECT FORMAT(o.Created, 'yyyy-MM') as month,
                   COUNT(DISTINCT o.ID) as orders,
                   ISNULL(SUM(oi.Qty * oi.PricePerItem), 0) as amount
            FROM dbo.[Order] o
            JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.ClientAgreementID IN ({agreement_list})
              AND o.Created >= DATEADD(month, -12, GETDATE())
              AND o.Deleted = 0
            GROUP BY FORMAT(o.Created, 'yyyy-MM')
            ORDER BY month
        """)
        result["charts"]["purchase_history"] = [
            {"month": r[0], "orders": r[1], "amount": float(r[2])}
            for r in cursor.fetchall()
        ]

        # Customer proof stats
        cursor.execute(f"""
            SELECT
                COUNT(DISTINCT o.ID) as total_orders,
                ISNULL(AVG(order_totals.total), 0) as avg_order_value,
                MAX(o.Created) as last_order_date,
                DATEDIFF(day, MAX(o.Created), GETDATE()) as days_since_last,
                COUNT(DISTINCT oi.ProductID) as total_products,
                ISNULL(SUM(oi.Qty * oi.PricePerItem), 0) as total_spent
            FROM dbo.[Order] o
            JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            LEFT JOIN (
                SELECT o2.ID, SUM(oi2.Qty * oi2.PricePerItem) as total
                FROM dbo.[Order] o2
                JOIN dbo.OrderItem oi2 ON o2.ID = oi2.OrderID
                WHERE o2.ClientAgreementID IN ({agreement_list}) AND o2.Deleted = 0
                GROUP BY o2.ID
            ) order_totals ON o.ID = order_totals.ID
            WHERE o.ClientAgreementID IN ({agreement_list})
              AND o.Deleted = 0
        """)
        stats = cursor.fetchone()
        if stats:
            result["proof"]["total_orders"] = stats[0] or 0
            result["proof"]["avg_order_value"] = round(float(stats[1] or 0), 2)
            result["proof"]["last_order_date"] = stats[2].strftime('%Y-%m-%d') if stats[2] else None
            result["proof"]["days_since_last_order"] = stats[3]
            result["proof"]["total_products_purchased"] = stats[4] or 0
            result["proof"]["total_spent"] = round(float(stats[5] or 0), 2)

            # Calculate loyalty score (0-1) based on recency and frequency
            total_orders = stats[0] or 0
            days_since = stats[3] or 365
            recency_score = max(0, 1 - (days_since / 365))
            frequency_score = min(1, total_orders / 100)
            result["proof"]["loyalty_score"] = round((recency_score * 0.4 + frequency_score * 0.6), 2)

            # Determine segment
            if total_orders >= 500:
                result["segment"] = "HEAVY"
            elif total_orders >= 100:
                result["segment"] = "REGULAR"
            elif total_orders > 0:
                result["segment"] = "LIGHT"
            else:
                result["segment"] = "COLD_START"

    except Exception as e:
        logger.warning(f"Error fetching client analytics: {e}")

    return result



def fetch_product_analytics(conn, product_id: int) -> Dict[str, Any]:
    """Fetch product analytics data for charts and proof metrics."""
    cursor = conn.cursor()
    result = {
        "product_name": None,
        "vendor_code": None,
        "category": None,
        "charts": {
            "sales_history": [],
            "top_customers": [],
            "monthly_trend": []
        },
        "proof": {
            "total_orders": 0,
            "total_qty_sold": 0,
            "total_revenue": 0,
            "unique_customers": 0,
            "avg_order_qty": 0,
            "last_sale_date": None,
            "days_since_last_sale": None,
            "first_sale_date": None,
            "product_age_days": None
        }
    }

    try:
        # Get product info (simplified - without ProductGroup join that may fail)
        cursor.execute("""
            SELECT p.Name, p.VendorCode
            FROM dbo.Product p
            WHERE p.ID = ? AND p.Deleted = 0
        """, (product_id,))
        row = cursor.fetchone()
        if row:
            result["product_name"] = row[0]
            result["vendor_code"] = row[1]
            result["category"] = None  # ProductGroup join removed due to schema differences

        # Sales history (last 12 months)
        cursor.execute("""
            SELECT FORMAT(o.Created, 'yyyy-MM') as month,
                   COUNT(DISTINCT o.ID) as orders,
                   ISNULL(SUM(oi.Qty), 0) as qty,
                   ISNULL(SUM(oi.Qty * oi.PricePerItem), 0) as amount
            FROM dbo.[Order] o
            JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE oi.ProductID = ?
              AND o.Created >= DATEADD(month, -12, GETDATE())
              AND o.Deleted = 0
            GROUP BY FORMAT(o.Created, 'yyyy-MM')
            ORDER BY month
        """, (product_id,))
        result["charts"]["sales_history"] = [
            {"month": r[0], "orders": r[1], "qty": float(r[2]), "amount": float(r[3])}
            for r in cursor.fetchall()
        ]

        # Top customers for this product (last 12 months)
        cursor.execute("""
            SELECT TOP 10
                   c.ID as customer_id,
                   c.Name as customer_name,
                   SUM(oi.Qty) as total_qty,
                   COUNT(DISTINCT o.ID) as order_count,
                   SUM(oi.Qty * oi.PricePerItem) as total_amount
            FROM dbo.[Order] o
            JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
            JOIN dbo.Client c ON ca.ClientID = c.ID
            WHERE oi.ProductID = ?
              AND o.Created >= DATEADD(month, -12, GETDATE())
              AND o.Deleted = 0
            GROUP BY c.ID, c.Name
            ORDER BY total_qty DESC
        """, (product_id,))
        result["charts"]["top_customers"] = [
            {"customer_id": r[0], "customer_name": r[1], "total_qty": float(r[2]),
             "order_count": r[3], "total_amount": float(r[4])}
            for r in cursor.fetchall()
        ]

        # Product proof stats
        cursor.execute("""
            SELECT
                COUNT(DISTINCT o.ID) as total_orders,
                ISNULL(SUM(oi.Qty), 0) as total_qty,
                ISNULL(SUM(oi.Qty * oi.PricePerItem), 0) as total_revenue,
                COUNT(DISTINCT ca.ClientID) as unique_customers,
                ISNULL(AVG(oi.Qty), 0) as avg_order_qty,
                MAX(o.Created) as last_sale_date,
                MIN(o.Created) as first_sale_date,
                DATEDIFF(day, MAX(o.Created), GETDATE()) as days_since_last
            FROM dbo.[Order] o
            JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
            WHERE oi.ProductID = ?
              AND o.Deleted = 0
        """, (product_id,))
        stats = cursor.fetchone()
        if stats:
            result["proof"]["total_orders"] = stats[0] or 0
            result["proof"]["total_qty_sold"] = float(stats[1] or 0)
            result["proof"]["total_revenue"] = round(float(stats[2] or 0), 2)
            result["proof"]["unique_customers"] = stats[3] or 0
            result["proof"]["avg_order_qty"] = round(float(stats[4] or 0), 2)
            result["proof"]["last_sale_date"] = stats[5].strftime('%Y-%m-%d') if stats[5] else None
            result["proof"]["first_sale_date"] = stats[6].strftime('%Y-%m-%d') if stats[6] else None
            result["proof"]["days_since_last_sale"] = stats[7]
            if stats[6]:
                result["proof"]["product_age_days"] = (datetime.now() - stats[6]).days

    except Exception as e:
        logger.warning(f"Error fetching product analytics: {e}")

    return result

@app.get("/recommendations/{customer_id}", tags=["Recommendations"])
async def get_recommendations(customer_id: int):
    """Get recommendations for a customer with charts and proof metrics."""
    start_time = time.time()
    as_of_date = datetime.now().strftime('%Y-%m-%d')

    try:
        recommendations = None
        cached = False
        analytics = None

        # Try to get from cache first
        if weekly_cache:
            recommendations = weekly_cache.get_recommendations(customer_id, as_of_date)
            if recommendations:
                cached = True
                logger.debug(f"Cache HIT for customer {customer_id} (date {as_of_date})")
            else:
                logger.debug(f"Cache MISS for customer {customer_id} (date {as_of_date})")

        # Get connection for recommendations and analytics
        conn = get_connection()
        try:
            if recommendations is None:
                logger.info(f"Generating on-demand recommendations for customer {customer_id}")
                recommender = ImprovedHybridRecommenderV33(conn=conn, use_cache=True)
                recommendations = recommender.get_recommendations(
                    customer_id=customer_id,
                    as_of_date=as_of_date,
                    top_n=20,
                    include_discovery=True
                )

            # Fetch client analytics (charts + proof)
            analytics = fetch_client_analytics(conn, customer_id)
        finally:
            conn.close()

        if len(recommendations) > 20:
            recommendations = recommendations[:20]

        # Count recommendation sources for chart
        source_counts = {}
        for r in recommendations:
            source = r.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        analytics["charts"]["recommendation_sources"] = [
            {"source": k, "count": v} for k, v in source_counts.items()
        ]

        latency_ms = (time.time() - start_time) * 1000
        discovery_count = sum(1 for r in recommendations if r.get('source') in ['discovery', 'hybrid'])

        metrics['requests'] += 1
        metrics['total_latency_ms'] += latency_ms

        return {
            "customer_id": customer_id,
            "client_name": analytics.get("client_name"),
            "segment": analytics.get("segment"),
            "date": as_of_date,
            "recommendations": recommendations,
            "count": len(recommendations),
            "discovery_count": discovery_count,
            "charts": analytics.get("charts", {}),
            "proof": analytics.get("proof", {}),
            "cached": cached,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        metrics['errors'] += 1
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get(
    "/forecast/{product_id}",
    response_model=ProductForecastResponse,
    responses={404: {"model": ForecastErrorResponse}, 500: {"model": ForecastErrorResponse}},
    tags=["Forecasting"],
    summary="Get product demand forecast",
    description="Returns 1 week historical (Mon-Fri) + configurable forecast horizon (Mon-Fri business weeks)"
)
async def get_product_forecast(
    product_id: int,
    as_of_date: Optional[str] = Query(
        None,
        description="Reference date for forecast (ISO format YYYY-MM-DD). Default: today"
    ),
    forecast_weeks: int = Query(
        12,
        ge=4,
        le=26,
        description="Forecast horizon in weeks (Mon-Fri). Presets: 4 (short), 12 (default), 26 (long)."
    ),
    use_cache: bool = Query(
        True,
        description="Use Redis cache for performance"
    )
):
    """
    Get demand forecast for a specific product.

    Returns:
    - 1 week of historical actual data (Mon-Fri)
    - N weeks of predicted data (Mon-Fri, default 12)
    - Total: 1 + N weeks in weekly_data array

    Week boundaries are Mon-Fri (business days only).
    """
    start_time = time.time()

    try:

        conn = get_connection()

        try:

            engine = ForecastEngine(conn=conn, forecast_weeks=forecast_weeks)

            if use_cache and redis_client:
                forecast = engine.generate_forecast_cached(
                    product_id=product_id,
                    redis_client=redis_client,
                    as_of_date=as_of_date,
                    cache_ttl=CACHE_TTL
                )
            else:
                forecast = engine.generate_forecast(
                    product_id=product_id,
                    as_of_date=as_of_date
                )

            if forecast is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No predictable customers found for product {product_id}"
                )

            latency_ms = (time.time() - start_time) * 1000

            metrics['requests'] += 1
            metrics['total_latency_ms'] += latency_ms

            # Fetch product analytics
            analytics = fetch_product_analytics(conn, product_id)

            response_dict = {
                'product_id': forecast.product_id,
                'product_name': analytics.get('product_name') or forecast.product_name,
                'vendor_code': analytics.get('vendor_code'),
                'category': analytics.get('category'),
                'forecast_period_weeks': forecast.forecast_period_weeks,
                'historical_weeks': forecast.historical_weeks,
                'summary': forecast.summary,
                'weekly_data': forecast.weekly_data,  # Unified timeline (historical + forecast)
                'top_customers_by_volume': forecast.top_customers_by_volume,
                'at_risk_customers': forecast.at_risk_customers,
                'model_metadata': forecast.model_metadata,
                'charts': analytics.get('charts', {}),
                'proof': analytics.get('proof', {})
            }

            logger.info(
                f"Forecast generated for product {product_id}: "
                f"{forecast.summary['total_predicted_quantity']} units, "
                f"{latency_ms:.1f}ms"
            )

            return response_dict

        finally:

            conn.close()

    except HTTPException:
        metrics['errors'] += 1
        raise
    except Exception as e:
        metrics['errors'] += 1
        logger.error(f"Error generating forecast for product {product_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error generating forecast: {str(e)}"
        )

@app.delete("/cache/{customer_id}", tags=["Cache Management"])
async def clear_customer_cache(customer_id: int):
    """Clear all cached recommendations for a specific customer."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Use unified cache key pattern
        pattern = f"{CACHE_KEY_PREFIX}:{customer_id}:*"
        keys = list(redis_client.scan_iter(match=pattern))

        if keys:
            redis_client.delete(*keys)
            return {"status": "success", "deleted_keys": len(keys)}
        else:
            return {"status": "success", "deleted_keys": 0, "message": "No cached data found"}

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear-all", tags=["Cache Management"])
async def clear_all_cache():
    """Clear all cached recommendations."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Use unified cache key pattern
        pattern = f"{CACHE_KEY_PREFIX}:*"
        keys = list(redis_client.scan_iter(match=pattern))

        if keys:
            redis_client.delete(*keys)
            return {"status": "success", "deleted_keys": len(keys)}
        else:
            return {"status": "success", "deleted_keys": 0}

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
