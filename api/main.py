#!/usr/bin/env python3
"""
Production Recommendation API - Enterprise Grade (Improved V3 + Connection Pooling)

FastAPI service for B2B product recommendations with segment-specific strategies
- 75.4% precision@50 overall (Heavy: 89.2%, Regular: 88.2%, Light: 54.1%)
- Connection pooling for concurrent request support (20+ concurrent users)
- <400ms P99 latency under concurrent load
- Redis caching for hot customers
- Comprehensive monitoring & logging

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

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

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v32 import ImprovedHybridRecommenderV32
from scripts.redis_helper import WeeklyRecommendationCache
from scripts.forecasting import ForecastEngine
from api.db_pool import get_connection, close_pool
from api.models.forecast_schemas import ProductForecastResponse, ForecastErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')  # Use 127.0.0.1 instead of localhost
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # 1 hour default

# Performance targets
TARGET_P99_MS = 100
TARGET_P50_MS = 50

# Global state
redis_client = None
weekly_cache = None  # WeeklyRecommendationCache for pre-computed weekly recommendations
# Connection pool is managed by db_pool module (no global recommender needed)
metrics = {
    'requests': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'errors': 0,
    'total_latency_ms': 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global redis_client, weekly_cache

    # Startup
    logger.info("="*60)
    logger.info("Starting Production Recommendation API")
    logger.info("="*60)

    # Initialize Redis
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

    # Initialize Weekly Recommendation Cache
    try:
        weekly_cache = WeeklyRecommendationCache()
        logger.info(f"✓ Weekly recommendation cache initialized")
    except Exception as e:
        logger.warning(f"⚠ Weekly cache not available: {e}. Weekly endpoint will use fallback.")
        weekly_cache = None

    # Connection pool is initialized in db_pool module
    logger.info("✓ Database connection pool initialized (20 connections, max 30)")
    logger.info("✓ Improved Hybrid Recommender V3 (75.4% precision@50)")
    logger.info("✓ API ready to serve concurrent requests")
    logger.info("="*60)

    yield

    # Shutdown
    logger.info("Shutting down API...")
    if redis_client:
        redis_client.close()
    if weekly_cache:
        weekly_cache.close()
    close_pool()
    logger.info("✓ Database connection pool closed")


# Initialize FastAPI
app = FastAPI(
    title="B2B Product Recommendation API",
    description="Enterprise-grade recommendation service with 75.4% precision@50 (Improved V3 + Connection Pooling)",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
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


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Track request latency"""
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(process_time_ms, 2))
    return response


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """API root - basic info"""
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
    """Health check endpoint"""
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
    """Detailed metrics endpoint"""
    total_requests = max(metrics['requests'], 1)

    return MetricsResponse(
        total_requests=metrics['requests'],
        cache_hit_rate=metrics['cache_hits'] / total_requests,
        error_rate=metrics['errors'] / total_requests,
        avg_latency_ms=metrics['total_latency_ms'] / total_requests,
        p99_target_ms=TARGET_P99_MS,
        p50_target_ms=TARGET_P50_MS
    )


def get_cache_key(customer_id: int, top_n: int, as_of_date: Optional[datetime]) -> str:
    """Generate Redis cache key for ad-hoc requests"""
    date_str = as_of_date.strftime('%Y-%m-%d') if as_of_date else 'latest'
    return f"recs:v1:{customer_id}:{top_n}:{date_str}"


def get_worker_cache_key(customer_id: int, as_of_date: str) -> str:
    """
    Generate Redis cache key for weekly worker pre-computed recommendations.
    Must match the format used in weekly_recommendation_worker.py

    Format: recommendations:customer:{customer_id}:{as_of_date}
    Example: recommendations:customer:410376:2024-07-01
    """
    return f"recommendations:customer:{customer_id}:{as_of_date}"


def get_from_worker_cache(customer_id: int, as_of_date: str) -> Optional[Dict[str, Any]]:
    """
    Get pre-computed recommendations from weekly worker cache.
    Returns full result dict with metadata, not just recommendations list.
    """
    if not redis_client:
        return None

    try:
        cache_key = get_worker_cache_key(customer_id, as_of_date)
        cached = redis_client.get(cache_key)
        if cached:
            metrics['cache_hits'] += 1
            logger.debug(f"Worker Cache HIT: {cache_key}")
            return json.loads(cached)
        else:
            logger.debug(f"Worker Cache MISS: {cache_key}")
            return None
    except Exception as e:
        logger.warning(f"Worker cache read error: {e}")
        return None


def get_from_cache(cache_key: str) -> Optional[List[Dict]]:
    """Get recommendations from Redis cache"""
    if not redis_client:
        return None

    try:
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


def set_in_cache(cache_key: str, recommendations: List[Dict], ttl: int = CACHE_TTL):
    """Store recommendations in Redis cache"""
    if not redis_client:
        return

    try:
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(recommendations)
        )
        logger.debug(f"Cached: {cache_key} (TTL: {ttl}s)")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Generate product recommendations for a customer (V3.1 with discovery)

    Returns top-N product recommendations with expected 75.4% precision@50
    - Heavy users (500+ orders): 89.2% precision
    - Regular users (100-500 orders): 88.2% precision
    - Light users (<100 orders): 54.1% precision

    NEW in V3.1: Collaborative filtering for new product discovery
    - Recommends BOTH repurchase products AND new products customer hasn't bought
    - Finds similar customers using Jaccard similarity
    - Segment-specific blending (Heavy: 80% repurchase, Light: 60% discovery)

    Performance: <2s latency (acceptable for quality improvement)
    """
    start_time = time.time()
    metrics['requests'] += 1

    try:
        # Parse as_of_date
        as_of_date = None
        if request.as_of_date:
            try:
                as_of_date = datetime.fromisoformat(request.as_of_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid date format: {request.as_of_date}. Use YYYY-MM-DD"
                )

        # Check caches (worker cache first, then ad-hoc cache)
        cached = False
        from_worker = False
        recommendations = None

        # Convert datetime to string for cache key
        as_of_str = as_of_date.strftime('%Y-%m-%d') if as_of_date else datetime.now().strftime('%Y-%m-%d')

        if request.use_cache:
            # 1. Try worker cache first (pre-computed weekly recommendations)
            worker_result = get_from_worker_cache(request.customer_id, as_of_str)
            if worker_result:
                # Worker cache stores full result dict, extract recommendations
                recommendations = worker_result.get('recommendations', [])
                # If top_n requested is different, truncate
                if len(recommendations) > request.top_n:
                    recommendations = recommendations[:request.top_n]
                cached = True
                from_worker = True
                logger.info(f"✅ Served from worker cache: customer {request.customer_id}")

            # 2. If not in worker cache, try ad-hoc cache
            if recommendations is None:
                cache_key = get_cache_key(request.customer_id, request.top_n, as_of_date)
                recommendations = get_from_cache(cache_key)
                if recommendations:
                    cached = True
                    logger.debug(f"Served from ad-hoc cache: customer {request.customer_id}")

        # Generate recommendations if not cached
        if recommendations is None:
            # Get connection from pool (thread-safe, no lock needed)
            conn = get_connection()
            try:
                # Create recommender instance with pooled connection
                recommender = ImprovedHybridRecommenderV32(conn=conn, use_cache=request.use_cache)

                # as_of_str already defined above for cache keys
                recommendations = recommender.get_recommendations(
                    customer_id=request.customer_id,
                    as_of_date=as_of_str,
                    top_n=request.top_n,
                    include_discovery=request.include_discovery
                )
            finally:
                # Return connection to pool
                conn.close()

            # Cache results
            if request.use_cache:
                set_in_cache(cache_key, recommendations)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        metrics['total_latency_ms'] += latency_ms

        # Count discovery recommendations
        discovery_count = sum(1 for r in recommendations if r.get('source') in ['discovery', 'hybrid'])

        # Log slow requests
        if latency_ms > TARGET_P99_MS:
            logger.warning(f"Slow request: {latency_ms:.2f}ms (customer: {request.customer_id})")

        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=recommendations,
            count=len(recommendations),
            discovery_count=discovery_count,
            precision_estimate=0.754,  # From Improved V3 validation (50-customer test)
            latency_ms=round(latency_ms, 2),
            cached=cached,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        metrics['errors'] += 1
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/weekly-recommendations/{customer_id}", tags=["Recommendations"])
async def get_weekly_recommendations(customer_id: int):
    """
    Get pre-computed weekly recommendations for a customer (fast retrieval from Redis).

    This endpoint returns 25 product recommendations that were pre-computed by the
    weekly recommendation worker job. Recommendations are unique per week and include
    a mix of repurchase products (old) and discovery products (new).

    Performance: <10ms latency (cached in Redis)
    Fallback: If cache miss, generates on-demand (slower, ~2s)

    Returns:
        - customer_id: Customer ID
        - week: Week identifier (e.g., "2025_W45")
        - recommendations: List of 25 product recommendations
        - cached: Whether from cache (True) or generated on-demand (False)
        - latency_ms: Response time
    """
    start_time = time.time()

    try:
        # Try to get from weekly cache first
        recommendations = None
        week_key = None
        cached = False

        if weekly_cache:
            week_key = weekly_cache.get_week_key()
            recommendations = weekly_cache.get_recommendations(customer_id)

            if recommendations:
                cached = True
                logger.debug(f"Weekly cache HIT for customer {customer_id} (week {week_key})")
            else:
                logger.debug(f"Weekly cache MISS for customer {customer_id} (week {week_key})")

        # Fallback: Generate on-demand if not in cache
        if recommendations is None:
            logger.info(f"Generating on-demand recommendations for customer {customer_id}")

            conn = get_connection()
            try:
                recommender = ImprovedHybridRecommenderV32(conn=conn, use_cache=True)
                as_of_date = datetime.now().strftime('%Y-%m-%d')
                recommendations = recommender.get_recommendations(
                    customer_id=customer_id,
                    as_of_date=as_of_date,
                    top_n=25,
                    include_discovery=True
                )
            finally:
                conn.close()

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Count discovery recommendations
        discovery_count = sum(1 for r in recommendations if r.get('source') in ['discovery', 'hybrid'])

        return {
            "customer_id": customer_id,
            "week": week_key or weekly_cache.get_week_key() if weekly_cache else "unknown",
            "recommendations": recommendations,
            "count": len(recommendations),
            "discovery_count": discovery_count,
            "cached": cached,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting weekly recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/forecast/{product_id}",
    response_model=ProductForecastResponse,
    responses={404: {"model": ForecastErrorResponse}, 500: {"model": ForecastErrorResponse}},
    tags=["Forecasting"]
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
        description="Number of weeks to forecast (4-26, default: 12 = ~3 months)"
    ),
    use_cache: bool = Query(
        True,
        description="Use Redis cache for performance"
    )
):
    """
    Generate 3-month product sales forecast

    Returns weekly forecasts with:
    - Predicted quantities, revenue, order counts
    - Confidence intervals (95%)
    - Expected customers per week
    - At-risk customer identification
    - Top customer contributions

    Uses customer-based forecasting with Bayesian statistics,
    Mann-Kendall trend detection, and FFT seasonality analysis.

    **Performance**: ~1-2s for products with 20-40 customers (uses Redis caching)

    **Example**: `/forecast/25367399?forecast_weeks=12&as_of_date=2024-07-01`
    """
    start_time = time.time()

    try:
        # Get connection from pool
        conn = get_connection()

        try:
            # Initialize forecast engine
            engine = ForecastEngine(conn=conn, forecast_weeks=forecast_weeks)

            # Generate forecast (with optional caching)
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

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            metrics['requests'] += 1
            metrics['total_latency_ms'] += latency_ms

            # Convert to dict for response
            response_dict = {
                'product_id': forecast.product_id,
                'product_name': None,  # Will be enriched by engine
                'forecast_period_weeks': forecast.forecast_period_weeks,
                'summary': forecast.summary,
                'weekly_forecasts': forecast.weekly_forecasts,
                'top_customers_by_volume': forecast.top_customers_by_volume,
                'at_risk_customers': forecast.at_risk_customers,
                'model_metadata': forecast.model_metadata
            }

            logger.info(
                f"Forecast generated for product {product_id}: "
                f"{forecast.summary['total_predicted_quantity']} units, "
                f"{latency_ms:.1f}ms"
            )

            return response_dict

        finally:
            # Return connection to pool
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
    """Clear all cached recommendations for a customer"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Find all keys for this customer
        pattern = f"recs:v1:{customer_id}:*"
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
    """Clear all cached recommendations (use with caution)"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        pattern = "recs:v1:*"
        keys = list(redis_client.scan_iter(match=pattern))

        if keys:
            redis_client.delete(*keys)
            return {"status": "success", "deleted_keys": len(keys)}
        else:
            return {"status": "success", "deleted_keys": 0}

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Initialize start time
@app.on_event("startup")
async def startup_event():
    """Record start time for uptime calculation"""
    app.state.start_time = time.time()


if __name__ == "__main__":
    import uvicorn

    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
