#!/usr/bin/env python3

import os
import sys
import time
import math
import logging
import asyncio
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from collections import defaultdict

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
from scripts.datetime_utils import parse_as_of_date, format_date_iso
from api.db_pool import get_connection, close_pool
from api.models.forecast_schemas import ProductForecastResponse, ForecastErrorResponse
from api.models.recommendation_schemas import ClientScoreResponse, PaymentScoreMetrics, MonthlyScore
from api.models.order_recommendation_schemas_v2 import (
    OrderRecommendationRequestV2,
    OrderRecommendationResponseV2,
    SupplierRecommendationV2,
    OrderRecommendationItemV2
)
from api.demand_forecaster import get_enhanced_demand_stats

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


def fetch_client_payment_score(conn, client_id: int) -> Dict[str, Any]:
    """
    Fetch client payment score with exponential decay calculation.

    Score formula: 100 * e^(-0.05 * max(0, days_to_pay - 7))
    - 7 day grace period (100% if paid within 7 days)
    - Fast initial decay, then slows

    Overall score: (paid_score * 0.7) + (unpaid_score * 0.3)
    """
    import math
    cursor = conn.cursor()

    result = {
        "client_name": None,
        "is_cold_start": False,
        "overall_score": 0.0,
        "score_grade": "F",
        "paid_order_count": 0,
        "avg_days_to_pay": None,
        "on_time_percentage": 0.0,
        "paid_amount": 0.0,
        "unpaid_order_count": 0,
        "unpaid_amount": 0.0,
        "oldest_unpaid_days": None,
        "paid_score_component": 0.0,
        "unpaid_score_component": 100.0,
        "monthly_scores": []
    }

    try:
        # Get client name
        cursor.execute("""
            SELECT Name, FullName FROM dbo.Client WHERE ID = ? AND Deleted = 0
        """, (client_id,))
        row = cursor.fetchone()
        if row:
            result["client_name"] = row[1] if row[1] else row[0]

        # Get agreement IDs for this client
        cursor.execute("""
            SELECT ID FROM dbo.ClientAgreement WHERE ClientID = ? AND Deleted = 0
        """, (client_id,))
        agreement_ids = [r[0] for r in cursor.fetchall()]

        if not agreement_ids:
            # No agreements = cold start, neutral score
            result["is_cold_start"] = True
            result["overall_score"] = 50.0
            result["score_grade"] = "C"
            return result

        agreement_list = ','.join(str(a) for a in agreement_ids)

        # Get paid orders with payment dates (last 12 months)
        # Join path: Order -> Sale -> IncomePaymentOrderSale -> IncomePaymentOrder
        cursor.execute(f"""
            WITH PaidOrders AS (
                SELECT
                    o.ID as OrderID,
                    o.Created as OrderDate,
                    MIN(ipo.Created) as PaymentDate,
                    DATEDIFF(day, o.Created, MIN(ipo.Created)) as DaysToPay,
                    SUM(oi.Qty * oi.PricePerItem) as OrderAmount
                FROM dbo.[Order] o
                JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
                JOIN dbo.Sale s ON s.OrderID = o.ID
                JOIN dbo.IncomePaymentOrderSale ipos ON ipos.SaleID = s.ID
                JOIN dbo.IncomePaymentOrder ipo ON ipo.ID = ipos.IncomePaymentOrderID
                WHERE o.ClientAgreementID IN ({agreement_list})
                  AND o.Created >= DATEADD(month, -12, GETDATE())
                  AND o.Deleted = 0
                  AND s.Deleted = 0
                GROUP BY o.ID, o.Created
            )
            SELECT
                COUNT(*) as paid_count,
                AVG(CAST(DaysToPay as FLOAT)) as avg_days,
                SUM(CASE WHEN DaysToPay <= 7 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as on_time_pct,
                SUM(OrderAmount) as paid_amount,
                -- Individual scores for weighted average
                AVG(100.0 * EXP(-0.05 * CASE WHEN DaysToPay > 7 THEN DaysToPay - 7 ELSE 0 END)) as paid_score
            FROM PaidOrders
        """)

        paid_row = cursor.fetchone()
        if paid_row and paid_row[0] > 0:
            result["paid_order_count"] = paid_row[0]
            result["avg_days_to_pay"] = round(paid_row[1], 1) if paid_row[1] else None
            result["on_time_percentage"] = round(paid_row[2], 1) if paid_row[2] else 0.0
            result["paid_amount"] = round(float(paid_row[3] or 0), 2)
            result["paid_score_component"] = round(paid_row[4], 1) if paid_row[4] else 0.0

        # Get unpaid orders (orders without payment link)
        cursor.execute(f"""
            SELECT
                COUNT(DISTINCT o.ID) as unpaid_count,
                ISNULL(SUM(oi.Qty * oi.PricePerItem), 0) as unpaid_amount,
                MAX(DATEDIFF(day, o.Created, GETDATE())) as oldest_days
            FROM dbo.[Order] o
            JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            LEFT JOIN dbo.Sale s ON s.OrderID = o.ID AND s.Deleted = 0
            WHERE o.ClientAgreementID IN ({agreement_list})
              AND o.Created >= DATEADD(month, -12, GETDATE())
              AND o.Deleted = 0
              AND NOT EXISTS (
                  SELECT 1 FROM dbo.IncomePaymentOrderSale ipos
                  WHERE ipos.SaleID = s.ID
              )
        """)

        unpaid_row = cursor.fetchone()
        if unpaid_row:
            result["unpaid_order_count"] = unpaid_row[0] or 0
            result["unpaid_amount"] = round(float(unpaid_row[1] or 0), 2)
            result["oldest_unpaid_days"] = unpaid_row[2]

            # Calculate unpaid score component (penalty based on oldest unpaid)
            if unpaid_row[0] and unpaid_row[0] > 0 and unpaid_row[2]:
                oldest_days = unpaid_row[2]
                # Exponential decay penalty for unpaid orders
                unpaid_penalty = 100.0 * math.exp(-0.05 * max(0, oldest_days - 7))
                result["unpaid_score_component"] = round(unpaid_penalty, 1)
            else:
                result["unpaid_score_component"] = 100.0

        # Calculate overall score: 70% paid behavior + 30% unpaid penalty
        paid_weight = 0.7
        unpaid_weight = 0.3

        if result["paid_order_count"] > 0:
            overall = (result["paid_score_component"] * paid_weight) + \
                      (result["unpaid_score_component"] * unpaid_weight)
        else:
            # No paid orders yet - use only unpaid component with neutral base
            overall = 50.0 * (result["unpaid_score_component"] / 100.0)

        result["overall_score"] = round(overall, 1)

        # Mark as cold start if no order history at all
        if result["paid_order_count"] == 0 and result["unpaid_order_count"] == 0:
            result["is_cold_start"] = True

        # Assign grade
        if overall >= 90:
            result["score_grade"] = "A"
        elif overall >= 75:
            result["score_grade"] = "B"
        elif overall >= 60:
            result["score_grade"] = "C"
        elif overall >= 40:
            result["score_grade"] = "D"
        else:
            result["score_grade"] = "F"

        # Get monthly scores (last 6 months)
        cursor.execute(f"""
            WITH OrderPayments AS (
                SELECT
                    o.ID as OrderID,
                    FORMAT(o.Created, 'yyyy-MM') as month,
                    DATEDIFF(day, o.Created, MIN(ipo.Created)) as DaysToPay
                FROM dbo.[Order] o
                JOIN dbo.Sale s ON s.OrderID = o.ID
                JOIN dbo.IncomePaymentOrderSale ipos ON ipos.SaleID = s.ID
                JOIN dbo.IncomePaymentOrder ipo ON ipo.ID = ipos.IncomePaymentOrderID
                WHERE o.ClientAgreementID IN ({agreement_list})
                  AND o.Created >= DATEADD(month, -6, GETDATE())
                  AND o.Deleted = 0
                  AND s.Deleted = 0
                GROUP BY o.ID, o.Created
            )
            SELECT
                month,
                AVG(100.0 * EXP(-0.05 * CASE WHEN DaysToPay > 7 THEN DaysToPay - 7 ELSE 0 END)) as avg_score
            FROM OrderPayments
            GROUP BY month
            ORDER BY month
        """)

        result["monthly_scores"] = [
            {"month": r[0], "score": round(r[1], 1)}
            for r in cursor.fetchall()
        ]

    except Exception as e:
        logger.warning(f"Error fetching client payment score: {e}")

    return result


def _normal_inv_cdf(p: float) -> float:
    """Approximate inverse CDF for standard normal distribution."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("service_level must be between 0 and 1")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


MAX_IN_CLAUSE_ITEMS = 1000


def _build_in_clause(items: List[int]) -> str:
    return ",".join(["?"] * len(items))


def _chunked(items: List[int], size: int) -> Iterator[List[int]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _fetch_candidate_products(
    conn,
    history_start: str,
    as_of_date: str,
    max_products: int
) -> List[int]:
    cursor = conn.cursor()
    query = """
        SELECT TOP (?) oi.ProductID
        FROM dbo.OrderItem oi
        JOIN dbo.[Order] o ON oi.OrderID = o.ID
        JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
        JOIN dbo.Product p ON oi.ProductID = p.ID
        WHERE o.Deleted = 0
          AND oi.Deleted = 0
          AND ca.Deleted = 0
          AND p.Deleted = 0
          AND p.IsForSale = 1
          AND o.Created >= ?
          AND o.Created < ?
        GROUP BY oi.ProductID
        ORDER BY SUM(oi.Qty) DESC
    """
    cursor.execute(query, (max_products, history_start, as_of_date))
    rows = cursor.fetchall()
    cursor.close()
    return [row[0] for row in rows]


def _fetch_on_hand(
    conn,
    product_ids: List[int]
) -> tuple[Dict[int, float], List[int]]:
    if not product_ids:
        return {}, []

    cursor = conn.cursor()
    on_hand: Dict[int, float] = {}
    for chunk in _chunked(product_ids, MAX_IN_CLAUSE_ITEMS):
        placeholders = _build_in_clause(chunk)
        query = f"""
            SELECT pa.ProductID, SUM(pa.Amount) as on_hand
            FROM dbo.ProductAvailability pa
            JOIN dbo.Storage s ON pa.StorageID = s.ID
            JOIN dbo.Product p ON pa.ProductID = p.ID
            WHERE pa.Deleted = 0
              AND s.Deleted = 0
              AND s.ForDefective = 0
              AND p.Deleted = 0
              AND p.IsForSale = 1
              AND pa.ProductID IN ({placeholders})
            GROUP BY pa.ProductID
        """
        cursor.execute(query, chunk)
        rows = cursor.fetchall()
        for row in rows:
            on_hand[row[0]] = float(row[1] or 0)
        for product_id in chunk:
            on_hand.setdefault(product_id, 0.0)
    cursor.close()
    return on_hand, list(product_ids)


def _fetch_inbound_open(conn, product_ids: List[int]) -> Dict[int, float]:
    inbound = defaultdict(float)
    if not product_ids:
        return inbound

    cursor = conn.cursor()

    for chunk in _chunked(product_ids, MAX_IN_CLAUSE_ITEMS):
        placeholders = _build_in_clause(chunk)
        query = f"""
            WITH received AS (
                SELECT SupplyOrderItemID, SUM(Qty) AS received_qty
                FROM dbo.SupplyInvoiceOrderItem
                WHERE Deleted = 0
                GROUP BY SupplyOrderItemID
            )
            SELECT soi.ProductID,
                   SUM(soi.Qty - ISNULL(received.received_qty, 0)) AS inbound_qty
            FROM dbo.SupplyOrderItem soi
            JOIN dbo.SupplyOrder so ON soi.SupplyOrderID = so.ID
            LEFT JOIN received ON received.SupplyOrderItemID = soi.ID
            WHERE soi.Deleted = 0
              AND so.Deleted = 0
              AND (so.IsOrderArrived = 0 OR so.OrderArrivedDate IS NULL
                   OR ISNULL(received.received_qty, 0) < soi.Qty)
              AND soi.ProductID IN ({placeholders})
            GROUP BY soi.ProductID
        """
        cursor.execute(query, chunk)
        for row in cursor.fetchall():
            inbound[row[0]] += max(0.0, float(row[1] or 0))

        query = f"""
            SELECT soui.ProductID,
                   SUM(CASE WHEN soui.NotOrdered = 1 THEN 0
                            ELSE ISNULL(soui.RemainingQty, 0) END) AS inbound_qty
            FROM dbo.SupplyOrderUkraineItem soui
            JOIN dbo.SupplyOrderUkraine sou ON soui.SupplyOrderUkraineID = sou.ID
            WHERE soui.Deleted = 0
              AND sou.Deleted = 0
              AND ISNULL(soui.RemainingQty, 0) > 0
              AND soui.ProductID IN ({placeholders})
            GROUP BY soui.ProductID
        """
        cursor.execute(query, chunk)
        for row in cursor.fetchall():
            inbound[row[0]] += max(0.0, float(row[1] or 0))

    cursor.close()
    return inbound


def _fetch_demand_stats(
    conn,
    product_ids: List[int],
    start_date: str,
    end_date: str,
    history_weeks: int
) -> Dict[int, Dict[str, float]]:
    stats = {}
    if not product_ids:
        return stats

    placeholders = _build_in_clause(product_ids)
    query = f"""
        WITH weekly_demand AS (
            SELECT
                oi.ProductID,
                DATEADD(day, DATEDIFF(day, 0, o.Created) / 7 * 7, 0) AS week_start,
                SUM(oi.Qty) AS weekly_qty
            FROM dbo.OrderItem oi
            JOIN dbo.[Order] o ON oi.OrderID = o.ID
            JOIN dbo.Product p ON oi.ProductID = p.ID
            WHERE o.Deleted = 0
              AND oi.Deleted = 0
              AND p.Deleted = 0
              AND p.IsForSale = 1
              AND o.Created >= ?
              AND o.Created < ?
              AND oi.ProductID IN ({placeholders})
            GROUP BY oi.ProductID, DATEADD(day, DATEDIFF(day, 0, o.Created) / 7 * 7, 0)
        )
        SELECT ProductID,
               SUM(weekly_qty) AS total_qty,
               SUM(weekly_qty * weekly_qty) AS sum_sq
        FROM weekly_demand
        GROUP BY ProductID
    """
    params = [start_date, end_date] + product_ids
    cursor = conn.cursor()
    cursor.execute(query, params)

    for row in cursor.fetchall():
        product_id = row[0]
        total_qty = float(row[1] or 0)
        sum_sq = float(row[2] or 0)
        mean = total_qty / history_weeks
        variance = (sum_sq / history_weeks) - (mean * mean)
        stddev = math.sqrt(max(variance, 0))
        stats[product_id] = {
            "mean": mean,
            "stddev": stddev,
            "total_qty": total_qty
        }

    cursor.close()
    return stats


def _fetch_product_details(conn, product_ids: List[int]) -> Dict[int, Dict[str, Optional[str]]]:
    details = {}
    if not product_ids:
        return details

    cursor = conn.cursor()
    for chunk in _chunked(product_ids, MAX_IN_CLAUSE_ITEMS):
        placeholders = _build_in_clause(chunk)
        query = f"""
            SELECT p.ID, p.Name, p.VendorCode
            FROM dbo.Product p
            WHERE p.Deleted = 0
              AND p.IsForSale = 1
              AND p.ID IN ({placeholders})
        """
        cursor.execute(query, chunk)
        for row in cursor.fetchall():
            details[row[0]] = {
                "name": row[1],
                "vendor_code": row[2]
            }
    cursor.close()
    return details


def _fetch_supplier_map(conn, product_ids: List[int]) -> Dict[int, int]:
    supplier_map = {}
    if not product_ids:
        return supplier_map

    cursor = conn.cursor()
    for chunk in _chunked(product_ids, MAX_IN_CLAUSE_ITEMS):
        placeholders = _build_in_clause(chunk)
        query = f"""
            WITH supply_history AS (
                SELECT soi.ProductID, so.ClientID AS SupplierID, so.Created AS OrderDate
                FROM dbo.SupplyOrderItem soi
                JOIN dbo.SupplyOrder so ON soi.SupplyOrderID = so.ID
                WHERE soi.Deleted = 0
                  AND so.Deleted = 0
                  AND soi.ProductID IN ({placeholders})
                UNION ALL
                SELECT soui.ProductID, sou.SupplierID AS SupplierID, sou.Created AS OrderDate
                FROM dbo.SupplyOrderUkraineItem soui
                JOIN dbo.SupplyOrderUkraine sou ON soui.SupplyOrderUkraineID = sou.ID
                WHERE soui.Deleted = 0
                  AND sou.Deleted = 0
                  AND soui.ProductID IN ({placeholders})
            ),
            ranked AS (
                SELECT ProductID, SupplierID, OrderDate,
                       ROW_NUMBER() OVER (PARTITION BY ProductID ORDER BY OrderDate DESC) AS rn
                FROM supply_history
            )
            SELECT ProductID, SupplierID
            FROM ranked
            WHERE rn = 1
        """
        params = chunk + chunk
        cursor.execute(query, params)
        for row in cursor.fetchall():
            supplier_map[row[0]] = row[1]
    cursor.close()
    return supplier_map


def _fetch_supplier_names(conn, supplier_ids: List[int]) -> Dict[int, str]:
    names = {}
    if not supplier_ids:
        return names

    cursor = conn.cursor()
    for chunk in _chunked(supplier_ids, MAX_IN_CLAUSE_ITEMS):
        placeholders = _build_in_clause(chunk)
        query = f"""
            SELECT ID, COALESCE(FullName, Name) AS SupplierName
            FROM dbo.Client
            WHERE Deleted = 0
              AND ID IN ({placeholders})
        """
        cursor.execute(query, chunk)
        for row in cursor.fetchall():
            names[row[0]] = row[1]
    cursor.close()
    return names


@app.get("/client-score/{client_id}", response_model=ClientScoreResponse, tags=["Analytics"])
async def get_client_payment_score(client_id: int):
    """
    Get payment score and metrics for a client.

    Score calculation uses exponential decay:
    - 100% if paid within 7 days grace period
    - Formula: score = 100 * e^(-0.05 * max(0, days_to_pay - 7))
    - Overall: 70% paid behavior + 30% unpaid penalty

    Grades: A (90+), B (75-89), C (60-74), D (40-59), F (<40)
    """
    start_time = time.time()

    try:
        conn = get_connection()
        try:
            score_data = fetch_client_payment_score(conn, client_id)
        finally:
            conn.close()

        latency_ms = (time.time() - start_time) * 1000
        metrics['requests'] += 1
        metrics['total_latency_ms'] += latency_ms

        return ClientScoreResponse(
            client_id=client_id,
            client_name=score_data.get("client_name"),
            score=PaymentScoreMetrics(
                is_cold_start=score_data.get("is_cold_start", False),
                overall_score=score_data["overall_score"],
                score_grade=score_data["score_grade"],
                paid_order_count=score_data["paid_order_count"],
                avg_days_to_pay=score_data["avg_days_to_pay"],
                on_time_percentage=score_data["on_time_percentage"],
                paid_amount=score_data["paid_amount"],
                unpaid_order_count=score_data["unpaid_order_count"],
                unpaid_amount=score_data["unpaid_amount"],
                oldest_unpaid_days=score_data["oldest_unpaid_days"],
                paid_score_component=score_data["paid_score_component"],
                unpaid_score_component=score_data["unpaid_score_component"],
                monthly_scores=[
                    MonthlyScore(month=m["month"], score=m["score"])
                    for m in score_data["monthly_scores"]
                ]
            ),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        metrics['errors'] += 1
        logger.error(f"Error getting client payment score: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post(
    "/order-recommendations/v2",
    response_model=OrderRecommendationResponseV2,
    tags=["Supply Planning"]
)
async def get_order_recommendations_v2(request: OrderRecommendationRequestV2):
    """
    Enhanced order recommendations with trend, seasonality, and churn adjustments.

    This v2 endpoint provides smarter demand forecasting by:
    - **Trend Analysis**: Detects growing/declining demand and adjusts forecast
    - **Seasonality Detection**: Identifies seasonal patterns (monthly, quarterly, annual)
    - **Churn Risk Adjustment**: Reduces forecast for at-risk customer demand

    Returns additional fields showing the adjustments applied.
    """
    start_time = time.time()

    try:
        as_of_dt = parse_as_of_date(request.as_of_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    lead_time_days = request.manufacturing_days + request.logistics_days + request.warehouse_days
    lead_time_weeks = max(1, math.ceil(lead_time_days / 7))
    as_of_date = format_date_iso(as_of_dt)
    expected_arrival_date = format_date_iso(as_of_dt + timedelta(days=lead_time_days))
    history_start = format_date_iso(as_of_dt - timedelta(days=request.history_weeks * 7))

    try:
        z_score = _normal_inv_cdf(request.service_level)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        product_ids_input = request.product_ids or []
        product_ids_input = list(dict.fromkeys(product_ids_input))
        if product_ids_input and len(product_ids_input) > request.max_products:
            product_ids_input = product_ids_input[:request.max_products]

        conn = get_connection()
        try:
            if product_ids_input:
                product_ids = product_ids_input
            else:
                product_ids = _fetch_candidate_products(
                    conn=conn,
                    history_start=history_start,
                    as_of_date=as_of_date,
                    max_products=request.max_products
                )

            on_hand, product_ids = _fetch_on_hand(
                conn=conn,
                product_ids=product_ids
            )
            if not product_ids:
                latency_ms = (time.time() - start_time) * 1000
                metrics['requests'] += 1
                metrics['total_latency_ms'] += latency_ms
                return OrderRecommendationResponseV2(
                    as_of_date=as_of_date,
                    manufacturing_days=request.manufacturing_days,
                    logistics_days=request.logistics_days,
                    warehouse_days=request.warehouse_days,
                    lead_time_days=lead_time_days,
                    service_level=request.service_level,
                    history_weeks=request.history_weeks,
                    use_trend_adjustment=request.use_trend_adjustment,
                    use_seasonality=request.use_seasonality,
                    use_churn_adjustment=request.use_churn_adjustment,
                    products_with_trend=0,
                    products_with_seasonality=0,
                    products_with_churn_risk=0,
                    recommendations=[],
                    count=0,
                    latency_ms=round(latency_ms, 2),
                    timestamp=datetime.now().isoformat()
                )

            inbound_open = _fetch_inbound_open(conn, product_ids)

            # Use enhanced demand stats with trend/seasonality/churn
            enhanced_stats = get_enhanced_demand_stats(
                conn=conn,
                product_ids=product_ids,
                start_date=history_start,
                end_date=as_of_date,
                history_weeks=request.history_weeks,
                lead_time_weeks=lead_time_weeks,
                as_of_date=as_of_dt,
                use_trend=request.use_trend_adjustment,
                use_seasonality=request.use_seasonality,
                use_churn=request.use_churn_adjustment,
                min_history_weeks=request.min_history_weeks
            )

            supplier_map = _fetch_supplier_map(conn, product_ids)
            product_details = _fetch_product_details(conn, product_ids)
            supplier_ids = list({sid for sid in supplier_map.values() if sid is not None})
            supplier_names = _fetch_supplier_names(conn, supplier_ids)
        finally:
            conn.close()

        supplier_groups: Dict[Optional[int], Dict[str, Any]] = {}
        products_with_trend = 0
        products_with_seasonality = 0
        products_with_churn_risk = 0

        for product_id in product_ids:
            details = product_details.get(product_id)
            if not details:
                continue

            supplier_id = supplier_map.get(product_id)
            if request.supplier_id is not None and supplier_id != request.supplier_id:
                continue

            stats = enhanced_stats.get(product_id)
            if not stats:
                continue

            # Use adjusted mean for demand calculation
            avg_weekly = stats.adjusted_mean
            std_weekly = stats.stddev

            demand_during_lt = avg_weekly * lead_time_weeks
            std_during_lt = std_weekly * math.sqrt(lead_time_weeks)
            safety_stock = z_score * std_during_lt
            reorder_point = demand_during_lt + safety_stock

            inventory_position = on_hand.get(product_id, 0.0) + inbound_open.get(product_id, 0.0)
            raw_recommend_qty = max(0.0, reorder_point - inventory_position)
            recommended_qty = float(math.ceil(raw_recommend_qty))

            if recommended_qty < request.min_recommend_qty:
                continue

            supplier_name = supplier_names.get(supplier_id)
            if not supplier_name:
                supplier_name = "Unknown Supplier"

            # Track statistics
            if stats.trend and stats.trend.direction != "stable":
                products_with_trend += 1
            if stats.seasonality and stats.seasonality.period_weeks:
                products_with_seasonality += 1
            if stats.churn and stats.churn.at_risk_pct > 0.05:
                products_with_churn_risk += 1

            item = OrderRecommendationItemV2(
                product_id=product_id,
                product_name=details.get("name"),
                vendor_code=details.get("vendor_code"),
                on_hand=round(float(on_hand.get(product_id, 0.0)), 2),
                inbound_open=round(float(inbound_open.get(product_id, 0.0)), 2),
                inventory_position=round(float(inventory_position), 2),
                avg_weekly_demand=round(float(stats.mean), 2),
                std_weekly_demand=round(float(std_weekly), 2),
                lead_time_weeks=lead_time_weeks,
                demand_during_lead_time=round(float(demand_during_lt), 2),
                safety_stock=round(float(safety_stock), 2),
                reorder_point=round(float(reorder_point), 2),
                recommended_qty=round(float(recommended_qty), 2),
                expected_arrival_date=expected_arrival_date,
                # V2 fields
                trend_factor=stats.trend.factor if stats.trend else None,
                trend_direction=stats.trend.direction if stats.trend else None,
                seasonal_index=stats.seasonality.index if stats.seasonality else None,
                seasonal_period_weeks=stats.seasonality.period_weeks if stats.seasonality else None,
                churn_adjustment=stats.churn.adjustment if stats.churn else None,
                at_risk_demand_pct=stats.churn.at_risk_pct if stats.churn else None,
                forecast_method=stats.forecast_method,
                forecast_confidence=stats.forecast_confidence,
                data_weeks=stats.data_weeks
            )

            group = supplier_groups.setdefault(
                supplier_id,
                {
                    "supplier_id": supplier_id,
                    "supplier_name": supplier_name,
                    "total_qty": 0.0,
                    "products": []
                }
            )
            group["products"].append(item)
            group["total_qty"] += recommended_qty

        recommendations = []
        for group in supplier_groups.values():
            group["products"].sort(key=lambda x: x.recommended_qty, reverse=True)
            recommendations.append(SupplierRecommendationV2(
                supplier_id=group["supplier_id"],
                supplier_name=group["supplier_name"],
                total_recommended_qty=round(float(group["total_qty"]), 2),
                products=group["products"]
            ))

        recommendations.sort(key=lambda g: g.total_recommended_qty, reverse=True)
        count = sum(len(group.products) for group in recommendations)

        latency_ms = (time.time() - start_time) * 1000
        metrics['requests'] += 1
        metrics['total_latency_ms'] += latency_ms

        return OrderRecommendationResponseV2(
            as_of_date=as_of_date,
            manufacturing_days=request.manufacturing_days,
            logistics_days=request.logistics_days,
            warehouse_days=request.warehouse_days,
            lead_time_days=lead_time_days,
            service_level=request.service_level,
            history_weeks=request.history_weeks,
            use_trend_adjustment=request.use_trend_adjustment,
            use_seasonality=request.use_seasonality,
            use_churn_adjustment=request.use_churn_adjustment,
            products_with_trend=products_with_trend,
            products_with_seasonality=products_with_seasonality,
            products_with_churn_risk=products_with_churn_risk,
            recommendations=recommendations,
            count=count,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        metrics['errors'] += 1
        logger.error(f"Error generating order recommendations v2: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


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
