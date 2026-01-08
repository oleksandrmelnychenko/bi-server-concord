"""
Pydantic schemas for the Recommendation API.

These schemas match the actual response structure from the API endpoints
and the ImprovedHybridRecommenderV33 algorithm.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class RecommendationItem(BaseModel):
    """Single product recommendation from the recommender engine."""
    product_id: int = Field(description="Product ID")
    score: float = Field(description="Recommendation score (0.0-1.0)")
    rank: int = Field(description="Rank position (1 = best)")
    segment: str = Field(
        description="Customer segment: HEAVY, REGULAR, REGULAR_CONSISTENT, LIGHT, or COLD_START"
    )
    source: str = Field(
        description="Recommendation source: repurchase, discovery, hybrid, or popular"
    )
    agreement_id: Optional[int] = Field(
        default=None,
        description="Agreement ID if applicable"
    )


class RecommendationResponse(BaseModel):
    """Response from GET /recommendations/{customer_id}."""
    customer_id: int = Field(description="Customer ID")
    date: str = Field(description="Date key (YYYY-MM-DD format)")
    recommendations: List[RecommendationItem] = Field(
        description="List of recommended products"
    )
    count: int = Field(description="Number of recommendations returned")
    discovery_count: int = Field(
        default=0,
        description="Number of discovery/hybrid recommendations (new products)"
    )
    cached: bool = Field(description="Whether result was served from cache")
    latency_ms: float = Field(description="Processing time in milliseconds")
    timestamp: str = Field(description="ISO timestamp of response")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": 410376,
                "date": "2024-01-15",
                "recommendations": [
                    {
                        "product_id": 12345,
                        "score": 0.87,
                        "rank": 1,
                        "segment": "HEAVY",
                        "source": "repurchase",
                        "agreement_id": 5678
                    },
                    {
                        "product_id": 67890,
                        "score": 0.72,
                        "rank": 2,
                        "segment": "HEAVY",
                        "source": "discovery",
                        "agreement_id": 5678
                    }
                ],
                "count": 20,
                "discovery_count": 5,
                "cached": True,
                "latency_ms": 45.2,
                "timestamp": "2024-01-15T10:30:00.000000"
            }
        }


class RecommendationFilters(BaseModel):
    """Optional filters for recommendation requests."""
    in_stock_only: bool = Field(
        default=False,
        description="Only recommend products that are currently active"
    )
    exclude_purchased: bool = Field(
        default=False,
        description="Exclude products the customer has already purchased"
    )


class RecommendationRequest(BaseModel):
    """Request body for POST recommendation endpoints (if needed)."""
    num_recommendations: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of product recommendations to return (1-50)"
    )
    filters: Optional[RecommendationFilters] = Field(
        default=None,
        description="Optional filters for recommendations"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )


# ============================================================================
# Client Payment Score Schemas
# ============================================================================

class MonthlyScore(BaseModel):
    """Monthly payment score data point."""
    month: str = Field(description="Month in YYYY-MM format")
    score: float = Field(ge=0, le=100, description="Payment score for the month")


class PaymentScoreMetrics(BaseModel):
    """Payment score metrics with exponential decay calculation."""
    # Cold start indicator
    is_cold_start: bool = Field(default=False, description="True if client has no order history in calculation period")

    # Overall composite score
    overall_score: float = Field(ge=0, le=100, description="Weighted composite payment score (0-100)")
    score_grade: str = Field(description="Grade: A (90+), B (75-89), C (60-74), D (40-59), F (<40)")

    # Paid order metrics
    paid_order_count: int = Field(default=0, description="Total paid orders in calculation period")
    avg_days_to_pay: Optional[float] = Field(default=None, description="Average days from order to payment")
    on_time_percentage: float = Field(default=0, ge=0, le=100, description="Percentage paid within 7 days")
    paid_amount: float = Field(default=0, description="Total paid order amount (UAH)")

    # Unpaid order metrics
    unpaid_order_count: int = Field(default=0, description="Count of unpaid orders")
    unpaid_amount: float = Field(default=0, description="Total unpaid amount (UAH)")
    oldest_unpaid_days: Optional[int] = Field(default=None, description="Days since oldest unpaid order")

    # Score components
    paid_score_component: float = Field(default=0, ge=0, le=100, description="Score from paid orders (0-100)")
    unpaid_score_component: float = Field(default=100, ge=0, le=100, description="Score from unpaid orders (0-100, higher = better)")

    # Trend data
    monthly_scores: List[MonthlyScore] = Field(default_factory=list, description="Score trend by month (last 6 months)")


class ClientScoreResponse(BaseModel):
    """Response from GET /client-score/{client_id}."""
    client_id: int = Field(description="Client ID")
    client_name: Optional[str] = Field(default=None, description="Client name")
    score: PaymentScoreMetrics = Field(description="Payment score metrics")
    latency_ms: float = Field(description="Processing time in milliseconds")
    timestamp: str = Field(description="ISO timestamp of response")

    class Config:
        json_schema_extra = {
            "example": {
                "client_id": 410376,
                "client_name": "АФРАКО ТЗОВ",
                "score": {
                    "overall_score": 85.5,
                    "score_grade": "B",
                    "paid_order_count": 156,
                    "avg_days_to_pay": 12.3,
                    "on_time_percentage": 72.5,
                    "paid_amount": 450000.0,
                    "unpaid_order_count": 3,
                    "unpaid_amount": 45000.0,
                    "oldest_unpaid_days": 21,
                    "paid_score_component": 82.1,
                    "unpaid_score_component": 93.5,
                    "monthly_scores": [
                        {"month": "2025-07", "score": 78.5},
                        {"month": "2025-08", "score": 82.1},
                        {"month": "2025-09", "score": 85.5}
                    ]
                },
                "latency_ms": 125.3,
                "timestamp": "2026-01-06T10:30:00.000000"
            }
        }
