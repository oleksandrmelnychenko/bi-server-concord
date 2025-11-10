#!/usr/bin/env python3
"""
Pydantic models for Product Forecasting API

Matches the exact API response format specified by user requirements.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ExpectedCustomer(BaseModel):
    """Customer expected to order in a specific week"""
    customer_id: int = Field(..., description="Customer ID")
    customer_name: str = Field(..., description="Customer name")
    probability: float = Field(..., ge=0, le=1, description="Probability of ordering this week")
    expected_quantity: float = Field(..., description="Expected order quantity")
    expected_date: str = Field(..., description="Expected order date (ISO format)")
    days_since_last_order: int = Field(..., description="Days since last order")
    avg_reorder_cycle: float = Field(..., description="Average reorder cycle in days")


class WeeklyForecast(BaseModel):
    """Forecast for a single week"""
    week_start: str = Field(..., description="Week start date (ISO format)")
    week_end: str = Field(..., description="Week end date (ISO format)")
    predicted_quantity: float = Field(..., description="Predicted total quantity for week")
    predicted_revenue: float = Field(..., description="Predicted total revenue for week")
    predicted_orders: float = Field(..., description="Expected number of orders for week")
    confidence_lower: float = Field(..., description="95% confidence interval lower bound (quantity)")
    confidence_upper: float = Field(..., description="95% confidence interval upper bound (quantity)")
    expected_customers: List[ExpectedCustomer] = Field(
        default_factory=list,
        description="List of customers expected to order this week (prob >= 15%)"
    )


class ForecastSummary(BaseModel):
    """Summary metrics across entire forecast period"""
    total_predicted_quantity: float = Field(..., description="Total predicted quantity for period")
    total_predicted_revenue: float = Field(..., description="Total predicted revenue for period")
    total_predicted_orders: float = Field(..., description="Total expected orders for period")
    active_customers: int = Field(..., description="Number of active customers")
    at_risk_customers: int = Field(..., description="Number of at-risk customers")


class TopCustomer(BaseModel):
    """Top customer by predicted volume"""
    customer_id: int = Field(..., description="Customer ID")
    customer_name: str = Field(..., description="Customer name")
    predicted_quantity: float = Field(..., description="Total predicted quantity for period")
    contribution_pct: float = Field(..., description="Percentage contribution to total volume")


class AtRiskCustomer(BaseModel):
    """Customer at risk of churn"""
    customer_id: int = Field(..., description="Customer ID")
    customer_name: str = Field(..., description="Customer name")
    last_order: str = Field(..., description="Date of last order (ISO format)")
    expected_reorder: str = Field(..., description="Expected reorder date (ISO format)")
    days_overdue: int = Field(..., description="Days overdue (0 if not overdue)")
    churn_probability: float = Field(..., ge=0, le=1, description="Churn probability (0-1)")
    action: str = Field(
        ...,
        description="Recommended action: 'urgent_outreach_required', 'proactive_outreach_recommended', 'monitor_closely'"
    )


class ModelMetadata(BaseModel):
    """Model metadata and statistics"""
    model_type: str = Field(default="customer_based_aggregate", description="Model type")
    training_customers: int = Field(..., description="Number of customers used for training")
    forecast_accuracy_estimate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Estimated forecast accuracy (0-1)"
    )
    seasonality_detected: bool = Field(..., description="Whether seasonality was detected")
    model_version: str = Field(default="1.0.0", description="Model version")
    statistical_methods: List[str] = Field(
        default_factory=list,
        description="Statistical methods used"
    )


class ProductForecastResponse(BaseModel):
    """
    Complete product forecast response

    Matches exact API format specified by user requirements.
    """
    product_id: int = Field(..., description="Product ID")
    product_name: Optional[str] = Field(None, description="Product name")
    forecast_period_weeks: int = Field(..., description="Number of weeks forecasted")

    summary: ForecastSummary = Field(..., description="Summary metrics")
    weekly_forecasts: List[WeeklyForecast] = Field(..., description="Weekly forecast breakdown")
    top_customers_by_volume: List[TopCustomer] = Field(
        default_factory=list,
        description="Top customers by predicted volume"
    )
    at_risk_customers: List[AtRiskCustomer] = Field(
        default_factory=list,
        description="Customers at risk of churn"
    )
    model_metadata: ModelMetadata = Field(..., description="Model metadata and statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": 25367399,
                "product_name": "Widget Pro 3000",
                "forecast_period_weeks": 12,
                "summary": {
                    "total_predicted_quantity": 2450.0,
                    "total_predicted_revenue": 85750.00,
                    "total_predicted_orders": 87.0,
                    "active_customers": 34,
                    "at_risk_customers": 5
                },
                "weekly_forecasts": [
                    {
                        "week_start": "2025-11-11",
                        "week_end": "2025-11-17",
                        "predicted_quantity": 245.0,
                        "predicted_revenue": 8575.00,
                        "predicted_orders": 8.0,
                        "confidence_lower": 180.0,
                        "confidence_upper": 310.0,
                        "expected_customers": [
                            {
                                "customer_id": 412138,
                                "customer_name": "Acme Corp",
                                "probability": 0.85,
                                "expected_quantity": 45.0,
                                "expected_date": "2025-11-14",
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
                        "last_order": "2025-09-15",
                        "expected_reorder": "2025-10-20",
                        "days_overdue": 21,
                        "churn_probability": 0.65,
                        "action": "proactive_outreach_recommended"
                    }
                ],
                "model_metadata": {
                    "model_type": "customer_based_aggregate",
                    "training_customers": 34,
                    "forecast_accuracy_estimate": 0.78,
                    "seasonality_detected": True,
                    "model_version": "1.0.0",
                    "statistical_methods": [
                        "bayesian_inference",
                        "mann_kendall_trend",
                        "fft_seasonality",
                        "survival_analysis"
                    ]
                }
            }
        }


class ForecastErrorResponse(BaseModel):
    """Error response for forecast API"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    product_id: Optional[int] = Field(None, description="Product ID that caused error")
