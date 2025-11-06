"""
Pydantic schemas for forecasting endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date


class ForecastDataPoint(BaseModel):
    """Single forecast data point"""
    date: date
    predicted_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastRequest(BaseModel):
    """Request for sales forecast"""
    periods: int = Field(..., ge=1, le=365, description="Number of periods to forecast")
    product_id: Optional[str] = Field(None, description="Specific product ID")
    category: Optional[str] = Field(None, description="Product category")
    include_confidence: bool = Field(True, description="Include confidence intervals")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level")


class ForecastResponse(BaseModel):
    """Response with forecast data"""
    forecast: List[ForecastDataPoint]
    product_id: Optional[str] = None
    category: Optional[str] = None
    model_version: str
    generated_at: datetime
    metrics: Optional[dict] = Field(None, description="Model performance metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "forecast": [
                    {
                        "date": "2025-01-20",
                        "predicted_value": 15420.50,
                        "lower_bound": 14200.00,
                        "upper_bound": 16800.00
                    }
                ],
                "product_id": "PROD-123",
                "category": "Electronics",
                "model_version": "v1.0.0",
                "generated_at": "2025-01-15T10:30:00Z",
                "metrics": {
                    "mae": 245.67,
                    "rmse": 312.45,
                    "mape": 0.042
                }
            }
        }
