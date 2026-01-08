"""
Pydantic schemas for the Order Recommendation API v2.
Enhanced with trend, seasonality, and churn adjustments.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class OrderRecommendationRequestV2(BaseModel):
    """Request body for enhanced order recommendations (v2)."""
    # Inherited from v1
    as_of_date: Optional[str] = Field(
        default=None,
        description="Reference date (YYYY-MM-DD). Defaults to today."
    )
    manufacturing_days: int = Field(
        default=14,
        ge=0,
        le=365,
        description="Manufacturing lead time in days."
    )
    logistics_days: int = Field(
        default=21,
        ge=0,
        le=365,
        description="Logistics lead time in days."
    )
    warehouse_days: int = Field(
        default=3,
        ge=0,
        le=365,
        description="Warehouse placement lead time in days."
    )
    service_level: float = Field(
        default=0.95,
        gt=0.5,
        lt=0.999,
        description="Target cycle service level (0.5-0.999)."
    )
    history_weeks: int = Field(
        default=26,
        ge=4,
        le=104,
        description="Weeks of history for demand statistics."
    )
    min_recommend_qty: float = Field(
        default=1.0,
        ge=0,
        description="Minimum recommended quantity to include in output."
    )
    product_ids: Optional[List[int]] = Field(
        default=None,
        description="Optional list of product IDs to evaluate."
    )
    supplier_id: Optional[int] = Field(
        default=None,
        description="Optional supplier ID to filter results."
    )
    max_products: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Max products to evaluate when product_ids is not provided."
    )

    # NEW v2 parameters
    use_trend_adjustment: bool = Field(
        default=True,
        description="Apply trend-based demand adjustment."
    )
    use_seasonality: bool = Field(
        default=True,
        description="Apply seasonality-based demand adjustment."
    )
    use_churn_adjustment: bool = Field(
        default=True,
        description="Apply customer churn risk adjustment."
    )
    min_history_weeks: int = Field(
        default=8,
        ge=4,
        le=52,
        description="Minimum weeks of data required for enhanced forecasting."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "as_of_date": "2026-01-08",
                "manufacturing_days": 14,
                "logistics_days": 21,
                "warehouse_days": 3,
                "service_level": 0.95,
                "history_weeks": 26,
                "min_recommend_qty": 1.0,
                "product_ids": [25208942, 25211473],
                "supplier_id": 123,
                "max_products": 500,
                "use_trend_adjustment": True,
                "use_seasonality": True,
                "use_churn_adjustment": True,
                "min_history_weeks": 8
            }
        }


class OrderRecommendationItemV2(BaseModel):
    """Single product recommendation with enhanced forecasting fields."""
    # Core fields from v1
    product_id: int = Field(description="Product ID")
    product_name: Optional[str] = Field(default=None, description="Product name")
    vendor_code: Optional[str] = Field(default=None, description="Vendor code")
    on_hand: float = Field(description="Current on-hand quantity")
    inbound_open: float = Field(description="Open inbound quantity")
    inventory_position: float = Field(description="On-hand + inbound")
    avg_weekly_demand: float = Field(description="Average weekly demand (base)")
    std_weekly_demand: float = Field(description="Weekly demand standard deviation")
    lead_time_weeks: int = Field(description="Lead time in weeks")
    demand_during_lead_time: float = Field(description="Expected demand during lead time (adjusted)")
    safety_stock: float = Field(description="Safety stock for target service level")
    reorder_point: float = Field(description="Reorder point (demand + safety stock)")
    recommended_qty: float = Field(description="Recommended order quantity")
    expected_arrival_date: str = Field(description="Expected arrival date (YYYY-MM-DD)")

    # NEW v2 fields - trend
    trend_factor: Optional[float] = Field(
        default=None,
        description="Trend multiplier (e.g., 1.05 = 5% growth expected)"
    )
    trend_direction: Optional[str] = Field(
        default=None,
        description="Trend direction: 'growing', 'declining', or 'stable'"
    )

    # NEW v2 fields - seasonality
    seasonal_index: Optional[float] = Field(
        default=None,
        description="Seasonal index (e.g., 1.2 = 20% above average for this period)"
    )
    seasonal_period_weeks: Optional[int] = Field(
        default=None,
        description="Detected seasonal period in weeks (e.g., 52 for annual)"
    )

    # NEW v2 fields - churn
    churn_adjustment: Optional[float] = Field(
        default=None,
        description="Churn multiplier (e.g., 0.95 = 5% reduction for at-risk customers)"
    )
    at_risk_demand_pct: Optional[float] = Field(
        default=None,
        description="Percentage of demand from at-risk customers"
    )

    # NEW v2 fields - forecast metadata
    forecast_method: str = Field(
        default="basic",
        description=(
            "Forecasting method used: 'basic', 'trend_adjusted', 'seasonal', "
            "'trend_seasonal', 'churn_adjusted', 'full'"
        )
    )
    forecast_confidence: Optional[float] = Field(
        default=None,
        description="Confidence score (0-1) based on data quality"
    )
    data_weeks: Optional[int] = Field(
        default=None,
        description="Actual weeks of data available for this product"
    )


class SupplierRecommendationV2(BaseModel):
    """Recommendations grouped by supplier (v2)."""
    supplier_id: Optional[int] = Field(default=None, description="Supplier ID")
    supplier_name: Optional[str] = Field(default=None, description="Supplier name")
    total_recommended_qty: float = Field(description="Total recommended quantity for supplier")
    products: List[OrderRecommendationItemV2] = Field(description="Recommended products")


class OrderRecommendationResponseV2(BaseModel):
    """Response for enhanced order recommendations (v2)."""
    as_of_date: str = Field(description="Reference date (YYYY-MM-DD)")
    manufacturing_days: int = Field(description="Manufacturing lead time in days")
    logistics_days: int = Field(description="Logistics lead time in days")
    warehouse_days: int = Field(description="Warehouse placement lead time in days")
    lead_time_days: int = Field(description="Total lead time in days")
    service_level: float = Field(description="Target service level used")
    history_weeks: int = Field(description="History weeks used for demand stats")

    # NEW v2 metadata
    use_trend_adjustment: bool = Field(description="Whether trend adjustment was applied")
    use_seasonality: bool = Field(description="Whether seasonality adjustment was applied")
    use_churn_adjustment: bool = Field(description="Whether churn adjustment was applied")
    products_with_trend: int = Field(default=0, description="Products with trend detected")
    products_with_seasonality: int = Field(default=0, description="Products with seasonality detected")
    products_with_churn_risk: int = Field(default=0, description="Products with at-risk customers")

    recommendations: List[SupplierRecommendationV2] = Field(description="Supplier recommendations")
    count: int = Field(description="Total recommended products")
    latency_ms: float = Field(description="Processing time in milliseconds")
    timestamp: str = Field(description="ISO timestamp of response")
