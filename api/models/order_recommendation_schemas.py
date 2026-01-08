"""
Pydantic schemas for the Order Recommendation API.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class OrderRecommendationRequest(BaseModel):
    """Request body for order recommendations."""
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

    class Config:
        json_schema_extra = {
            "example": {
                "as_of_date": "2026-01-07",
                "manufacturing_days": 14,
                "logistics_days": 21,
                "warehouse_days": 3,
                "service_level": 0.95,
                "history_weeks": 26,
                "min_recommend_qty": 1.0,
                "product_ids": [25208942, 25211473],
                "supplier_id": 123,
                "max_products": 500
            }
        }


class OrderRecommendationItem(BaseModel):
    """Single product recommendation."""
    product_id: int = Field(description="Product ID")
    product_name: Optional[str] = Field(default=None, description="Product name")
    vendor_code: Optional[str] = Field(default=None, description="Vendor code")
    on_hand: float = Field(description="Current on-hand quantity")
    inbound_open: float = Field(description="Open inbound quantity")
    inventory_position: float = Field(description="On-hand + inbound")
    avg_weekly_demand: float = Field(description="Average weekly demand")
    std_weekly_demand: float = Field(description="Weekly demand standard deviation")
    lead_time_weeks: int = Field(description="Lead time in weeks")
    demand_during_lead_time: float = Field(description="Expected demand during lead time")
    safety_stock: float = Field(description="Safety stock for target service level")
    reorder_point: float = Field(description="Reorder point (demand + safety stock)")
    recommended_qty: float = Field(description="Recommended order quantity")
    expected_arrival_date: str = Field(description="Expected arrival date (YYYY-MM-DD)")


class SupplierRecommendation(BaseModel):
    """Recommendations grouped by supplier."""
    supplier_id: Optional[int] = Field(default=None, description="Supplier ID")
    supplier_name: Optional[str] = Field(default=None, description="Supplier name")
    total_recommended_qty: float = Field(description="Total recommended quantity for supplier")
    products: List[OrderRecommendationItem] = Field(description="Recommended products")


class OrderRecommendationResponse(BaseModel):
    """Response for order recommendations."""
    as_of_date: str = Field(description="Reference date (YYYY-MM-DD)")
    manufacturing_days: int = Field(description="Manufacturing lead time in days")
    logistics_days: int = Field(description="Logistics lead time in days")
    warehouse_days: int = Field(description="Warehouse placement lead time in days")
    lead_time_days: int = Field(description="Total lead time in days")
    service_level: float = Field(description="Target service level used")
    history_weeks: int = Field(description="History weeks used for demand stats")
    recommendations: List[SupplierRecommendation] = Field(description="Supplier recommendations")
    count: int = Field(description="Total recommended products")
    latency_ms: float = Field(description="Processing time in milliseconds")
    timestamp: str = Field(description="ISO timestamp of response")
