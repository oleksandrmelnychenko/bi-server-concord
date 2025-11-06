"""
Pydantic schemas for recommendation endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ProductRecommendation(BaseModel):
    """Single product recommendation"""
    product_id: str = Field(..., description="Product identifier")
    product_name: Optional[str] = Field(None, description="Product name")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score")
    category: Optional[str] = Field(None, description="Product category")
    price: Optional[float] = Field(None, description="Product price")
    reason: Optional[str] = Field(None, description="Explanation for recommendation")


class RecommendationRequest(BaseModel):
    """Request for product recommendations"""
    customer_id: str = Field(..., description="Customer identifier")
    n_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    exclude_purchased: bool = Field(True, description="Exclude already purchased products")
    category_filter: Optional[str] = Field(None, description="Filter by product category")


class RecommendationResponse(BaseModel):
    """Response with product recommendations"""
    customer_id: str
    recommendations: List[ProductRecommendation]
    model_version: str
    generated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST-12345",
                "recommendations": [
                    {
                        "product_id": "PROD-789",
                        "product_name": "Premium Widget",
                        "score": 0.92,
                        "category": "Electronics",
                        "price": 299.99,
                        "reason": "Frequently bought together with your previous purchases"
                    }
                ],
                "model_version": "v1.2.0",
                "generated_at": "2025-01-15T10:30:00Z"
            }
        }
