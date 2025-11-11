from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class RecommendationFilters(BaseModel):
    in_stock_only: bool = Field(
        default=False,
        description="Only recommend products that are currently active (sold in last 30 days)"
    )
    exclude_purchased: bool = Field(
        default=False,
        description="Exclude products the customer has already purchased"
    )

class RecommendationRequest(BaseModel):
    num_recommendations: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of product recommendations to return (1-50)"
    )
    filters: RecommendationFilters = Field(
        default_factory=RecommendationFilters,
        description="Optional filters for recommendations"
    )

class ProductMetadata(BaseModel):
    price: Optional[float] = Field(description="Current product price")
    num_analogues: int = Field(description="Number of alternative products available")
    times_ordered: int = Field(description="How many times this product has been ordered")
    product_status: str = Field(description="Product status (Active, Slow Moving, etc.)")

class ProductRecommendation(BaseModel):
    product_id: str = Field(description="Unique product identifier")
    product_name: str = Field(description="Product name")
    score: float = Field(description="Recommendation score (higher = more relevant)")
    reason: str = Field(description="Why this product was recommended (personalized, popular, etc.)")
    metadata: ProductMetadata = Field(description="Additional product information")

class RecommendationMetadata(BaseModel):
    model_version: str = Field(description="Model version used for recommendations")
    generated_at: datetime = Field(description="When recommendations were generated")
    num_results: int = Field(description="Number of recommendations returned")
    filtered: RecommendationFilters = Field(description="Filters that were applied")

class RecommendationResponse(BaseModel):
    customer_id: str = Field(description="Customer ID recommendations are for")
    recommendations: List[ProductRecommendation] = Field(description="List of recommended products")
    metadata: RecommendationMetadata = Field(description="Response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "12345",
                "recommendations": [
                    {
                        "product_id": "67890",
                        "product_name": "Амортизатор подвески прицепа",
                        "score": 0.87,
                        "reason": "personalized",
                        "metadata": {
                            "price": 250.50,
                            "num_analogues": 26,
                            "times_ordered": 42,
                            "product_status": "Active"
                        }
                    }
                ],
                "metadata": {
                    "model_version": "v1",
                    "generated_at": "2025-10-23T10:30:00Z",
                    "num_results": 20,
                    "filtered": {
                        "in_stock_only": True,
                        "exclude_purchased": False
                    }
                }
            }
        }

class ErrorResponse(BaseModel):
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
