"""
Recommendation API Routes
"""

from fastapi import APIRouter, HTTPException, Path, Body
from datetime import datetime
import logging
from pathlib import Path as FilePath

from api.models.recommendation_schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationMetadata,
    ProductRecommendation,
    ErrorResponse
)
from ml.models.lightfm_recommender import LightFMRecommender

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])

# Initialize recommender (singleton)
MODEL_PATH = "/opt/dagster/app/models/lightfm/recommendation_v1.pkl"
DUCKDB_PATH = "/opt/dagster/app/data/dbt/concord_bi.duckdb"

recommender = None

def get_recommender() -> LightFMRecommender:
    """Get or create recommender instance"""
    global recommender
    if recommender is None:
        logger.info("Initializing LightFM recommender...")
        recommender = LightFMRecommender(MODEL_PATH, DUCKDB_PATH)
    return recommender

@router.post(
    "/customer/{customer_id}",
    response_model=RecommendationResponse,
    responses={
        200: {"description": "Successful recommendations"},
        404: {"model": ErrorResponse, "description": "Customer not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get Product Recommendations",
    description="""
    Get personalized product recommendations for a customer.

    The recommendations are generated using a hybrid LightFM model that considers:
    - Customer purchase history
    - Customer RFM segment (Recency, Frequency, Monetary)
    - Product popularity and sales metrics
    - Product relationships (analogues/alternatives)

    **Filtering Options**:
    - `in_stock_only`: Only recommend products marked as "Active" (sold in last 30 days)
    - `exclude_purchased`: Don't recommend products the customer already bought

    **Number of Recommendations**:
    - Default: 20 products
    - Minimum: 1 product
    - Maximum: 50 products

    **Response**:
    - Products are sorted by relevance score (higher = more relevant)
    - Each product includes metadata (price, alternatives, order count)
    - If customer has no history, popular products are returned instead
    """
)
async def get_recommendations(
    customer_id: str = Path(..., description="Customer ID to get recommendations for"),
    request: RecommendationRequest = Body(
        default=RecommendationRequest(),
        description="Recommendation request parameters"
    )
) -> RecommendationResponse:
    """
    Get product recommendations for a customer

    Args:
        customer_id: Customer ID
        request: Recommendation parameters (num_recommendations, filters)

    Returns:
        RecommendationResponse with recommended products

    Raises:
        HTTPException: If customer not found or error occurs
    """
    try:
        # Get recommender
        rec = get_recommender()

        # Get customer info (to verify exists)
        customer_info = rec.get_customer_info(customer_id)

        if customer_info is None:
            logger.warning(f"Customer {customer_id} not found")
            # Still try to recommend popular products
            # raise HTTPException(
            #     status_code=404,
            #     detail=f"Customer {customer_id} not found in database"
            # )

        # Get recommendations
        logger.info(
            f"Generating {request.num_recommendations} recommendations "
            f"for customer {customer_id} "
            f"(in_stock={request.filters.in_stock_only}, "
            f"exclude_purchased={request.filters.exclude_purchased})"
        )

        recommendations = rec.recommend(
            customer_id=customer_id,
            num_recommendations=request.num_recommendations,
            in_stock_only=request.filters.in_stock_only,
            exclude_purchased=request.filters.exclude_purchased
        )

        logger.info(f"Generated {len(recommendations)} recommendations for {customer_id}")

        # Build response
        response = RecommendationResponse(
            customer_id=customer_id,
            recommendations=[
                ProductRecommendation(**rec_dict)
                for rec_dict in recommendations
            ],
            metadata=RecommendationMetadata(
                model_version="v1",
                generated_at=datetime.now(),
                num_results=len(recommendations),
                filtered=request.filters
            )
        )

        return response

    except Exception as e:
        logger.error(f"Error generating recommendations for {customer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@router.get(
    "/health",
    summary="Health Check",
    description="Check if the recommendation service is running and model is loaded"
)
async def health_check():
    """Health check endpoint"""
    try:
        rec = get_recommender()
        return {
            "status": "healthy",
            "model_loaded": rec.model is not None,
            "num_users": len(rec.user_id_map),
            "num_items": len(rec.item_id_map)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
