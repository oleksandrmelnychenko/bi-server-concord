"""
Product recommendation endpoints
"""
from fastapi import APIRouter, HTTPException, status, Depends, Body
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

from app.services.recommendation_service import RecommendationService
from app.schemas.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
    ProductRecommendation
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_recommendation_service() -> RecommendationService:
    """Dependency injection for recommendation service"""
    return RecommendationService()


@router.post("/predict", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
) -> RecommendationResponse:
    """
    Get personalized product recommendations for a customer

    Args:
        request: Recommendation request with customer_id and optional filters

    Returns:
        List of recommended products with scores
    """
    try:
        logger.info(f"Getting recommendations for customer: {request.customer_id}")

        recommendations = await service.get_recommendations(
            customer_id=request.customer_id,
            n_recommendations=request.n_recommendations,
            exclude_purchased=request.exclude_purchased,
            category_filter=request.category_filter
        )

        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=recommendations,
            model_version=service.model_version,
            generated_at=service.get_current_timestamp()
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


@router.post("/batch", response_model=List[RecommendationResponse])
async def get_batch_recommendations(
    customer_ids: List[str] = Body(..., min_items=1, max_items=100),
    n_recommendations: int = Body(10, ge=1, le=100),
    service: RecommendationService = Depends(get_recommendation_service)
) -> List[RecommendationResponse]:
    """
    Get recommendations for multiple customers in batch

    Args:
        customer_ids: List of customer IDs (max 100)
        n_recommendations: Number of recommendations per customer

    Returns:
        List of recommendation responses for each customer
    """
    try:
        logger.info(f"Getting batch recommendations for {len(customer_ids)} customers")

        results = []
        for customer_id in customer_ids:
            try:
                recommendations = await service.get_recommendations(
                    customer_id=customer_id,
                    n_recommendations=n_recommendations
                )
                results.append(
                    RecommendationResponse(
                        customer_id=customer_id,
                        recommendations=recommendations,
                        model_version=service.model_version,
                        generated_at=service.get_current_timestamp()
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to get recommendations for customer {customer_id}: {e}")
                # Continue with other customers

        return results

    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate batch recommendations"
        )


@router.get("/model/info")
async def get_model_info(
    service: RecommendationService = Depends(get_recommendation_service)
) -> dict:
    """Get information about the current recommendation model"""
    try:
        return await service.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )
