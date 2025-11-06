"""
Sales forecasting endpoints
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional
import logging

from app.services.forecasting_service import ForecastingService
from app.schemas.forecasting import (
    ForecastRequest,
    ForecastResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_forecasting_service() -> ForecastingService:
    """Dependency injection for forecasting service"""
    return ForecastingService()


@router.post("/predict", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    service: ForecastingService = Depends(get_forecasting_service)
) -> ForecastResponse:
    """
    Generate sales forecast for specified period

    Args:
        request: Forecast request with parameters

    Returns:
        Forecast data with predictions and confidence intervals
    """
    try:
        logger.info(f"Generating forecast for {request.periods} periods")

        forecast_data = await service.generate_forecast(
            periods=request.periods,
            product_id=request.product_id,
            category=request.category,
            include_confidence=request.include_confidence,
            confidence_level=request.confidence_level
        )

        return forecast_data

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating forecast: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate forecast"
        )


@router.get("/model/info")
async def get_model_info(
    service: ForecastingService = Depends(get_forecasting_service)
) -> dict:
    """Get information about the current forecasting model"""
    try:
        return await service.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )
