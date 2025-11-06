"""
Natural Language Query endpoints
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional
import logging

from app.services.nlq_service import NLQService
from app.schemas.nlq import (
    NLQueryRequest,
    NLQueryResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_nlq_service() -> NLQService:
    """Dependency injection for NLQ service"""
    return NLQService()


@router.post("/query", response_model=NLQueryResponse)
async def execute_natural_language_query(
    request: NLQueryRequest,
    service: NLQService = Depends(get_nlq_service)
) -> NLQueryResponse:
    """
    Execute a natural language query against the database

    Args:
        request: Natural language query request

    Returns:
        Query results with generated SQL and data
    """
    try:
        logger.info(f"Executing NL query: {request.query[:100]}...")

        result = await service.execute_query(
            query=request.query,
            return_sql=request.return_sql,
            max_rows=request.max_rows
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error executing NL query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute natural language query"
        )


@router.get("/schema/summary")
async def get_schema_summary(
    service: NLQService = Depends(get_nlq_service)
) -> dict:
    """Get a summary of available database schema for querying"""
    try:
        return await service.get_schema_summary()
    except Exception as e:
        logger.error(f"Error getting schema summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve schema summary"
        )
