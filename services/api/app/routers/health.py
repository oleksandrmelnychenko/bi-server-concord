"""
Health check endpoints
"""
from fastapi import APIRouter, status
from typing import Dict, Any
import redis
import psycopg2
from datetime import datetime
import logging

from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Concord BI Server API",
        "version": "1.0.0"
    }


@router.get("/health/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check including dependencies"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Concord BI Server API",
        "version": "1.0.0",
        "dependencies": {}
    }

    # Check PostgreSQL
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            connect_timeout=5
        )
        conn.close()
        health_status["dependencies"]["postgres"] = {
            "status": "healthy",
            "message": "Connected successfully"
        }
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        health_status["dependencies"]["postgres"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        health_status["status"] = "degraded"

    # Check Redis
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_connect_timeout=5
        )
        r.ping()
        health_status["dependencies"]["redis"] = {
            "status": "healthy",
            "message": "Connected successfully"
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["dependencies"]["redis"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        health_status["status"] = "degraded"

    # Check MLflow
    try:
        import requests
        response = requests.get(f"{settings.MLFLOW_TRACKING_URI}/health", timeout=5)
        if response.status_code == 200:
            health_status["dependencies"]["mlflow"] = {
                "status": "healthy",
                "message": "Connected successfully"
            }
        else:
            health_status["dependencies"]["mlflow"] = {
                "status": "unhealthy",
                "message": f"Status code: {response.status_code}"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"MLflow health check failed: {e}")
        health_status["dependencies"]["mlflow"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        health_status["status"] = "degraded"

    return health_status


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, str]:
    """Kubernetes readiness probe"""
    return {"status": "ready"}


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe"""
    return {"status": "alive"}
