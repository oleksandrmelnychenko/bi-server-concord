"""
Configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Environment
    ENVIRONMENT: str = "development"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # Database
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "concord_bi"
    POSTGRES_USER: str = "concord"
    POSTGRES_PASSWORD: str = "change-me-in-production"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"

    # LLM Configuration
    LLM_MODEL: str = "llama3.1:8b"
    LLM_ENDPOINT: str = "http://ollama:11434"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # Model Configuration
    RECOMMENDATION_MODEL_NAME: str = "product-recommendation"
    RECOMMENDATION_MODEL_VERSION: str = "latest"

    FORECASTING_MODEL_NAME: str = "sales-forecasting"
    FORECASTING_MODEL_VERSION: str = "latest"

    # Cache settings
    CACHE_TTL_SECONDS: int = 3600  # 1 hour

    # Feature flags
    ENABLE_CACHING: bool = True
    ENABLE_MONITORING: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
