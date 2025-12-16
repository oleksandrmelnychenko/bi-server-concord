"""Main entry point for DB AI API."""
import sys
from loguru import logger
import uvicorn

from config import settings


# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level,
)

# Also log to file
logger.add(
    "logs/db-ai-api.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.log_level,
)


def main():
    """Start the API server."""
    logger.info("Starting DB AI API...")
    logger.info(f"Database: {settings.db_name}")
    logger.info(f"Model: {settings.ollama_model}")
    logger.info(f"Read-only mode: {settings.read_only_mode}")

    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
