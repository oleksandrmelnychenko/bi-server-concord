"""Main entry point for the AI Provider API."""
import sys
from pathlib import Path
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
log_path = Path("logs/ai-provider.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(log_path, rotation="500 MB", retention="10 days", level=settings.log_level)


def main():
    """Start the API server."""
    logger.info("Starting AI Provider API...")
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
