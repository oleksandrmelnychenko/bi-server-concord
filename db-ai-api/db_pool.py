"""Database connection pool for MSSQL using SQLAlchemy.

This module provides a shared connection pool to avoid creating
new connections for each database operation.
"""
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from loguru import logger

from config import settings


class DatabasePool:
    """Singleton database connection pool manager."""

    _engine: Optional[Engine] = None

    @classmethod
    def get_engine(cls) -> Engine:
        """Get or create the shared SQLAlchemy engine.

        Returns:
            SQLAlchemy Engine with connection pooling
        """
        if cls._engine is None:
            logger.info("Creating database connection pool...")
            cls._engine = create_engine(
                settings.database_url,
                poolclass=QueuePool,
                pool_size=5,           # 5 persistent connections
                max_overflow=10,       # Up to 15 total
                pool_pre_ping=False,   # Disable ping (use pool_recycle)
                pool_recycle=1800,     # Recycle after 30 min
                pool_timeout=30,       # Wait 30s for connection
            )
            logger.info(f"Database pool created for: {settings.db_name}")

        return cls._engine

    @classmethod
    def get_connection(cls):
        """Get a connection from the pool.

        Returns:
            SQLAlchemy connection from the pool

        Usage:
            with DatabasePool.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
        """
        return cls.get_engine().connect()

    @classmethod
    def dispose(cls):
        """Dispose of the connection pool.

        Call this when shutting down the application.
        """
        if cls._engine:
            cls._engine.dispose()
            cls._engine = None
            logger.info("Database pool disposed")

    @classmethod
    def get_stats(cls) -> dict:
        """Get pool statistics.

        Returns:
            Dictionary with pool stats
        """
        if cls._engine is None:
            return {"initialized": False}

        pool = cls._engine.pool
        return {
            "initialized": True,
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checkedin": pool.checkedin(),
        }


# Convenience function for backward compatibility
def get_engine() -> Engine:
    """Get the shared database engine."""
    return DatabasePool.get_engine()


def get_connection():
    """Get a connection from the pool."""
    return DatabasePool.get_connection()
