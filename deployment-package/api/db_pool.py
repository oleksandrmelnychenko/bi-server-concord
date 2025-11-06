"""
Database Connection Pool

Provides thread-safe connection pooling for concurrent API requests.
Replaces single connection bottleneck with pool of 20 connections.
"""

import os
import logging
from sqlalchemy import create_engine, pool

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
}

# Build connection string for SQLAlchemy
DATABASE_URL = (
    f"mssql+pymssql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['server']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Create connection pool
# pool_size=20: Number of persistent connections
# max_overflow=10: Additional connections during spikes (total 30 max)
# pool_pre_ping=True: Verify connection is alive before using
# pool_recycle=3600: Recycle connections every hour to avoid stale connections
engine = create_engine(
    DATABASE_URL,
    poolclass=pool.QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL query logging (debugging)
)

logger.info(f"Database connection pool created (pool_size=20, max_overflow=10)")


def get_connection():
    """
    Get a raw database connection from the pool.

    This connection is compatible with pymssql-style cursor operations.
    The connection MUST be closed when done to return it to the pool.

    Returns:
        pymssql-compatible connection object

    Example:
        conn = get_connection()
        try:
            cursor = conn.cursor(as_dict=True)
            cursor.execute("SELECT * FROM table")
            rows = cursor.fetchall()
        finally:
            conn.close()  # Return to pool
    """
    return engine.raw_connection()


def close_pool():
    """Close all connections in the pool (for shutdown)"""
    engine.dispose()
    logger.info("Database connection pool closed")
