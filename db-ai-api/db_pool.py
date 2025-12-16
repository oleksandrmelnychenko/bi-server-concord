"""Database connection pool for MSSQL."""
import pymssql

# Database configuration
DB_HOST = '78.152.175.67'
DB_PORT = 1433
DB_USER = 'ef_migrator'
DB_PASSWORD = 'Grimm_jow92'
DB_NAME = 'ConcordDb_v5'


def get_connection():
    """
    Create and return a database connection.
    
    Returns:
        pymssql.Connection: Database connection
    """
    return pymssql.connect(
        server=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        timeout=30
    )
