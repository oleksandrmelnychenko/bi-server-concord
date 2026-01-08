import os
import logging
import pyodbc
from queue import LifoQueue
from threading import Lock
from pathlib import Path

# Load .env file from api directory if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', os.environ.get('DB_HOST', 'localhost')),
    'port': int(os.environ.get('MSSQL_PORT', os.environ.get('DB_PORT', '1433'))),
    'database': os.environ.get('MSSQL_DATABASE', os.environ.get('DB_NAME', 'ConcordDb_v5')),
    'driver': os.environ.get('DB_DRIVER', 'ODBC Driver 17 for SQL Server'),
    'trusted_connection': os.environ.get('DB_TRUSTED_CONNECTION', 'no').lower() in ('yes', 'true', '1'),
    # Credentials required via env vars if not using Windows Auth (DB_TRUSTED_CONNECTION=yes)
    'user': os.environ.get('MSSQL_USER', os.environ.get('DB_USER', '')),
    'password': os.environ.get('MSSQL_PASSWORD', os.environ.get('DB_PASSWORD', '')),
}

# Connection pool using pyodbc with Windows Auth support
class SimpleConnectionPool:
    def __init__(self, maxsize=20):
        self._maxsize = maxsize
        self._pool = LifoQueue(maxsize)
        self._lock = Lock()

    def _create_conn(self):
        # Only include port if non-default
        server = DB_CONFIG['server']
        if DB_CONFIG['port'] != 1433:
            server = f"{server},{DB_CONFIG['port']}"

        if DB_CONFIG['trusted_connection']:
            conn_str = (
                f"DRIVER={{{DB_CONFIG['driver']}}};"
                f"SERVER={server};"
                f"DATABASE={DB_CONFIG['database']};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{{DB_CONFIG['driver']}}};"
                f"SERVER={server};"
                f"DATABASE={DB_CONFIG['database']};"
                f"UID={DB_CONFIG['user']};"
                f"PWD={DB_CONFIG['password']};"
                f"TrustServerCertificate=yes;"
            )
        return pyodbc.connect(conn_str, timeout=30)

    def get(self):
        try:
            conn = self._pool.get_nowait()
            return conn
        except Exception:
            with self._lock:
                if self._pool.qsize() + 1 <= self._maxsize:
                    return self._create_conn()
                # If pool is exhausted, block briefly
            return self._pool.get()

    def put(self, conn):
        try:
            self._pool.put_nowait(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    def closeall(self):
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass

pool = SimpleConnectionPool(maxsize=20)
logger.info(f"Database connection pool (pyodbc) created (maxsize=20)")

def get_connection():
    return pool.get()

def close_pool():
    pool.closeall()
    logger.info("Database connection pool closed")
