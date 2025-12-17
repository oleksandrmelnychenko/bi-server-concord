import os
import logging
import pymssql
from queue import LifoQueue
from threading import Lock

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', os.environ.get('DB_HOST', '78.152.175.67')),
    'port': int(os.environ.get('MSSQL_PORT', os.environ.get('DB_PORT', '1433'))),
    'database': os.environ.get('MSSQL_DATABASE', os.environ.get('DB_NAME', 'ConcordDb_v5')),
    'user': os.environ.get('MSSQL_USER', os.environ.get('DB_USER', 'ef_migrator')),
    'password': os.environ.get('MSSQL_PASSWORD', os.environ.get('DB_PASSWORD', 'Grimm_jow92')),
}

# Lightweight pymssql pool to avoid ODBC dependency
class SimpleConnectionPool:
    def __init__(self, maxsize=20):
        self._maxsize = maxsize
        self._pool = LifoQueue(maxsize)
        self._lock = Lock()

    def _create_conn(self):
        return pymssql.connect(
            server=DB_CONFIG['server'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            as_dict=True,
            timeout=30,
            login_timeout=10,
        )

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
logger.info(f"Database connection pool (pymssql) created (maxsize=20)")

def get_connection():
    return pool.get()

def close_pool():
    pool.closeall()
    logger.info("Database connection pool closed")
