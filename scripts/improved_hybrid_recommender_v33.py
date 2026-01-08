#!/usr/bin/env python3

import os
import json
import math
import pyodbc
import logging
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from contextlib import contextmanager

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
}

REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
SIMILAR_CUSTOMERS_CACHE_TTL = 86400

MAX_SIMILAR_CUSTOMERS = 100
MIN_SIMILARITY_THRESHOLD = 0.05
COLD_START_LOOKBACK_DAYS = 90
MIN_CO_PURCHASE_ORDERS = 3
MIN_CYCLE_PURCHASES = 3
COLLAB_SPARSE_CUSTOMERS = 5
COLLAB_SPARSE_PRODUCTS = 10
MAX_CANDIDATES = 1000
ANN_NEIGHBORS_PATH = os.getenv('ANN_NEIGHBORS_PATH')  # JSON file containing {product_id: [{\"product_id\": int, \"score\": float}, ...]}
ANN_NEIGHBORS_TOPK = int(os.getenv('ANN_NEIGHBORS_TOPK', '200'))

# =============================================================================
# SCORING CONSTANTS
# =============================================================================

# Recency decay: score = e^(-days_ago / RECENCY_DECAY_DAYS)
# 90 days = ~37% score remaining after 90 days (half-life behavior)
RECENCY_DECAY_DAYS = 90

# Co-purchase analysis window (days to look back for co-occurrence patterns)
CO_PURCHASE_WINDOW_DAYS = 180

# Cycle detection constants
MIN_CYCLE_DAYS = 3            # Cycles shorter than this are ignored (likely noise)
CYCLE_TOLERANCE_MIN_DAYS = 7  # Minimum tolerance window for cycle detection

# Diversity filter: max products per product group
MAX_PRODUCTS_PER_GROUP = 3

# =============================================================================
# SEGMENT-SPECIFIC WEIGHT CONFIGURATIONS
# =============================================================================
# V3.3: 5-component scoring weights (frequency, recency, co_purchase, cycle, collaborative)
# These weights sum to 1.0 and control how each signal influences final score

SEGMENT_WEIGHTS = {
    'HEAVY': {
        'frequency': 0.30,
        'recency': 0.20,
        'co_purchase': 0.25,
        'cycle': 0.15,
        'collaborative': 0.10
    },
    'REGULAR': {
        'frequency': 0.30,
        'recency': 0.20,
        'co_purchase': 0.25,
        'cycle': 0.15,
        'collaborative': 0.10
    },
    'LIGHT': {
        'frequency': 0.35,
        'recency': 0.20,
        'co_purchase': 0.25,
        'cycle': 0.10,
        'collaborative': 0.10
    }
}

# Collaborative filtering adaptive weight factors
COLLAB_WEIGHT_SPARSE = 0.5   # Weight multiplier when sparse signals
COLLAB_WEIGHT_FULL = 1.0     # Weight multiplier when sufficient signals

# Cycle score ranges
CYCLE_SCORE_DUE_MAX = 1.0       # Max score when product is due
CYCLE_SCORE_DUE_MIN = 0.6       # Min score in "due" range
CYCLE_SCORE_OVERDUE_MAX = 0.6   # Max score when overdue
CYCLE_SCORE_OVERDUE_MIN = 0.3   # Min score when overdue
CYCLE_SCORE_EARLY_MAX = 0.2     # Max score when too early

# Customer segment thresholds (order counts)
SEGMENT_THRESHOLD_HEAVY = 500
SEGMENT_THRESHOLD_REGULAR = 100
REPURCHASE_RATE_CONSISTENT = 0.40  # Rate to classify as CONSISTENT subsegment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedHybridRecommenderV33:
    """
    V3.3: Quick Wins - B2B-specific improvements
    - Co-purchase scoring (products bought together)
    - Purchase cycle detection (reorder timing)
    - Enhanced ranking formula

    Expected: +10-15% precision improvement over V3.2
    """

    def __init__(self, conn=None, use_cache=True):
        if conn:
            self.conn = conn
            self.owns_connection = False
            logger.debug("Using provided database connection")
        else:
            self.conn = None
            self.owns_connection = True
            self._connect()

        # Detect connection type for SQL placeholder style
        # pyodbc uses '?', pymssql uses '%s'
        conn_module = type(self.conn).__module__
        self._use_pymssql = 'pymssql' in conn_module
        logger.debug(f"SQL placeholder style: {'%s' if self._use_pymssql else '?'}")

        self.redis_client = None
        self.use_cache = use_cache
        if use_cache:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    decode_responses=False,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
                logger.debug(f"Connected to Redis for caching")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Running without cache.")
                self.redis_client = None

        self.ann_index = self._load_ann_neighbors()

    def _connect(self):
        try:
            conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={DB_CONFIG['server']},{DB_CONFIG['port']};"
                f"DATABASE={DB_CONFIG['database']};"
                f"UID={DB_CONFIG['user']};"
                f"PWD={DB_CONFIG['password']};"
                f"TrustServerCertificate=yes;"
            )
            self.conn = pyodbc.connect(conn_str)
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    class DictCursorWrapper:
        def __init__(self, cursor):
            self._cursor = cursor
            self._columns = None

        def _ensure_columns(self):
            if self._columns is None and self._cursor.description:
                self._columns = [col[0] for col in self._cursor.description]

        def execute(self, *args, **kwargs):
            return self._cursor.execute(*args, **kwargs)

        def fetchone(self):
            row = self._cursor.fetchone()
            if row is None:
                return None
            if isinstance(row, dict):
                return row
            self._ensure_columns()
            return {col: row[idx] for idx, col in enumerate(self._columns)}

        def fetchall(self):
            rows = self._cursor.fetchall()
            if not rows:
                return []
            if isinstance(rows[0], dict):
                return rows
            self._ensure_columns()
            return [
                {col: row[idx] for idx, col in enumerate(self._columns)}
                for row in rows
            ]

        def __iter__(self):
            for row in self.fetchall():
                yield row

        def close(self):
            return self._cursor.close()

    def _dict_cursor(self):
        """Create a cursor that yields dictionaries even for pyodbc"""
        try:
            cursor = self.conn.cursor(as_dict=True)
            return cursor
        except Exception:
            return self.DictCursorWrapper(self.conn.cursor())

    def _sql(self, query: str) -> str:
        """
        Convert SQL placeholder style based on connection type.
        pyodbc uses '?', pymssql uses '%s'.

        Args:
            query: SQL query with '?' placeholders

        Returns:
            Query with placeholders converted for the active connection
        """
        if self._use_pymssql:
            return query.replace('?', '%s')
        return query

    @contextmanager
    def _cursor_context(self):
        """
        Context manager for cursor handling.
        Ensures cursors are properly closed after use.

        Usage:
            with self._cursor_context() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
            # cursor automatically closed here
        """
        cursor = self._dict_cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _load_ann_neighbors(self) -> Optional[Dict[int, List[Tuple[int, float]]]]:
        """
        Load precomputed ANN neighbors from JSON file.
        Expected format: {product_id: [{"product_id": int, "score": float}, ...]}
        """
        if not ANN_NEIGHBORS_PATH:
            logger.warning("ANN_NEIGHBORS_PATH not set; collaborative ANN disabled")
            return None
        if not os.path.exists(ANN_NEIGHBORS_PATH):
            logger.error(f"ANN neighbors file not found: {ANN_NEIGHBORS_PATH}")
            return None
        try:
            with open(ANN_NEIGHBORS_PATH, "r") as f:
                data = json.load(f)
            index = {}
            for pid, neighbors in data.items():
                try:
                    pid_int = int(pid)
                except Exception:
                    continue
                index[pid_int] = [
                    (int(n["product_id"]), float(n.get("score", 0.0)))
                    for n in neighbors[:ANN_NEIGHBORS_TOPK]
                ]
            logger.info(f"Loaded ANN neighbors for {len(index)} products from {ANN_NEIGHBORS_PATH}")
            return index
        except Exception as e:
            logger.error(f"Failed to load ANN neighbors: {e}")
            return None

    @staticmethod
    def _normalize_date(as_of_date: str) -> str:
        """
        Ensure we only ever send ISO dates to SQL to avoid injection and mixed formats.
        """
        if isinstance(as_of_date, datetime):
            return as_of_date.strftime('%Y-%m-%d')
        try:
            return datetime.strptime(as_of_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except Exception as e:
            raise ValueError(f"Invalid as_of_date format (expected YYYY-MM-DD): {as_of_date}") from e

    def _get_sellable_product_ids(self, product_ids: List[int]) -> Set[int]:
        """
        Filter product IDs to only include sellable products (IsForSale=1, Deleted=0).
        """
        if not product_ids:
            return set()

        # Use string interpolation for IDs (safe - integers only)
        ids_str = ','.join(str(int(pid)) for pid in product_ids)
        query = f"""
        SELECT ID FROM dbo.Product
        WHERE ID IN ({ids_str})
          AND IsForSale = 1
          AND Deleted = 0
        """

        with self._cursor_context() as cursor:
            cursor.execute(query)
            sellable = {row['ID'] for row in cursor}
        return sellable

    def classify_customer(self, customer_id: int, as_of_date: str) -> Tuple[str, str]:

        as_of_date = self._normalize_date(as_of_date)

        query = """
        SELECT COUNT(DISTINCT o.ID) as orders_before
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE ca.ClientID = ?
              AND o.Created < ?
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (customer_id, as_of_date))
            row = cursor.fetchone()
            orders_before = row['orders_before'] if row else 0

        if orders_before >= SEGMENT_THRESHOLD_HEAVY:
            segment = "HEAVY"
            subsegment = None
        elif orders_before >= SEGMENT_THRESHOLD_REGULAR:
            segment = "REGULAR"

            repurchase_query = """
            SELECT
                COUNT(DISTINCT oi.ProductID) as total_products,
                COUNT(DISTINCT CASE WHEN purchase_count >= 2 THEN oi.ProductID END) as repurchased_products
            FROM (
                SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
                FROM dbo.ClientAgreement ca
                INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
                INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
                WHERE ca.ClientID = ?
                      AND o.Created < ?
                      AND oi.ProductID IS NOT NULL
                GROUP BY oi.ProductID
            ) AS counts
            LEFT JOIN dbo.OrderItem oi ON oi.ProductID = counts.ProductID
            """

            with self._cursor_context() as cursor:
                cursor.execute(self._sql(repurchase_query), (customer_id, as_of_date))
                repurchase_row = cursor.fetchone()

            if repurchase_row and repurchase_row['total_products'] > 0:
                repurchase_rate = repurchase_row['repurchased_products'] / repurchase_row['total_products']
                subsegment = "CONSISTENT" if repurchase_rate >= REPURCHASE_RATE_CONSISTENT else "EXPLORATORY"
            else:
                subsegment = "EXPLORATORY"  # Default to exploratory if no data
        else:
            segment = "LIGHT"
            subsegment = None

        return segment, subsegment

    def get_customer_products(self, customer_id: int, as_of_date: str, limit: int = 500) -> Set[int]:

        as_of_date = self._normalize_date(as_of_date)

        query = f"""
        SELECT DISTINCT ProductID
        FROM (
            SELECT TOP {limit} oi.ProductID, o.Created
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE ca.ClientID = ?
                  AND o.Created < ?
                  AND oi.ProductID IS NOT NULL
            ORDER BY o.Created DESC
        ) AS RecentProducts
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (customer_id, as_of_date))
            products = {row['ProductID'] for row in cursor}

        logger.debug(f"Customer {customer_id} has {len(products)} products (limit: {limit})")
        return products

    def find_similar_customers(self, customer_id: int, as_of_date: str, limit: int = MAX_SIMILAR_CUSTOMERS) -> List[Tuple[int, float]]:

        as_of_date = self._normalize_date(as_of_date)

        if self.redis_client:
            cache_key = f"similar_customers:{customer_id}:{as_of_date}"
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"Cache HIT: Similar customers for {customer_id}")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        target_products = self.get_customer_products(customer_id, as_of_date)

        if not target_products:
            logger.warning(f"Customer {customer_id} has no purchase history")
            return []

        # Use string interpolation for product IDs (safe - integers only)
        product_ids_str = ','.join(str(int(pid)) for pid in target_products)

        query = f"""
        SELECT DISTINCT ca.ClientID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID != ?
              AND oi.ProductID IN ({product_ids_str})
              AND o.Created < ?
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (customer_id, as_of_date))
            candidate_customers = [row['ClientID'] for row in cursor]

        logger.debug(f"Found {len(candidate_customers)} candidate similar customers")

        similarities = []

        for candidate_id in candidate_customers:
            candidate_products = self.get_customer_products(candidate_id, as_of_date)

            intersection = len(target_products & candidate_products)
            union = len(target_products | candidate_products)

            if union > 0:
                similarity = intersection / union

                if similarity >= MIN_SIMILARITY_THRESHOLD:
                    similarities.append((candidate_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_customers = similarities[:limit]

        logger.debug(f"Found {len(similar_customers)} similar customers (threshold: {MIN_SIMILARITY_THRESHOLD})")

        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    SIMILAR_CUSTOMERS_CACHE_TTL,
                    json.dumps(similar_customers)
                )
                logger.debug(f"Cached similar customers for {customer_id}")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

        return similar_customers

    def get_collaborative_score(self, customer_id: int, as_of_date: str) -> Tuple[Dict[int, float], int]:
        """
        ANN-based collaborative score using precomputed item neighbors. No SQL fallback.
        """
        if not self.ann_index:
            logger.warning("ANN index not loaded; collaborative score disabled")
            return {}, 0

        as_of_date = self._normalize_date(as_of_date)
        owned_products = self.get_customer_products(customer_id, as_of_date, limit=5000)
        if not owned_products:
            return {}, 0

        candidate_scores = defaultdict(float)
        for pid in owned_products:
            neighbors = self.ann_index.get(pid, [])
            for nbr_id, sim in neighbors:
                if nbr_id in owned_products:
                    continue
                candidate_scores[nbr_id] += sim

        if not candidate_scores:
            return {}, len(owned_products)

        sellable_ids = self._get_sellable_product_ids(list(candidate_scores.keys()))
        collaborative_scores = {
            pid: float(score) for pid, score in candidate_scores.items() if pid in sellable_ids
        }

        return collaborative_scores, len(owned_products)

    def get_frequency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """
        Utility method: Get frequency scores for a customer across ALL their agreements.

        Note: This is a customer-level utility method for analysis/debugging.
        The main recommendation flow uses agreement-level scoring in get_recommendations_for_agreement().

        Args:
            customer_id: Customer ID to analyze
            as_of_date: Cutoff date (YYYY-MM-DD)

        Returns:
            Dict mapping product_id -> normalized frequency score (0.0-1.0)
        """
        as_of_date = self._normalize_date(as_of_date)

        query = """
        SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = ?
              AND o.Created < ?
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (customer_id, as_of_date))

            scores = {}
            max_count = 1
            for row in cursor:
                scores[row['ProductID']] = row['purchase_count']
                max_count = max(max_count, row['purchase_count'])

        scores = {pid: count / max_count for pid, count in scores.items()}
        return scores

    def get_recency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """
        Utility method: Get recency scores for a customer across ALL their agreements.

        Note: This is a customer-level utility method for analysis/debugging.
        The main recommendation flow uses agreement-level scoring in get_recommendations_for_agreement().

        Uses exponential decay: score = e^(-days_ago / RECENCY_DECAY_DAYS)

        Args:
            customer_id: Customer ID to analyze
            as_of_date: Cutoff date (YYYY-MM-DD)

        Returns:
            Dict mapping product_id -> recency score (0.0-1.0)
        """
        as_of_date = self._normalize_date(as_of_date)

        query = """
        SELECT oi.ProductID, MAX(o.Created) as last_purchase
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = ?
              AND o.Created < ?
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (customer_id, as_of_date))

            as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
            scores = {}

            for row in cursor:
                last_purchase = row['last_purchase']
                days_ago = (as_of_datetime - last_purchase).days
                score = math.e ** (-days_ago / RECENCY_DECAY_DAYS)
                scores[row['ProductID']] = score

        return scores

    def get_product_groups(self, product_ids: List[int]) -> Dict[int, int]:
        if not product_ids:
            return {}

        # Use string interpolation for product IDs (safe - integers only)
        ids_str = ','.join(str(int(pid)) for pid in product_ids)

        query = f"""
        SELECT ProductID, ProductGroupID
        FROM dbo.ProductProductGroup
        WHERE ProductID IN ({ids_str})
              AND Deleted = 0
        """

        with self._cursor_context() as cursor:
            cursor.execute(query)
            groups = {}
            for row in cursor:
                groups[row['ProductID']] = row['ProductGroupID']

        logger.debug(f"Found product groups for {len(groups)}/{len(product_ids)} products")
        return groups

    def apply_diversity_filter(self, recommendations: List[Dict], max_per_group: int = MAX_PRODUCTS_PER_GROUP) -> List[Dict]:
        if not recommendations:
            return recommendations

        product_ids = [r['product_id'] for r in recommendations]
        groups = self.get_product_groups(product_ids)

        group_counts = defaultdict(int)
        filtered = []

        for rec in recommendations:
            group_id = groups.get(rec['product_id'])

            if group_id is None or group_counts[group_id] < max_per_group:
                filtered.append(rec)
                if group_id is not None:
                    group_counts[group_id] += 1

        for idx, rec in enumerate(filtered):
            rec['rank'] = idx + 1

        logger.debug(f"Diversity filter: {len(recommendations)} → {len(filtered)} products " +
                    f"(max {max_per_group} per group, {len(group_counts)} groups)")

        return filtered

    def get_customer_agreements(self, customer_id: int) -> List[int]:
        query = """
        SELECT ID as AgreementID
        FROM dbo.ClientAgreement
        WHERE ClientID = ?
              AND Deleted = 0
        ORDER BY Created DESC
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (customer_id,))
            agreement_ids = [row['AgreementID'] for row in cursor]

        logger.debug(f"Customer {customer_id} has {len(agreement_ids)} agreements")
        return agreement_ids

    def get_co_purchase_scores(self, agreement_id: int, product_ids: List[int],
                               as_of_date: str, days_window: int = CO_PURCHASE_WINDOW_DAYS) -> Dict[int, float]:
        """
        V3.3 FEATURE: Co-purchase scoring
        Products frequently bought TOGETHER in the same order get higher scores.

        Logic:
        - For each candidate product, find how often it appears in same orders
          as products the agreement already bought
        - Higher co-occurrence = higher score

        Args:
            agreement_id: The agreement to analyze
            product_ids: Candidate products to score
            as_of_date: Cutoff date
            days_window: Look back this many days for co-purchase patterns

        Returns:
            Dict of {product_id: co_purchase_score (0.0-1.0)}
        """
        if not product_ids:
            return {}

        as_of_date = self._normalize_date(as_of_date)
        # Use string interpolation for product IDs (safe - integers only)
        product_id_list = ','.join(str(int(pid)) for pid in product_ids)

        query = f"""
        WITH AgreementProducts AS (
            -- Products this agreement has purchased
            SELECT DISTINCT oi.ProductID
            FROM dbo.[Order] o
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.ClientAgreementID = ?
                  AND o.Created < ?
                  AND o.Created >= DATEADD(day, -?, ?)
        ),
        CoOccurrence AS (
            -- For candidate products, count co-occurrences with agreement's products
            SELECT
                oi_candidate.ProductID as CandidateProduct,
                COUNT(DISTINCT o.ID) as CoOccurrenceCount
            FROM dbo.[Order] o
            INNER JOIN dbo.OrderItem oi_candidate ON o.ID = oi_candidate.OrderID
            INNER JOIN dbo.OrderItem oi_existing ON o.ID = oi_existing.OrderID
            INNER JOIN AgreementProducts ap ON oi_existing.ProductID = ap.ProductID
            WHERE o.ClientAgreementID = ?
                  AND o.Created < ?
                  AND o.Created >= DATEADD(day, -?, ?)
                  AND oi_candidate.ProductID IN ({product_id_list})
                  AND oi_candidate.ProductID != oi_existing.ProductID
            GROUP BY oi_candidate.ProductID
        ),
        TotalOrders AS (
            SELECT COUNT(DISTINCT ID) as total FROM dbo.[Order]
            WHERE ClientAgreementID = ?
                  AND Created < ?
                  AND Created >= DATEADD(day, -?, ?)
        )
        SELECT
            co.CandidateProduct as ProductID,
            CASE
                WHEN t.total > 0 THEN co.CoOccurrenceCount * 1.0 / t.total
                ELSE 0.0
            END as co_purchase_rate
        FROM CoOccurrence co
        CROSS JOIN TotalOrders t
        """

        params = (
            agreement_id, as_of_date, days_window, as_of_date,  # AgreementProducts
            agreement_id, as_of_date, days_window, as_of_date,  # CoOccurrence
            agreement_id, as_of_date, days_window, as_of_date   # TotalOrders
        )

        # First check sample size to avoid noisy boosts
        with self._cursor_context() as cursor:
            cursor.execute(
                self._sql("""
                SELECT COUNT(DISTINCT ID) as total_orders
                FROM dbo.[Order]
                WHERE ClientAgreementID = ?
                  AND Created < ?
                  AND Created >= DATEADD(day, -?, ?)
                """),
                (agreement_id, as_of_date, days_window, as_of_date)
            )
            row = cursor.fetchone()
            total_orders = row['total_orders'] if row else 0

        if total_orders < MIN_CO_PURCHASE_ORDERS:
            logger.debug(f"Co-purchase suppressed: only {total_orders} orders in window")
            return {}

        # Now fetch co-purchase scores
        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), params)
            scores = {}
            for row in cursor:
                scores[row['ProductID']] = float(row['co_purchase_rate'])

        logger.debug(f"Co-purchase scores: {len(scores)} products analyzed")
        return scores

    def get_cycle_scores(self, agreement_id: int, product_ids: List[int],
                         as_of_date: str) -> Dict[int, float]:
        """
        V3.3 FEATURE: Purchase cycle detection
        Products with predictable purchase cycles get boosted when they're "due".

        Logic:
        - Calculate average days between purchases for each product
        - If product is "due" (last purchase + avg cycle ≈ today), boost it
        - If overdue, boost even more

        Args:
            agreement_id: The agreement to analyze
            product_ids: Candidate products to score
            as_of_date: Current date for cycle calculation

        Returns:
            Dict of {product_id: cycle_score (0.0-1.0)}
        """
        if not product_ids:
            return {}

        as_of_date = self._normalize_date(as_of_date)
        # Use string interpolation for product IDs (safe - integers only)
        product_id_list = ','.join(str(int(pid)) for pid in product_ids)

        query = f"""
        WITH PurchaseDates AS (
            SELECT
                oi.ProductID,
                o.Created,
                LAG(o.Created) OVER (PARTITION BY oi.ProductID ORDER BY o.Created) as PrevPurchase
            FROM dbo.[Order] o
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.ClientAgreementID = ?
                  AND o.Created < ?
                  AND oi.ProductID IN ({product_id_list})
        ),
        CycleStats AS (
            SELECT
                ProductID,
                AVG(DATEDIFF(day, PrevPurchase, Created)) as avg_cycle_days,
                STDEV(DATEDIFF(day, PrevPurchase, Created)) as cycle_variance,
                MAX(Created) as last_purchase_date,
            COUNT(*) as purchase_count
        FROM PurchaseDates
        WHERE PrevPurchase IS NOT NULL
        GROUP BY ProductID
        HAVING COUNT(*) >= {MIN_CYCLE_PURCHASES}  -- Need sufficient purchases to detect cycle
        )
        SELECT
            ProductID,
            avg_cycle_days,
            cycle_variance,
            DATEDIFF(day, last_purchase_date, ?) as days_since_last,
            purchase_count
        FROM CycleStats
        WHERE avg_cycle_days IS NOT NULL
        """

        scores = {}
        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (agreement_id, as_of_date, as_of_date))

            for row in cursor:
                product_id = row['ProductID']
                avg_cycle = row['avg_cycle_days']
                variance = row['cycle_variance'] or avg_cycle * 0.2  # Default 20% variance
                days_since = row['days_since_last']

                # V3.3 BUG FIX: Skip products with very short or zero cycles
                # A cycle of < MIN_CYCLE_DAYS days doesn't make sense for reorder detection
                if avg_cycle < MIN_CYCLE_DAYS:
                    scores[product_id] = 0.0
                    continue

                # Calculate how close we are to the expected reorder date
                deviation = abs(days_since - avg_cycle)
                tolerance = max(CYCLE_TOLERANCE_MIN_DAYS, variance * 0.5)

                if deviation <= tolerance:
                    # Due soon - high score (CYCLE_SCORE_DUE_MIN to CYCLE_SCORE_DUE_MAX range)
                    score_range = CYCLE_SCORE_DUE_MAX - CYCLE_SCORE_DUE_MIN
                    score = CYCLE_SCORE_DUE_MAX - (deviation / tolerance) * score_range
                elif days_since > avg_cycle:
                    # Overdue - medium score, decaying with time (CYCLE_SCORE_OVERDUE_MIN to CYCLE_SCORE_OVERDUE_MAX range)
                    overdue_days = days_since - avg_cycle
                    score_range = CYCLE_SCORE_OVERDUE_MAX - CYCLE_SCORE_OVERDUE_MIN
                    score = max(CYCLE_SCORE_OVERDUE_MIN, CYCLE_SCORE_OVERDUE_MAX - (overdue_days / avg_cycle) * score_range)
                else:
                    # Too early - low score (0 to CYCLE_SCORE_EARLY_MAX range)
                    too_early_days = avg_cycle - days_since
                    score = max(0.0, CYCLE_SCORE_EARLY_MAX - (too_early_days / avg_cycle) * CYCLE_SCORE_EARLY_MAX)

                scores[product_id] = float(score)

        logger.debug(f"Cycle scores: {len(scores)} products with detectable cycles")
        return scores

    def get_recommendations_for_agreement(self, agreement_id: int, as_of_date: str, top_n: int = 20,
                                         include_discovery: bool = True) -> List[Dict]:
        as_of_date = self._normalize_date(as_of_date)

        # Get customer ID from agreement
        with self._cursor_context() as cursor:
            cursor.execute(self._sql("SELECT ClientID FROM dbo.ClientAgreement WHERE ID = ?"), (agreement_id,))
            result = cursor.fetchone()

        if not result:
            logger.warning(f"Agreement {agreement_id} not found")
            return []

        customer_id = result['ClientID']

        # Get segment based on order count
        segment_query = """
        SELECT COUNT(DISTINCT o.ID) as orders_before
        FROM dbo.[Order] o
        WHERE o.ClientAgreementID = ?
              AND o.Created < ?
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(segment_query), (agreement_id, as_of_date))
            row = cursor.fetchone()
            orders_before = row['orders_before'] if row else 0

        if orders_before >= SEGMENT_THRESHOLD_HEAVY:
            segment = "HEAVY"
            subsegment = None
        elif orders_before >= SEGMENT_THRESHOLD_REGULAR:
            segment = "REGULAR"
            subsegment = "CONSISTENT"
        else:
            segment = "LIGHT"
            subsegment = None

        logger.debug(f"Agreement {agreement_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

        # Get frequency scores
        frequency_query = """
        SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = ?
              AND o.Created < ?
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        frequency_scores = {}
        max_count = 1
        with self._cursor_context() as cursor:
            cursor.execute(self._sql(frequency_query), (agreement_id, as_of_date))
            for row in cursor:
                frequency_scores[row['ProductID']] = row['purchase_count']
                max_count = max(max_count, row['purchase_count'])

        frequency_scores = {pid: count / max_count for pid, count in frequency_scores.items()}

        # Get recency scores
        recency_query = """
        SELECT oi.ProductID, MAX(o.Created) as last_purchase
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = ?
              AND o.Created < ?
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
        recency_scores = {}

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(recency_query), (agreement_id, as_of_date))
            for row in cursor:
                last_purchase = row['last_purchase']
                days_ago = (as_of_datetime - last_purchase).days
                score = math.e ** (-days_ago / RECENCY_DECAY_DAYS)
                recency_scores[row['ProductID']] = score

        # V3.3 ENHANCEMENT: Get co-purchase and cycle scores
        all_repurchase_products = set(frequency_scores.keys()) | set(recency_scores.keys())
        product_list = list(all_repurchase_products)

        co_purchase_scores = self.get_co_purchase_scores(
            agreement_id=agreement_id,
            product_ids=product_list,
            as_of_date=as_of_date
        )

        cycle_scores = self.get_cycle_scores(
            agreement_id=agreement_id,
            product_ids=product_list,
            as_of_date=as_of_date
        )

        # V3.3 FIXED: Get collaborative filtering scores (discovery from similar customers)
        if include_discovery:
            collaborative_scores, similar_count = self.get_collaborative_score(
                customer_id=customer_id,
                as_of_date=as_of_date
            )
        else:
            collaborative_scores, similar_count = {}, 0

        # V3.3 FIXED WEIGHTS: Use segment-specific weights from SEGMENT_WEIGHTS constant
        # 5 components: frequency, recency, co_purchase, cycle, collaborative
        v3_weights = SEGMENT_WEIGHTS.get(segment, SEGMENT_WEIGHTS['LIGHT']).copy()

        # Adaptive collaborative weight: downweight when sparse signals
        collab_count = len([s for s in collaborative_scores.values() if s > 0])
        if similar_count == 0 or collab_count == 0:
            collab_factor = 0.0
        elif similar_count < COLLAB_SPARSE_CUSTOMERS or collab_count < COLLAB_SPARSE_PRODUCTS:
            collab_factor = COLLAB_WEIGHT_SPARSE
        else:
            collab_factor = COLLAB_WEIGHT_FULL

        v3_weights['collaborative'] *= collab_factor

        # V3.3 FIXED: Combine repurchase products + discovery products from collaborative filtering
        all_candidate_products = all_repurchase_products | set(collaborative_scores.keys())

        repurchase_scores = {}
        for product_id in all_candidate_products:  # expanded from just all_repurchase_products
            freq_score = frequency_scores.get(product_id, 0)
            rec_score = recency_scores.get(product_id, 0)
            co_purch_score = co_purchase_scores.get(product_id, 0)
            cyc_score = cycle_scores.get(product_id, 0)
            collab_score = collaborative_scores.get(product_id, 0)  # NEW

            # V3.3 FIXED FORMULA: Weighted sum of 5 signals (was 4)
            repurchase_scores[product_id] = (
                v3_weights['frequency'] * freq_score +
                v3_weights['recency'] * rec_score +
                v3_weights['co_purchase'] * co_purch_score +
                v3_weights['cycle'] * cyc_score +
                v3_weights['collaborative'] * collab_score  # NEW
            )

        sorted_repurchase = sorted(repurchase_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []

        for idx, (product_id, score) in enumerate(sorted_repurchase[:top_n]):
            # V3.3 FIXED: Correctly determine source based on repurchase vs discovery
            was_purchased = product_id in all_repurchase_products
            has_collaborative = product_id in collaborative_scores and collaborative_scores[product_id] > 0

            if was_purchased and has_collaborative:
                source = 'hybrid'  # Previously purchased AND recommended by similar customers
            elif was_purchased:
                source = 'repurchase'  # Only from customer's own history
            else:
                source = 'discovery'  # Only from similar customers (never purchased before)

            recommendations.append({
                'product_id': product_id,
                'score': float(score),
                'rank': idx + 1,
                'segment': f"{segment}_{subsegment}" if subsegment else segment,
                'source': source,  # Now correctly set!
                'agreement_id': agreement_id
            })

        logger.debug(f"Agreement {agreement_id}: Generated {len(recommendations)} recommendations")

        return recommendations

    def get_recommendations(self, customer_id: int, as_of_date: str, top_n: int = 20,
                           include_discovery: bool = True) -> List[Dict]:
        """
        Generate recommendations for a customer across all their agreements.

        Args:
            customer_id: Customer ID
            as_of_date: Cutoff date for training data (YYYY-MM-DD)
            top_n: Number of recommendations to return
            include_discovery: Whether to include collaborative filtering (discovery products)

        Returns:
            List of recommendation dicts with product_id, score, rank, source, etc.

        Note: Removed unused parameters repurchase_count and discovery_count in V3.3 fix.
              The algorithm naturally blends repurchase + discovery via weighted scoring.
        """

        as_of_date = self._normalize_date(as_of_date)
        agreement_ids = self.get_customer_agreements(customer_id)

        if not agreement_ids:
            logger.warning(f"Customer {customer_id} has no agreements; returning popular products")
            return self.get_popular_products(as_of_date=as_of_date, top_n=top_n)

        all_recommendations = []

        for agreement_id in agreement_ids:
            agreement_recs = self.get_recommendations_for_agreement(
                agreement_id=agreement_id,
                as_of_date=as_of_date,
                top_n=top_n,
                include_discovery=include_discovery
            )
            all_recommendations.extend(agreement_recs)

        product_scores = {}
        for rec in all_recommendations:
            product_id = rec['product_id']
            score = rec['score']

            if product_id not in product_scores or score > product_scores[product_id]['score']:
                product_scores[product_id] = rec

        sorted_recommendations = sorted(
            product_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_n]

        for idx, rec in enumerate(sorted_recommendations):
            rec['rank'] = idx + 1

        discovery_count = sum(1 for r in sorted_recommendations if r.get('source') in ['discovery', 'hybrid'])

        logger.info(f"Customer {customer_id}: Generated {len(sorted_recommendations)} recommendations " +
                   f"from {len(agreement_ids)} agreements ({discovery_count} discovery)")

        if sorted_recommendations:
            return sorted_recommendations

        logger.warning(f"Customer {customer_id}: No personalized recommendations; returning popular products fallback")
        return self.get_popular_products(as_of_date=as_of_date, top_n=top_n)

    def get_popular_products(self, as_of_date: str, top_n: int = 20) -> List[Dict]:
        """
        Cold-start fallback: return top products by order count in recent window.
        """
        as_of_date = self._normalize_date(as_of_date)
        lookback = COLD_START_LOOKBACK_DAYS
        top_n = max(1, int(top_n))

        query = f"""
        SELECT TOP {top_n}
            oi.ProductID,
            COUNT(DISTINCT o.ID) as orders_count
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.Created < ?
          AND o.Created >= DATEADD(day, -?, ?)
          AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        ORDER BY orders_count DESC
        """

        with self._cursor_context() as cursor:
            cursor.execute(self._sql(query), (as_of_date, lookback, as_of_date))
            rows = cursor.fetchall()

        fallback = []
        for idx, row in enumerate(rows):
            fallback.append({
                'product_id': row['ProductID'],
                'score': float(row['orders_count']),
                'rank': idx + 1,
                'segment': 'COLD_START',
                'source': 'popular',
                'agreement_id': None
            })

        return fallback

    def close(self):
        if self.conn:
            self.conn.close()
            if self.owns_connection:
                logger.info("✓ Database connection closed")
            else:
                logger.debug("✓ Connection returned to pool")

        if self.redis_client:
            self.redis_client.close()
            logger.debug("✓ Redis connection closed")

def main():
    logger.info("=" * 80)
    logger.info("IMPROVED HYBRID RECOMMENDER V3.3 - QUICK WINS (Co-purchase + Cycles)")
    logger.info("=" * 80)

    recommender = ImprovedHybridRecommenderV33()

    try:

        test_customers = [
            410169,
            410175,
            410176,
            410180
        ]

        for customer_id in test_customers:
            segment, subsegment = recommender.classify_customer(customer_id, '2024-07-01')
            logger.info(f"\nCustomer {customer_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

            recs = recommender.get_recommendations(customer_id, '2024-07-01', top_n=10, include_discovery=True)

            logger.info(f"  Top 10 recommendations:")
            for rec in recs:
                logger.info(f"    {rec['rank']}. Product {rec['product_id']} - Score: {rec['score']:.4f} - Source: {rec['source']}")

        logger.info("\n✓ Test complete")

    finally:
        recommender.close()

if __name__ == '__main__':
    main()
