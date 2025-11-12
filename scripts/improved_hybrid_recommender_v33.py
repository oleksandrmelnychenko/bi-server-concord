#!/usr/bin/env python3

import os
import json
import pymssql
import logging
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter

DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
    'as_dict': True
}

REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
SIMILAR_CUSTOMERS_CACHE_TTL = 86400

MAX_SIMILAR_CUSTOMERS = 100
MIN_SIMILARITY_THRESHOLD = 0.05

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

    def _connect(self):
        try:
            self.conn = pymssql.connect(**DB_CONFIG)
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def classify_customer(self, customer_id: int, as_of_date: str) -> Tuple[str, str]:

        query = f"""
        SELECT COUNT(DISTINCT o.ID) as orders_before
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)
        row = cursor.fetchone()
        orders_before = row['orders_before'] if row else 0

        if orders_before >= 500:
            segment = "HEAVY"
            subsegment = None
        elif orders_before >= 100:
            segment = "REGULAR"

            repurchase_query = f"""
            SELECT
                COUNT(DISTINCT oi.ProductID) as total_products,
                COUNT(DISTINCT CASE WHEN purchase_count >= 2 THEN oi.ProductID END) as repurchased_products
            FROM (
                SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
                FROM dbo.ClientAgreement ca
                INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
                INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
                WHERE ca.ClientID = {customer_id}
                      AND o.Created < '{as_of_date}'
                      AND oi.ProductID IS NOT NULL
                GROUP BY oi.ProductID
            ) AS counts
            LEFT JOIN dbo.OrderItem oi ON oi.ProductID = counts.ProductID
            """

            cursor = self.conn.cursor(as_dict=True)
            cursor.execute(repurchase_query)
            repurchase_row = cursor.fetchone()

            if repurchase_row and repurchase_row['total_products'] > 0:
                repurchase_rate = repurchase_row['repurchased_products'] / repurchase_row['total_products']
                subsegment = "CONSISTENT" if repurchase_rate >= 0.40 else "EXPLORATORY"
            else:
                subsegment = "EXPLORATORY"  # Default to exploratory if no data
        else:
            segment = "LIGHT"
            subsegment = None

        cursor.close()
        return segment, subsegment

    def get_customer_products(self, customer_id: int, as_of_date: str, limit: int = 500) -> Set[int]:

        query = f"""
        SELECT DISTINCT ProductID
        FROM (
            SELECT TOP {limit} oi.ProductID, o.Created
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE ca.ClientID = {customer_id}
                  AND o.Created < '{as_of_date}'
                  AND oi.ProductID IS NOT NULL
            ORDER BY o.Created DESC
        ) AS RecentProducts
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        products = {row['ProductID'] for row in cursor}
        cursor.close()

        logger.debug(f"Customer {customer_id} has {len(products)} products (limit: {limit})")
        return products

    def find_similar_customers(self, customer_id: int, as_of_date: str, limit: int = MAX_SIMILAR_CUSTOMERS) -> List[Tuple[int, float]]:

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

        product_list = ','.join(str(p) for p in target_products)

        query = f"""
        SELECT DISTINCT ca.ClientID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID != {customer_id}
              AND oi.ProductID IN ({product_list})
              AND o.Created < '{as_of_date}'
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        candidate_customers = [row['ClientID'] for row in cursor]
        cursor.close()

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

    def get_collaborative_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:

        similar_customers = self.find_similar_customers(customer_id, as_of_date)

        if not similar_customers:
            logger.debug(f"No similar customers found for {customer_id}")
            return {}

        target_products = self.get_customer_products(customer_id, as_of_date, limit=5000)
        target_product_ids = ','.join(str(pid) for pid in target_products) if target_products else '0'

        similarity_values = ','.join(f"({cid}, {sim})" for cid, sim in similar_customers)

        query = f"""
        WITH SimilarityScores AS (
            -- Inline similarity scores as a virtual table
            SELECT customer_id, similarity
            FROM (VALUES {similarity_values}) AS t(customer_id, similarity)
        ),
        ProductPurchases AS (
            -- Get all products bought by similar customers with their similarity scores
            SELECT
                oi.ProductID,
                ss.customer_id,
                ss.similarity
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            INNER JOIN SimilarityScores ss ON ca.ClientID = ss.customer_id
            WHERE o.Created < '{as_of_date}'
                  AND oi.ProductID IS NOT NULL
                  AND oi.ProductID NOT IN ({target_product_ids})  -- Exclude owned products
        )
        SELECT
            ProductID,
            SUM(similarity) / COUNT(DISTINCT customer_id) as weighted_score,
            COUNT(DISTINCT customer_id) as customer_count
        FROM ProductPurchases
        GROUP BY ProductID
        HAVING COUNT(DISTINCT customer_id) >= 2  -- At least 2 similar customers bought it
        ORDER BY weighted_score DESC
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        collaborative_scores = {}
        for row in cursor:
            collaborative_scores[row['ProductID']] = float(row['weighted_score'])

        cursor.close()

        logger.debug(f"Generated {len(collaborative_scores)} collaborative recommendations")

        return collaborative_scores

    def get_frequency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        query = f"""
        SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        scores = {}
        max_count = 1
        for row in cursor:
            scores[row['ProductID']] = row['purchase_count']
            max_count = max(max_count, row['purchase_count'])

        scores = {pid: count / max_count for pid, count in scores.items()}

        cursor.close()
        return scores

    def get_recency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        query = f"""
        SELECT oi.ProductID, MAX(o.Created) as last_purchase
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
        scores = {}

        for row in cursor:
            last_purchase = row['last_purchase']
            days_ago = (as_of_datetime - last_purchase).days

            score = 2.718 ** (-days_ago / 90)
            scores[row['ProductID']] = score

        cursor.close()
        return scores

    def get_product_groups(self, product_ids: List[int]) -> Dict[int, int]:
        if not product_ids:
            return {}

        cursor = self.conn.cursor()

        ids_str = ','.join(map(str, product_ids))

        query = f"""
        SELECT ProductID, ProductGroupID
        FROM dbo.ProductProductGroup
        WHERE ProductID IN ({ids_str})
              AND Deleted = 0
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        groups = {}
        for row in rows:
            if isinstance(row, dict):
                groups[row['ProductID']] = row['ProductGroupID']
            else:
                groups[row[0]] = row[1]

        cursor.close()
        logger.debug(f"Found product groups for {len(groups)}/{len(product_ids)} products")
        return groups

    def apply_diversity_filter(self, recommendations: List[Dict], max_per_group: int = 3) -> List[Dict]:
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
        query = f"""
        SELECT ID as AgreementID
        FROM dbo.ClientAgreement
        WHERE ClientID = {customer_id}
              AND Deleted = 0
        ORDER BY Created DESC
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        agreement_ids = [row['AgreementID'] for row in cursor]
        cursor.close()

        logger.debug(f"Customer {customer_id} has {len(agreement_ids)} agreements")
        return agreement_ids

    def get_co_purchase_scores(self, agreement_id: int, product_ids: List[int],
                               as_of_date: str, days_window: int = 180) -> Dict[int, float]:
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

        product_id_list = ','.join(str(pid) for pid in product_ids)

        query = f"""
        WITH AgreementProducts AS (
            -- Products this agreement has purchased
            SELECT DISTINCT oi.ProductID
            FROM dbo.[Order] o
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.ClientAgreementID = {agreement_id}
                  AND o.Created < '{as_of_date}'
                  AND o.Created >= DATEADD(day, -{days_window}, '{as_of_date}')
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
            WHERE o.ClientAgreementID = {agreement_id}
                  AND o.Created < '{as_of_date}'
                  AND o.Created >= DATEADD(day, -{days_window}, '{as_of_date}')
                  AND oi_candidate.ProductID IN ({product_id_list})
                  AND oi_candidate.ProductID != oi_existing.ProductID
            GROUP BY oi_candidate.ProductID
        ),
        TotalOrders AS (
            SELECT COUNT(DISTINCT ID) as total FROM dbo.[Order]
            WHERE ClientAgreementID = {agreement_id}
                  AND Created < '{as_of_date}'
                  AND Created >= DATEADD(day, -{days_window}, '{as_of_date}')
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

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        scores = {}
        for row in cursor:
            scores[row['ProductID']] = float(row['co_purchase_rate'])

        cursor.close()
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

        product_id_list = ','.join(str(pid) for pid in product_ids)

        query = f"""
        WITH PurchaseDates AS (
            SELECT
                oi.ProductID,
                o.Created,
                LAG(o.Created) OVER (PARTITION BY oi.ProductID ORDER BY o.Created) as PrevPurchase
            FROM dbo.[Order] o
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE o.ClientAgreementID = {agreement_id}
                  AND o.Created < '{as_of_date}'
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
            HAVING COUNT(*) >= 2  -- Need at least 2 purchases to detect cycle
        )
        SELECT
            ProductID,
            avg_cycle_days,
            cycle_variance,
            DATEDIFF(day, last_purchase_date, '{as_of_date}') as days_since_last,
            purchase_count
        FROM CycleStats
        WHERE avg_cycle_days IS NOT NULL
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        scores = {}
        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')

        for row in cursor:
            product_id = row['ProductID']
            avg_cycle = row['avg_cycle_days']
            variance = row['cycle_variance'] or avg_cycle * 0.2  # Default 20% variance
            days_since = row['days_since_last']

            # V3.3 BUG FIX: Skip products with very short or zero cycles
            # A cycle of < 3 days doesn't make sense for reorder detection
            if avg_cycle < 3:
                scores[product_id] = 0.0
                continue

            # Calculate how close we are to the expected reorder date
            deviation = abs(days_since - avg_cycle)
            tolerance = max(7, variance * 0.5)  # At least 7 days tolerance

            if deviation <= tolerance:
                # Due soon - high score
                score = 1.0 - (deviation / tolerance) * 0.4  # 0.6-1.0 range
            elif days_since > avg_cycle:
                # Overdue - medium score, decaying with time
                overdue_days = days_since - avg_cycle
                score = max(0.3, 0.6 - (overdue_days / avg_cycle) * 0.3)  # 0.3-0.6 range
            else:
                # Too early - low score
                too_early_days = avg_cycle - days_since
                score = max(0.0, 0.2 - (too_early_days / avg_cycle) * 0.2)  # 0.0-0.2 range

            scores[product_id] = float(score)

        cursor.close()
        logger.debug(f"Cycle scores: {len(scores)} products with detectable cycles")
        return scores

    def get_recommendations_for_agreement(self, agreement_id: int, as_of_date: str, top_n: int = 20,
                                         include_discovery: bool = True) -> List[Dict]:
        cursor = self.conn.cursor(as_dict=True)

        cursor.execute(f"SELECT ClientID FROM dbo.ClientAgreement WHERE ID = {agreement_id}")
        result = cursor.fetchone()
        cursor.close()

        if not result:
            logger.warning(f"Agreement {agreement_id} not found")
            return []

        customer_id = result['ClientID']

        segment_query = f"""
        SELECT COUNT(DISTINCT o.ID) as orders_before
        FROM dbo.[Order] o
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(segment_query)
        row = cursor.fetchone()
        cursor.close()
        orders_before = row['orders_before'] if row else 0

        if orders_before >= 500:
            segment = "HEAVY"
            subsegment = None
        elif orders_before >= 100:
            segment = "REGULAR"
            subsegment = "CONSISTENT"
        else:
            segment = "LIGHT"
            subsegment = None

        logger.debug(f"Agreement {agreement_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

        frequency_query = f"""
        SELECT oi.ProductID, COUNT(DISTINCT o.ID) as purchase_count
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(frequency_query)

        frequency_scores = {}
        max_count = 1
        for row in cursor:
            frequency_scores[row['ProductID']] = row['purchase_count']
            max_count = max(max_count, row['purchase_count'])

        frequency_scores = {pid: count / max_count for pid, count in frequency_scores.items()}
        cursor.close()

        recency_query = f"""
        SELECT oi.ProductID, MAX(o.Created) as last_purchase
        FROM dbo.[Order] o
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE o.ClientAgreementID = {agreement_id}
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(recency_query)

        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
        recency_scores = {}

        for row in cursor:
            last_purchase = row['last_purchase']
            days_ago = (as_of_datetime - last_purchase).days
            score = 2.718 ** (-days_ago / 90)
            recency_scores[row['ProductID']] = score

        cursor.close()

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

        # V3.3 NEW WEIGHTS: Reduced frequency/recency, added co-purchase & cycle
        # Old V3.2: freq=50-70%, recency=25-35%
        # New V3.3: freq=35%, recency=20%, co-purchase=30%, cycle=15%
        if segment == "HEAVY":
            v3_weights = {
                'frequency': 0.35,
                'recency': 0.20,
                'co_purchase': 0.30,
                'cycle': 0.15
            }
        elif segment == "REGULAR":
            v3_weights = {
                'frequency': 0.35,
                'recency': 0.20,
                'co_purchase': 0.30,
                'cycle': 0.15
            }
        else:  # LIGHT
            v3_weights = {
                'frequency': 0.40,  # Slightly higher for infrequent buyers
                'recency': 0.25,
                'co_purchase': 0.25,
                'cycle': 0.10  # Less reliable with few orders
            }

        repurchase_scores = {}
        for product_id in all_repurchase_products:
            freq_score = frequency_scores.get(product_id, 0)
            rec_score = recency_scores.get(product_id, 0)
            co_purch_score = co_purchase_scores.get(product_id, 0)
            cyc_score = cycle_scores.get(product_id, 0)

            # V3.3 FORMULA: Weighted sum of 4 signals
            repurchase_scores[product_id] = (
                v3_weights['frequency'] * freq_score +
                v3_weights['recency'] * rec_score +
                v3_weights['co_purchase'] * co_purch_score +
                v3_weights['cycle'] * cyc_score
            )

        sorted_repurchase = sorted(repurchase_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []

        for idx, (product_id, score) in enumerate(sorted_repurchase[:top_n]):
            recommendations.append({
                'product_id': product_id,
                'score': float(score),
                'rank': idx + 1,
                'segment': f"{segment}_{subsegment}" if subsegment else segment,
                'source': 'repurchase',
                'agreement_id': agreement_id
            })

        logger.debug(f"Agreement {agreement_id}: Generated {len(recommendations)} recommendations")

        return recommendations

    def get_recommendations(self, customer_id: int, as_of_date: str, top_n: int = 20,
                           repurchase_count: int = 20, discovery_count: int = 5,
                           include_discovery: bool = True) -> List[Dict]:

        agreement_ids = self.get_customer_agreements(customer_id)

        if not agreement_ids:
            logger.warning(f"Customer {customer_id} has no agreements")
            return []

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

        return sorted_recommendations

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
