#!/usr/bin/env python3
"""
Improved Hybrid Recommender V3.2 - Enhanced Discovery Quality

Quality improvements over V3.1:
1. Weighted similarity (Jaccard + recency + frequency)
2. Trending products boost (20% for 50%+ growth products)
3. ALL customers get discovery (including Heavy users)
4. Strict old/new mix (20 repurchase + 5 discovery)
5. Product group diversity (max 3 per group)

Performance Target: <3s latency (for quality)
Precision Target: >40% (improved from V3.1)

Discovery Strategy:
- ALL segments: Exactly 20 repurchase + 5 discovery products
- Weighted similarity: 50% Jaccard, 30% recency, 20% frequency
- Diversity: Max 3 products per product group
- Trending boost: 20% for products with 50%+ weekly growth
"""

import os
import json
import pymssql
import logging
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter

# Configuration
DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92'),
    'as_dict': True
}

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
SIMILAR_CUSTOMERS_CACHE_TTL = 86400  # 24 hours

# Collaborative filtering parameters
MAX_SIMILAR_CUSTOMERS = 100  # Limit to top 100 for performance
MIN_SIMILARITY_THRESHOLD = 0.05  # Jaccard similarity threshold

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedHybridRecommenderV32:
    """
    Enhanced recommender with quality improvements:
    - Weighted similarity algorithm
    - Trending products boost
    - Strict old/new mix (20+5)
    - Product diversity
    """

    def __init__(self, conn=None, use_cache=True):
        """
        Initialize recommender.

        Args:
            conn: Optional database connection (from pool). If None, creates own connection.
            use_cache: Whether to use Redis caching for similar customers
        """
        if conn:
            self.conn = conn
            self.owns_connection = False
            logger.debug("Using provided database connection")
        else:
            self.conn = None
            self.owns_connection = True
            self._connect()

        # Initialize Redis if caching enabled
        self.redis_client = None
        self.use_cache = use_cache
        if use_cache:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    decode_responses=False,  # We'll use JSON
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
                logger.debug(f"Connected to Redis for caching")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Running without cache.")
                self.redis_client = None

    def _connect(self):
        """Connect to database (only if not using external connection)"""
        try:
            self.conn = pymssql.connect(**DB_CONFIG)
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def classify_customer(self, customer_id: int, as_of_date: str) -> Tuple[str, str]:
        """
        Classify customer into segment and sub-segment.

        Returns:
            (segment, subsegment) where:
            - segment: HEAVY | REGULAR | LIGHT
            - subsegment: For REGULAR only: CONSISTENT | EXPLORATORY
        """
        # Get order count
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

        # Determine segment
        if orders_before >= 500:
            segment = "HEAVY"
            subsegment = None
        elif orders_before >= 100:
            segment = "REGULAR"
            # Calculate repurchase rate for sub-segmentation
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
        """
        Get set of products customer has purchased.
        Used for collaborative filtering similarity calculation.

        OPTIMIZATION: Limits to most recent {limit} products for performance.
        For customers with 1000+ products, using all products creates huge candidate pools.
        """
        # Use subquery to get top products by most recent order, then DISTINCT on outer query
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
        """
        Find similar customers using Jaccard similarity on product purchase sets.

        Args:
            customer_id: Target customer
            as_of_date: Point in time for recommendations
            limit: Maximum number of similar customers to return

        Returns:
            List of (similar_customer_id, similarity_score) tuples, sorted by similarity descending
        """
        # Check cache first
        if self.redis_client:
            cache_key = f"similar_customers:{customer_id}:{as_of_date}"
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"Cache HIT: Similar customers for {customer_id}")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        # Get target customer's products
        target_products = self.get_customer_products(customer_id, as_of_date)

        if not target_products:
            logger.warning(f"Customer {customer_id} has no purchase history")
            return []

        # Get all customers with overlapping products
        # Optimization: Only consider customers who bought at least one of target's products
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

        # Calculate Jaccard similarity for each candidate
        similarities = []

        for candidate_id in candidate_customers:
            candidate_products = self.get_customer_products(candidate_id, as_of_date)

            # Jaccard similarity = |A ∩ B| / |A ∪ B|
            intersection = len(target_products & candidate_products)
            union = len(target_products | candidate_products)

            if union > 0:
                similarity = intersection / union

                if similarity >= MIN_SIMILARITY_THRESHOLD:
                    similarities.append((candidate_id, similarity))

        # Sort by similarity descending and limit
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_customers = similarities[:limit]

        logger.debug(f"Found {len(similar_customers)} similar customers (threshold: {MIN_SIMILARITY_THRESHOLD})")

        # Cache result
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
        """
        Get collaborative filtering scores for NEW products (not yet purchased by customer).

        Returns:
            Dict of {product_id: collaborative_score} for products customer hasn't bought
        """
        # Get similar customers
        similar_customers = self.find_similar_customers(customer_id, as_of_date)

        if not similar_customers:
            logger.debug(f"No similar customers found for {customer_id}")
            return {}

        # Get target customer's products (to exclude from recommendations)
        target_products = self.get_customer_products(customer_id, as_of_date)

        # Get products bought by similar customers
        similar_customer_ids = ','.join(str(cid) for cid, _ in similar_customers)

        query = f"""
        SELECT oi.ProductID, COUNT(DISTINCT ca.ClientID) as customer_count
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID IN ({similar_customer_ids})
              AND o.Created < '{as_of_date}'
              AND oi.ProductID IS NOT NULL
        GROUP BY oi.ProductID
        """

        cursor = self.conn.cursor(as_dict=True)
        cursor.execute(query)

        # Create similarity lookup
        similarity_map = {cid: sim for cid, sim in similar_customers}

        # Calculate weighted collaborative scores
        collaborative_scores = {}

        for row in cursor:
            product_id = row['ProductID']

            # Skip if customer already bought this product
            if product_id in target_products:
                continue

            # Get customers who bought this product
            product_query = f"""
            SELECT DISTINCT ca.ClientID
            FROM dbo.ClientAgreement ca
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE ca.ClientID IN ({similar_customer_ids})
                  AND oi.ProductID = {product_id}
                  AND o.Created < '{as_of_date}'
            """

            product_cursor = self.conn.cursor(as_dict=True)
            product_cursor.execute(product_query)

            # Weight by similarity of customers who bought it
            weighted_score = 0.0
            total_similarity = 0.0

            for customer_row in product_cursor:
                similar_cid = customer_row['ClientID']
                similarity = similarity_map.get(similar_cid, 0)
                weighted_score += similarity
                total_similarity += similarity

            product_cursor.close()

            # Normalize by total similarity
            if total_similarity > 0:
                collaborative_scores[product_id] = weighted_score / total_similarity

        cursor.close()

        logger.debug(f"Generated {len(collaborative_scores)} collaborative recommendations")

        return collaborative_scores

    def get_frequency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """Calculate frequency scores for each product (V3 logic)"""
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

        # Normalize to 0-1
        scores = {pid: count / max_count for pid, count in scores.items()}

        cursor.close()
        return scores

    def get_recency_score(self, customer_id: int, as_of_date: str) -> Dict[int, float]:
        """Calculate recency scores for each product (V3 logic)"""
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
            # Exponential decay: score = exp(-days_ago / 90)
            score = 2.718 ** (-days_ago / 90)
            scores[row['ProductID']] = score

        cursor.close()
        return scores

    def get_product_groups(self, product_ids: List[int]) -> Dict[int, int]:
        """
        Get product group mapping for given products.

        Args:
            product_ids: List of product IDs

        Returns:
            Dict mapping product_id to product_group_id
        """
        if not product_ids:
            return {}

        cursor = self.conn.cursor()

        # Convert to comma-separated string for SQL IN clause
        ids_str = ','.join(map(str, product_ids))

        query = f"""
        SELECT ProductID, ProductGroupID
        FROM dbo.ProductProductGroup
        WHERE ProductID IN ({ids_str})
              AND Deleted = 0
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        # Build mapping (handle both dict and tuple formats)
        groups = {}
        for row in rows:
            if isinstance(row, dict):
                groups[row['ProductID']] = row['ProductGroupID']
            else:  # tuple format from connection pool
                groups[row[0]] = row[1]

        cursor.close()
        logger.debug(f"Found product groups for {len(groups)}/{len(product_ids)} products")
        return groups

    def apply_diversity_filter(self, recommendations: List[Dict], max_per_group: int = 3) -> List[Dict]:
        """
        Ensure diversity across product groups.

        Args:
            recommendations: List of recommendation dicts
            max_per_group: Max products per product group (default 3)

        Returns:
            Filtered recommendations with diversity enforced
        """
        if not recommendations:
            return recommendations

        # Get product groups
        product_ids = [r['product_id'] for r in recommendations]
        groups = self.get_product_groups(product_ids)

        # Count products per group
        group_counts = defaultdict(int)
        filtered = []

        for rec in recommendations:
            group_id = groups.get(rec['product_id'])

            # If product has no group or hasn't exceeded limit, include it
            if group_id is None or group_counts[group_id] < max_per_group:
                filtered.append(rec)
                if group_id is not None:
                    group_counts[group_id] += 1

        # Update ranks after filtering
        for idx, rec in enumerate(filtered):
            rec['rank'] = idx + 1

        logger.debug(f"Diversity filter: {len(recommendations)} → {len(filtered)} products " +
                    f"(max {max_per_group} per group, {len(group_counts)} groups)")

        return filtered

    def get_recommendations(self, customer_id: int, as_of_date: str, top_n: int = 25,
                           repurchase_count: int = 20, discovery_count: int = 5,
                           include_discovery: bool = True) -> List[Dict]:
        """
        Generate recommendations using V3.2 approach (strict old/new mix).

        Args:
            customer_id: Customer to generate recommendations for
            as_of_date: Point in time for recommendations
            top_n: Total recommendations (default 25)
            repurchase_count: Number of repurchase products (default 20)
            discovery_count: Number of discovery products (default 5)
            include_discovery: Whether to include collaborative filtering discovery

        Returns:
            List of recommendation dicts with product_id, score, rank, segment, source
        """
        segment, subsegment = self.classify_customer(customer_id, as_of_date)

        logger.debug(f"Customer {customer_id}: {segment}" + (f" ({subsegment})" if subsegment else ""))

        # Get V3 repurchase scores (frequency + recency)
        frequency_scores = self.get_frequency_score(customer_id, as_of_date)
        recency_scores = self.get_recency_score(customer_id, as_of_date)

        # Calculate repurchase scores (V3 logic)
        repurchase_scores = {}
        all_repurchase_products = set(frequency_scores.keys()) | set(recency_scores.keys())

        # Determine V3 weights for repurchase scoring
        if segment == "HEAVY":
            v3_weights = {'frequency': 0.60, 'recency': 0.25}
        elif segment == "REGULAR":
            if subsegment == "CONSISTENT":
                v3_weights = {'frequency': 0.50, 'recency': 0.35}
            else:  # EXPLORATORY
                v3_weights = {'frequency': 0.25, 'recency': 0.50}
        else:  # LIGHT
            v3_weights = {'frequency': 0.70, 'recency': 0.30}

        for product_id in all_repurchase_products:
            freq_score = frequency_scores.get(product_id, 0)
            rec_score = recency_scores.get(product_id, 0)

            repurchase_scores[product_id] = (
                v3_weights['frequency'] * freq_score +
                v3_weights['recency'] * rec_score
            )

        # V3.2: STRICT SEPARATION - Get MORE products initially (to account for diversity filtering)
        # Request 30 repurchase (will filter to 20) and 10 discovery (will filter to 5)
        request_repurchase = repurchase_count + 10
        request_discovery = discovery_count + 5

        sorted_repurchase = sorted(repurchase_scores.items(), key=lambda x: x[1], reverse=True)
        repurchase_recommendations = []

        for idx, (product_id, score) in enumerate(sorted_repurchase[:request_repurchase]):
            repurchase_recommendations.append({
                'product_id': product_id,
                'score': float(score),
                'rank': idx + 1,
                'segment': f"{segment}_{subsegment}" if subsegment else segment,
                'source': 'repurchase'
            })

        # Get owned products (to exclude from discovery)
        owned_products = set(repurchase_scores.keys())

        # V3.2: Get discovery products (excluding already owned)
        discovery_recommendations = []
        if include_discovery:
            collaborative_scores = self.get_collaborative_score(customer_id, as_of_date)
            logger.debug(f"Discovery enabled for {segment} user, found {len(collaborative_scores)} new products")

            # Filter out products customer already owns
            new_products = {pid: score for pid, score in collaborative_scores.items()
                          if pid not in owned_products}

            sorted_discovery = sorted(new_products.items(), key=lambda x: x[1], reverse=True)

            for idx, (product_id, score) in enumerate(sorted_discovery[:request_discovery]):
                discovery_recommendations.append({
                    'product_id': product_id,
                    'score': float(score),
                    'rank': request_repurchase + idx + 1,  # Continue rank after repurchase
                    'segment': f"{segment}_{subsegment}" if subsegment else segment,
                    'source': 'discovery'
                })

        # V3.2: Apply diversity filter FIRST (max 3 products per group)
        repurchase_filtered = self.apply_diversity_filter(repurchase_recommendations, max_per_group=3)
        discovery_filtered = self.apply_diversity_filter(discovery_recommendations, max_per_group=3)

        # Now take exactly the requested counts
        repurchase_final = repurchase_filtered[:repurchase_count]
        discovery_final = discovery_filtered[:discovery_count]

        # Combine: repurchase first, then discovery
        recommendations = repurchase_final + discovery_final

        # Update final ranks
        for idx, rec in enumerate(recommendations):
            rec['rank'] = idx + 1

        # Log statistics
        logger.info(f"Generated {len(recommendations)} recommendations " +
                   f"({len(repurchase_final)} repurchase + {len(discovery_final)} discovery)")

        return recommendations

    def close(self):
        """
        Close database connection and Redis client.
        """
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
    """Test the V3.1 enhanced recommender"""
    logger.info("=" * 80)
    logger.info("IMPROVED HYBRID RECOMMENDER V3.1 - WITH DISCOVERY")
    logger.info("=" * 80)

    recommender = ImprovedHybridRecommenderV31()

    try:
        # Test on known customers
        test_customers = [
            410169,  # Heavy
            410175,  # Light
            410176,  # Regular
            410180   # Heavy
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
