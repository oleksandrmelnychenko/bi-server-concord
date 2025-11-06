#!/usr/bin/env python3
"""
Phase H.1: Advanced Feature Extraction for Maximum Performance

Extracts sophisticated features to push precision@50 beyond 56.8%:
- Sequence patterns (A→B purchase dependencies)
- Basket/co-purchase features (products bought together)
- Customer behavior features (predictability, exploration rate)
- Time-based features (days since last purchase, purchase overdue)

These features complement existing frequency/recency signals by capturing:
- Temporal dependencies between product purchases
- Product associations and substitution patterns
- Customer purchase regularity and exploration behavior
"""

import os
import sys
import pandas as pd
import numpy as np
import pymssql
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from itertools import combinations
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection parameters (use same as hybrid_recommender)
DB_CONFIG = {
    'server': os.environ.get('MSSQL_HOST', '78.152.175.67'),
    'port': int(os.environ.get('MSSQL_PORT', '1433')),
    'database': os.environ.get('MSSQL_DATABASE', 'ConcordDb_v5'),
    'user': os.environ.get('MSSQL_USER', 'ef_migrator'),
    'password': os.environ.get('MSSQL_PASSWORD', 'Grimm_jow92')
}


class AdvancedFeatureExtractor:
    """Extract advanced features for recommendation system"""

    def __init__(self):
        self.conn = pymssql.connect(**DB_CONFIG)
        self.sequence_patterns = None
        self.basket_patterns = None
        self.customer_behavior = None

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    # ========================================================================
    # SEQUENCE PATTERN EXTRACTION (A→B purchase dependencies)
    # ========================================================================

    def extract_sequence_patterns(self, customer_id: int, as_of_date: datetime,
                                   window_days: int = 90):
        """
        Extract product sequence patterns: "After buying A, customer buys B within N days"

        Returns dict with:
        - sequence_frequency: How often this product follows recent purchases
        - sequence_confidence: P(B|A) - probability B follows A
        - sequence_recency: Days since last time this sequence occurred
        """

        query = f"""
        SELECT
            o.ID as order_id,
            o.Created as order_date,
            oi.ProductID as product_id
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
            AND o.Created < '{as_of_date.strftime('%Y-%m-%d')}'
            AND o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
        ORDER BY o.Created
        """

        orders = pd.read_sql(query, self.conn)

        if len(orders) == 0:
            return {}

        # Build sequence patterns: (product_A → product_B within window_days)
        sequence_counts = defaultdict(int)  # (prod_A, prod_B) → count
        product_counts = Counter()  # prod_A → count
        last_sequence_date = {}  # (prod_A, prod_B) → last_date

        # Group by order and get product purchase history
        orders_by_date = orders.groupby('order_date')['product_id'].apply(list).reset_index()
        orders_by_date = orders_by_date.sort_values('order_date')

        # For each pair of orders within window_days
        for i in range(len(orders_by_date)):
            order_i_date = orders_by_date.iloc[i]['order_date']
            products_i = set(orders_by_date.iloc[i]['product_id'])

            for j in range(i+1, len(orders_by_date)):
                order_j_date = orders_by_date.iloc[j]['order_date']

                # Check if within window
                days_diff = (order_j_date - order_i_date).days
                if days_diff > window_days:
                    break  # Orders are sorted, no need to check further

                products_j = set(orders_by_date.iloc[j]['product_id'])

                # Record sequences: each product in order_i followed by each in order_j
                for prod_a in products_i:
                    product_counts[prod_a] += 1
                    for prod_b in products_j:
                        if prod_a != prod_b:  # Don't count same product
                            sequence_counts[(prod_a, prod_b)] += 1
                            last_sequence_date[(prod_a, prod_b)] = order_j_date

        # Build feature dict for each product
        # For a candidate product B, look at customer's recent purchases (A's)
        recent_purchases = set(orders[orders['order_date'] >= (as_of_date - timedelta(days=90))]['product_id'])

        features = {}
        for target_product in orders['product_id'].unique():
            # How often does target_product follow recent purchases?
            sequence_freq = 0
            sequence_conf_sum = 0
            sequence_conf_count = 0
            min_recency = float('inf')

            for recent_prod in recent_purchases:
                if recent_prod == target_product:
                    continue

                seq_count = sequence_counts.get((recent_prod, target_product), 0)
                if seq_count > 0:
                    sequence_freq += seq_count

                    # Confidence: P(B|A) = count(A→B) / count(A)
                    prod_count = product_counts.get(recent_prod, 1)
                    confidence = seq_count / prod_count
                    sequence_conf_sum += confidence
                    sequence_conf_count += 1

                    # Recency: days since last occurrence
                    last_date = last_sequence_date.get((recent_prod, target_product))
                    if last_date:
                        recency_days = (as_of_date - last_date).days
                        min_recency = min(min_recency, recency_days)

            features[target_product] = {
                'sequence_frequency': sequence_freq,
                'sequence_confidence': sequence_conf_sum / max(sequence_conf_count, 1),
                'sequence_recency': min_recency if min_recency != float('inf') else 999
            }

        return features

    # ========================================================================
    # BASKET/CO-PURCHASE PATTERN EXTRACTION
    # ========================================================================

    def extract_basket_patterns(self, customer_id: int, as_of_date: datetime):
        """
        Extract co-purchase patterns: products bought together in same order

        Returns dict with:
        - basket_frequency: How often this product appears with customer's frequent items
        - basket_lift: Statistical lift for co-purchase associations
        - basket_confidence: P(B|A) for basket co-occurrence
        """

        query = f"""
        SELECT
            o.ID as order_id,
            o.Created as order_date,
            oi.ProductID as product_id
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
            AND o.Created < '{as_of_date.strftime('%Y-%m-%d')}'
            AND o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
        """

        orders = pd.read_sql(query, self.conn)

        if len(orders) == 0:
            return {}

        # Get product baskets (products per order)
        baskets = orders.groupby('order_id')['product_id'].apply(list).values

        # Count co-occurrences
        cooccurrence = defaultdict(int)  # (prod_A, prod_B) → count
        product_counts = Counter()  # prod → count
        total_baskets = len(baskets)

        for basket in baskets:
            unique_products = set(basket)
            for prod in unique_products:
                product_counts[prod] += 1

            # Count all pairs in this basket
            for prod_a, prod_b in combinations(sorted(unique_products), 2):
                cooccurrence[(prod_a, prod_b)] += 1

        # Identify frequent items (top 20% by purchase frequency)
        sorted_prods = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
        n_frequent = max(1, len(sorted_prods) // 5)
        frequent_items = set([prod for prod, count in sorted_prods[:n_frequent]])

        # Compute features for each product
        features = {}
        for target_product in orders['product_id'].unique():
            basket_freq = 0
            lift_sum = 0
            conf_sum = 0
            count = 0

            target_count = product_counts.get(target_product, 1)
            target_prob = target_count / total_baskets

            for freq_item in frequent_items:
                if freq_item == target_product:
                    continue

                # Get co-occurrence count (handle both orderings)
                pair = tuple(sorted([freq_item, target_product]))
                cooc_count = cooccurrence.get(pair, 0)

                if cooc_count > 0:
                    basket_freq += cooc_count

                    # Lift: P(A,B) / (P(A) * P(B))
                    freq_count = product_counts[freq_item]
                    freq_prob = freq_count / total_baskets
                    joint_prob = cooc_count / total_baskets
                    lift = joint_prob / (freq_prob * target_prob) if (freq_prob * target_prob) > 0 else 1
                    lift_sum += lift

                    # Confidence: P(B|A) = P(A,B) / P(A)
                    confidence = cooc_count / freq_count
                    conf_sum += confidence

                    count += 1

            features[target_product] = {
                'basket_frequency': basket_freq,
                'basket_lift': lift_sum / max(count, 1),
                'basket_confidence': conf_sum / max(count, 1)
            }

        return features

    # ========================================================================
    # CUSTOMER BEHAVIOR FEATURES
    # ========================================================================

    def extract_customer_behavior(self, customer_id: int, as_of_date: datetime):
        """
        Extract customer-level behavior features

        Returns dict with:
        - customer_predictability: CV of purchase intervals (lower = more regular)
        - category_novelty: Is product from new category? (0/1 per product)
        - exploration_momentum: Has customer been trying new categories lately?
        """

        query = f"""
        SELECT
            o.Created as order_date,
            oi.ProductID as product_id
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
            AND o.Created < '{as_of_date.strftime('%Y-%m-%d')}'
            AND o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
        ORDER BY o.Created
        """

        orders = pd.read_sql(query, self.conn)

        if len(orders) == 0:
            return {}

        # Calculate customer predictability (CV of order intervals)
        order_dates = sorted(orders['order_date'].unique())
        if len(order_dates) > 1:
            intervals = [(order_dates[i+1] - order_dates[i]).days
                        for i in range(len(order_dates)-1)]
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            predictability = std_interval / mean_interval if mean_interval > 0 else 999
        else:
            predictability = 999  # Unknown

        # Track product purchase history
        product_first_seen = {}
        for idx, row in orders.iterrows():
            prod = row['product_id']
            date = row['order_date']
            if prod not in product_first_seen:
                product_first_seen[prod] = date

        # Calculate exploration momentum: % of products in last 90 days that are new
        recent_cutoff = as_of_date - timedelta(days=90)
        recent_orders = orders[orders['order_date'] >= recent_cutoff]
        recent_products = set(recent_orders['product_id'].unique())

        new_products_recent = sum(1 for prod in recent_products
                                  if product_first_seen.get(prod, as_of_date) >= recent_cutoff)
        exploration_momentum = new_products_recent / max(len(recent_products), 1)

        # Per-product features
        features = {}
        all_products = set(orders['product_id'].unique())

        for product in all_products:
            # Category novelty: product never seen before as_of_date
            is_novel = 1 if product_first_seen.get(product, as_of_date) >= as_of_date else 0

            features[product] = {
                'customer_predictability': predictability,
                'category_novelty': is_novel,
                'exploration_momentum': exploration_momentum
            }

        return features

    # ========================================================================
    # TIME-BASED FEATURES
    # ========================================================================

    def extract_time_features(self, customer_id: int, as_of_date: datetime):
        """
        Extract time-based features per product

        Returns dict with:
        - days_since_last: Days since customer last purchased this product
        - purchase_overdue: Is product overdue based on historical cycle? (0/1)
        - cycle_acceleration: Is customer buying faster/slower than historical avg?
        """

        query = f"""
        SELECT
            oi.ProductID as product_id,
            o.Created as order_date
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
        WHERE ca.ClientID = {customer_id}
            AND o.Created < '{as_of_date.strftime('%Y-%m-%d')}'
            AND o.Created IS NOT NULL
            AND oi.ProductID IS NOT NULL
        ORDER BY oi.ProductID, o.Created
        """

        orders = pd.read_sql(query, self.conn)

        if len(orders) == 0:
            return {}

        features = {}

        for product in orders['product_id'].unique():
            prod_orders = orders[orders['product_id'] == product]['order_date'].sort_values()

            if len(prod_orders) == 0:
                continue

            # Days since last purchase
            last_purchase = prod_orders.iloc[-1]
            days_since_last = (as_of_date - last_purchase).days

            # Purchase cycle analysis (if bought 2+ times)
            if len(prod_orders) > 1:
                intervals = [(prod_orders.iloc[i+1] - prod_orders.iloc[i]).days
                            for i in range(len(prod_orders)-1)]
                mean_cycle = np.mean(intervals)
                std_cycle = np.std(intervals)

                # Is product overdue? (last purchase > 1.5x mean cycle ago)
                overdue = 1 if days_since_last > (mean_cycle * 1.5) else 0

                # Cycle acceleration: recent cycle vs historical average
                if len(intervals) >= 2:
                    recent_cycle = intervals[-1]
                    historical_avg = np.mean(intervals[:-1])
                    # Positive = buying faster, negative = buying slower
                    acceleration = (historical_avg - recent_cycle) / historical_avg if historical_avg > 0 else 0
                else:
                    acceleration = 0
            else:
                mean_cycle = days_since_last
                overdue = 0
                acceleration = 0

            features[product] = {
                'days_since_last': days_since_last,
                'purchase_overdue': overdue,
                'cycle_acceleration': acceleration,
                'mean_cycle': mean_cycle  # For reference
            }

        return features

    # ========================================================================
    # MAIN EXTRACTION
    # ========================================================================

    def extract_all_features(self, customer_id: int, as_of_date: datetime):
        """Extract all advanced features for a customer"""

        logger.info(f"Extracting advanced features for customer {customer_id} as of {as_of_date}")

        # Extract each feature set
        sequence = self.extract_sequence_patterns(customer_id, as_of_date)
        basket = self.extract_basket_patterns(customer_id, as_of_date)
        behavior = self.extract_customer_behavior(customer_id, as_of_date)
        time_feats = self.extract_time_features(customer_id, as_of_date)

        # Merge all features
        all_products = set()
        all_products.update(sequence.keys())
        all_products.update(basket.keys())
        all_products.update(behavior.keys())
        all_products.update(time_feats.keys())

        merged_features = {}
        for product in all_products:
            merged_features[product] = {
                **sequence.get(product, {'sequence_frequency': 0, 'sequence_confidence': 0, 'sequence_recency': 999}),
                **basket.get(product, {'basket_frequency': 0, 'basket_lift': 1.0, 'basket_confidence': 0}),
                **behavior.get(product, {'customer_predictability': 999, 'category_novelty': 0, 'exploration_momentum': 0}),
                **time_feats.get(product, {'days_since_last': 999, 'purchase_overdue': 0, 'cycle_acceleration': 0, 'mean_cycle': 999})
            }

        logger.info(f"Extracted features for {len(merged_features)} products")
        return merged_features


def test_feature_extraction():
    """Test feature extraction on a single customer"""

    extractor = AdvancedFeatureExtractor()

    # Test on a heavy customer
    test_customer = 410376  # High performer from Phase E
    as_of_date = datetime(2024, 7, 1)

    logger.info(f"Testing feature extraction on customer {test_customer}")

    features = extractor.extract_all_features(test_customer, as_of_date)

    # Print sample features
    logger.info(f"\nSample of extracted features:")
    for i, (product, feats) in enumerate(list(features.items())[:5]):
        logger.info(f"\nProduct {product}:")
        for feat_name, feat_value in feats.items():
            logger.info(f"  {feat_name}: {feat_value:.3f}")

        if i >= 4:
            break

    # Compute feature statistics
    logger.info(f"\n{'='*70}")
    logger.info("FEATURE STATISTICS")
    logger.info(f"{'='*70}")

    df = pd.DataFrame.from_dict(features, orient='index')

    for col in df.columns:
        values = df[col].values
        logger.info(f"\n{col}:")
        logger.info(f"  Mean: {np.mean(values):.3f}")
        logger.info(f"  Std:  {np.std(values):.3f}")
        logger.info(f"  Min:  {np.min(values):.3f}")
        logger.info(f"  Max:  {np.max(values):.3f}")
        logger.info(f"  Non-zero: {np.sum(values != 0)} / {len(values)} ({100*np.mean(values != 0):.1f}%)")


if __name__ == '__main__':
    test_feature_extraction()
