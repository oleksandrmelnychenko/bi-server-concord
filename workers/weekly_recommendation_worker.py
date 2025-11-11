#!/usr/bin/env python3
"""
Weekly Recommendation Worker

Background service that pre-computes recommendations for ALL customers
and stores them in Redis cache for fast API access.

Features:
- Processes all active customers (429 customers)
- Uses V3.2 recommendation engine with discovery
- Stores results in Redis with 8-day TTL
- Runs weekly (Sunday 2 AM via cron/scheduler)
- Progress tracking and error handling
- Metrics and statistics logging

Usage:
    python3 workers/weekly_recommendation_worker.py

Environment Variables:
    REDIS_HOST: Redis server host (default: localhost)
    REDIS_PORT: Redis server port (default: 6379)
    REDIS_DB: Redis database number (default: 0)
    AS_OF_DATE: Override as_of_date (default: today)
"""

import os
import sys
import json
import time
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.improved_hybrid_recommender_v32 import ImprovedHybridRecommenderV32
from api.db_pool import get_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workers/weekly_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_TTL = 691200  # 8 days in seconds
TOP_N = 50
INCLUDE_DISCOVERY = True


class WeeklyRecommendationWorker:
    """Worker that generates and caches recommendations for all customers"""

    def __init__(self):
        """Initialize worker with DB and Redis connections"""
        self.conn = None
        self.redis_client = None
        self.stats = {
            'total_customers': 0,
            'processed': 0,
            'cached': 0,
            'errors': 0,
            'total_recommendations': 0,
            'total_discovery': 0,
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0
        }

    def connect(self):
        """Establish database and Redis connections"""
        logger.info("Connecting to database...")
        self.conn = get_connection()

        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        # Test connection
        self.redis_client.ping()
        logger.info("✅ Connected to Redis successfully")

    def disconnect(self):
        """Close all connections"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")

    def get_all_active_customers(self, as_of_date: str) -> List[int]:
        """
        Get all active customers (with orders in last 365 days before as_of_date)

        Args:
            as_of_date: Date to use as reference point

        Returns:
            List of customer IDs
        """
        logger.info(f"Getting active customers as of {as_of_date}...")

        query = f"""
        SELECT DISTINCT ca.ClientID
        FROM dbo.ClientAgreement ca
        INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
        WHERE o.Created >= DATEADD(day, -365, '{as_of_date}')
              AND o.Created < '{as_of_date}'
        ORDER BY ca.ClientID
        """

        cursor = self.conn.cursor()
        cursor.execute(query)
        customer_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()

        logger.info(f"✅ Found {len(customer_ids)} active customers")
        return customer_ids

    def generate_recommendations(
        self,
        customer_id: int,
        as_of_date: str,
        recommender: ImprovedHybridRecommenderV32
    ) -> Dict[str, Any]:
        """
        Generate recommendations for a single customer

        Args:
            customer_id: Customer ID
            as_of_date: Date for recommendations
            recommender: Recommender instance

        Returns:
            Recommendation result dict
        """
        try:
            start_time = time.time()

            recommendations = recommender.get_recommendations(
                customer_id=customer_id,
                as_of_date=as_of_date,
                top_n=TOP_N,
                include_discovery=INCLUDE_DISCOVERY
            )

            latency_ms = (time.time() - start_time) * 1000

            # Count discovery recommendations
            discovery_count = sum(
                1 for rec in recommendations
                if rec['source'] in ['discovery', 'hybrid']
            )

            result = {
                'customer_id': customer_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'discovery_count': discovery_count,
                'latency_ms': latency_ms,
                'generated_at': datetime.now().isoformat(),
                'as_of_date': as_of_date
            }

            return result

        except Exception as e:
            logger.error(f"Error generating recommendations for customer {customer_id}: {e}")
            raise

    def cache_recommendations(
        self,
        customer_id: int,
        as_of_date: str,
        recommendations: Dict[str, Any]
    ):
        """
        Store recommendations in Redis cache

        Args:
            customer_id: Customer ID
            as_of_date: Date for recommendations
            recommendations: Recommendation result
        """
        cache_key = f"recommendations:customer:{customer_id}:{as_of_date}"

        # Store as JSON string
        self.redis_client.setex(
            cache_key,
            REDIS_TTL,
            json.dumps(recommendations)
        )

        logger.debug(f"Cached recommendations for customer {customer_id} (key: {cache_key})")

    def process_all_customers(self, as_of_date: str):
        """
        Main processing loop - generate and cache recommendations for all customers

        Args:
            as_of_date: Date to generate recommendations for
        """
        logger.info("=" * 80)
        logger.info("WEEKLY RECOMMENDATION WORKER - STARTING")
        logger.info("=" * 80)
        logger.info(f"As-of Date: {as_of_date}")
        logger.info(f"Top N: {TOP_N}")
        logger.info(f"Include Discovery: {INCLUDE_DISCOVERY}")
        logger.info(f"Redis TTL: {REDIS_TTL} seconds ({REDIS_TTL/86400:.1f} days)")

        self.stats['start_time'] = datetime.now()

        # Get all active customers
        customer_ids = self.get_all_active_customers(as_of_date)
        self.stats['total_customers'] = len(customer_ids)

        if not customer_ids:
            logger.warning("No active customers found. Exiting.")
            return

        # Initialize recommender (reuse connection)
        recommender = ImprovedHybridRecommenderV32(conn=self.conn, use_cache=False)

        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING CUSTOMERS")
        logger.info("=" * 80)

        # Process each customer
        for idx, customer_id in enumerate(customer_ids, 1):
            try:
                # Generate recommendations
                logger.info(f"[{idx}/{len(customer_ids)}] Processing customer {customer_id}...")

                result = self.generate_recommendations(
                    customer_id=customer_id,
                    as_of_date=as_of_date,
                    recommender=recommender
                )

                # Cache in Redis
                self.cache_recommendations(customer_id, as_of_date, result)

                # Update stats
                self.stats['processed'] += 1
                self.stats['cached'] += 1
                self.stats['total_recommendations'] += result['count']
                self.stats['total_discovery'] += result['discovery_count']

                logger.info(
                    f"  ✅ Customer {customer_id}: "
                    f"{result['count']} recs "
                    f"({result['discovery_count']} discovery), "
                    f"{result['latency_ms']:.0f}ms"
                )

                # Log progress every 10 customers
                if idx % 10 == 0:
                    elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                    avg_time = elapsed / idx
                    remaining = (len(customer_ids) - idx) * avg_time
                    logger.info(
                        f"\n📊 PROGRESS: {idx}/{len(customer_ids)} "
                        f"({idx/len(customer_ids)*100:.1f}%) - "
                        f"Elapsed: {elapsed/60:.1f}m, "
                        f"ETA: {remaining/60:.1f}m\n"
                    )

            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"  ❌ Error processing customer {customer_id}: {e}")
                continue

        # Final stats
        self.stats['end_time'] = datetime.now()
        self.stats['duration_seconds'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()

        self.print_final_report()

    def print_final_report(self):
        """Print final statistics report"""
        logger.info("\n" + "=" * 80)
        logger.info("WEEKLY RECOMMENDATION WORKER - COMPLETED")
        logger.info("=" * 80)

        logger.info(f"\n📊 FINAL STATISTICS:")
        logger.info(f"  Total Customers: {self.stats['total_customers']}")
        logger.info(f"  Successfully Processed: {self.stats['processed']}")
        logger.info(f"  Cached in Redis: {self.stats['cached']}")
        logger.info(f"  Errors: {self.stats['errors']}")

        logger.info(f"\n📦 RECOMMENDATIONS:")
        logger.info(f"  Total Recommendations: {self.stats['total_recommendations']}")
        logger.info(f"  Total Discovery Products: {self.stats['total_discovery']}")
        if self.stats['total_recommendations'] > 0:
            discovery_rate = self.stats['total_discovery'] / self.stats['total_recommendations'] * 100
            logger.info(f"  Discovery Rate: {discovery_rate:.1f}%")

        logger.info(f"\n⏱️  PERFORMANCE:")
        logger.info(f"  Start Time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  End Time: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Total Duration: {self.stats['duration_seconds']/60:.1f} minutes")
        if self.stats['processed'] > 0:
            avg_time = self.stats['duration_seconds'] / self.stats['processed']
            logger.info(f"  Avg Time per Customer: {avg_time:.2f} seconds")

        logger.info("\n" + "=" * 80)

        # Save stats to file
        stats_file = f"workers/worker_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info(f"Statistics saved to: {stats_file}")

    def run(self, as_of_date: str = None):
        """
        Main entry point for worker

        Args:
            as_of_date: Date to generate recommendations for (default: today)
        """
        try:
            # Use provided date or default to today
            if as_of_date is None:
                as_of_date = datetime.now().strftime('%Y-%m-%d')

            # Connect to services
            self.connect()

            # Process all customers
            self.process_all_customers(as_of_date)

            logger.info("\n✅ Worker completed successfully!")

        except Exception as e:
            logger.error(f"\n❌ Worker failed: {e}", exc_info=True)
            raise

        finally:
            # Always disconnect
            self.disconnect()


def main():
    """CLI entry point"""
    # Get as_of_date from environment or use today
    as_of_date = os.getenv('AS_OF_DATE')

    if as_of_date:
        logger.info(f"Using AS_OF_DATE from environment: {as_of_date}")
    else:
        as_of_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Using today's date: {as_of_date}")

    # Create and run worker
    worker = WeeklyRecommendationWorker()
    worker.run(as_of_date=as_of_date)


if __name__ == '__main__':
    main()
