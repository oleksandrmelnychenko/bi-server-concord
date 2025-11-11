#!/usr/bin/env python3

import os
import sys
import json
import time
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.improved_hybrid_recommender_v32 import ImprovedHybridRecommenderV32
from api.db_pool import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workers/weekly_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_TTL = 691200
TOP_N = 50
INCLUDE_DISCOVERY = True

class WeeklyRecommendationWorker:

    def __init__(self):
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
        logger.info("Connecting to database...")
        self.conn = get_connection()

        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )

        self.redis_client.ping()
        logger.info("✅ Connected to Redis successfully")

    def disconnect(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")

    def get_all_active_customers(self, as_of_date: str) -> List[int]:
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
        try:
            start_time = time.time()

            recommendations = recommender.get_recommendations(
                customer_id=customer_id,
                as_of_date=as_of_date,
                top_n=TOP_N,
                include_discovery=INCLUDE_DISCOVERY
            )

            latency_ms = (time.time() - start_time) * 1000

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
        cache_key = f"recommendations:customer:{customer_id}:{as_of_date}"

        self.redis_client.setex(
            cache_key,
            REDIS_TTL,
            json.dumps(recommendations)
        )

        logger.debug(f"Cached recommendations for customer {customer_id} (key: {cache_key})")

    def process_all_customers(self, as_of_date: str):
        logger.info("=" * 80)
        logger.info("WEEKLY RECOMMENDATION WORKER - STARTING")
        logger.info("=" * 80)
        logger.info(f"As-of Date: {as_of_date}")
        logger.info(f"Top N: {TOP_N}")
        logger.info(f"Include Discovery: {INCLUDE_DISCOVERY}")
        logger.info(f"Redis TTL: {REDIS_TTL} seconds ({REDIS_TTL/86400:.1f} days)")

        self.stats['start_time'] = datetime.now()

        customer_ids = self.get_all_active_customers(as_of_date)
        self.stats['total_customers'] = len(customer_ids)

        if not customer_ids:
            logger.warning("No active customers found. Exiting.")
            return

        recommender = ImprovedHybridRecommenderV32(conn=self.conn, use_cache=False)

        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING CUSTOMERS")
        logger.info("=" * 80)

        for idx, customer_id in enumerate(customer_ids, 1):
            try:

                logger.info(f"[{idx}/{len(customer_ids)}] Processing customer {customer_id}...")

                result = self.generate_recommendations(
                    customer_id=customer_id,
                    as_of_date=as_of_date,
                    recommender=recommender
                )

                self.cache_recommendations(customer_id, as_of_date, result)

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

        self.stats['end_time'] = datetime.now()
        self.stats['duration_seconds'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()

        self.print_final_report()

    def print_final_report(self):
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

        stats_file = f"workers/worker_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info(f"Statistics saved to: {stats_file}")

    def run(self, as_of_date: str = None):
        try:

            if as_of_date is None:
                as_of_date = datetime.now().strftime('%Y-%m-%d')

            self.connect()

            self.process_all_customers(as_of_date)

            logger.info("\n✅ Worker completed successfully!")

        except Exception as e:
            logger.error(f"\n❌ Worker failed: {e}", exc_info=True)
            raise

        finally:

            self.disconnect()

def main():

    as_of_date = os.getenv('AS_OF_DATE')

    if as_of_date:
        logger.info(f"Using AS_OF_DATE from environment: {as_of_date}")
    else:
        as_of_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Using today's date: {as_of_date}")

    worker = WeeklyRecommendationWorker()
    worker.run(as_of_date=as_of_date)

if __name__ == '__main__':
    main()
