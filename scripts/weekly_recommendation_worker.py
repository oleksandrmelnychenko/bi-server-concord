#!/usr/bin/env python3
"""
Weekly Recommendation Worker

Generates 25 product recommendations for all active clients weekly.
Runs as a background job (cron/scheduled task) to pre-compute recommendations.

Features:
- Processes all active clients (with orders in last 90 days)
- Generates 25 recommendations per client (mix of old/new products)
- Stores in Redis with weekly key (8-day TTL)
- Parallel processing with 4 workers for performance
- Database backup for persistence

Usage:
    python3 scripts/weekly_recommendation_worker.py [--dry-run] [--limit N] [--workers W]

    --dry-run: Test mode, doesn't store in Redis
    --limit N: Process only first N clients (for testing)
    --workers W: Number of parallel workers (default: 4)

Cron Example (every Monday at 6:00 AM):
    0 6 * * 1 cd /path/to/Concord-BI-Server && python3 scripts/weekly_recommendation_worker.py
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.improved_hybrid_recommender_v31 import ImprovedHybridRecommenderV31
from scripts.redis_helper import WeeklyRecommendationCache
from api.db_pool import get_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RECOMMENDATIONS_PER_CLIENT = 25
ACTIVE_DAYS_THRESHOLD = 90  # Consider clients active if ordered in last 90 days
DEFAULT_WORKERS = 4


class WeeklyRecommendationWorker:
    """Generates weekly recommendations for all active clients"""

    def __init__(self, dry_run: bool = False, num_workers: int = DEFAULT_WORKERS):
        """
        Initialize worker.

        Args:
            dry_run: If True, generate recommendations but don't store in Redis
            num_workers: Number of parallel workers
        """
        self.dry_run = dry_run
        self.num_workers = num_workers
        self.cache = None if dry_run else WeeklyRecommendationCache()
        self.week_key = WeeklyRecommendationCache.get_week_key()

        # Statistics
        self.stats = {
            'total_clients': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_recommendations': 0,
            'total_discovery': 0,
            'start_time': time.time()
        }

    def get_active_clients(self, limit: int = None) -> List[int]:
        """
        Get list of active client IDs (with orders in last N days).

        Args:
            limit: Optional limit for testing (process only first N clients)

        Returns:
            List of customer IDs
        """
        logger.info(f"Fetching active clients (orders in last {ACTIVE_DAYS_THRESHOLD} days)...")

        conn = get_connection()
        try:
            cursor = conn.cursor(as_dict=True)

            query = f"""
            SELECT DISTINCT c.ID
            FROM dbo.Client c
            INNER JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
            INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            WHERE o.Created >= DATEADD(day, -{ACTIVE_DAYS_THRESHOLD}, GETDATE())
                  AND c.IsActive = 1
                  AND c.IsBlocked = 0
                  AND c.Deleted = 0
            ORDER BY c.ID
            """

            if limit:
                query = query.replace("SELECT DISTINCT", f"SELECT DISTINCT TOP {limit}")

            cursor.execute(query)
            clients = [row['ID'] for row in cursor]
            cursor.close()

            logger.info(f"Found {len(clients)} active clients")
            return clients

        finally:
            conn.close()

    def process_client(self, customer_id: int) -> Tuple[int, bool, int, int, str]:
        """
        Generate recommendations for a single client.

        Args:
            customer_id: Client ID to process

        Returns:
            Tuple of (customer_id, success, num_recs, num_discovery, error_message)
        """
        conn = get_connection()
        try:
            # Generate recommendations
            recommender = ImprovedHybridRecommenderV31(conn=conn, use_cache=True)

            # Get recommendations (25 products, mix of old/new)
            as_of_date = datetime.now().strftime('%Y-%m-%d')
            recommendations = recommender.get_recommendations(
                customer_id=customer_id,
                as_of_date=as_of_date,
                top_n=RECOMMENDATIONS_PER_CLIENT,
                include_discovery=True
            )

            num_recommendations = len(recommendations)
            num_discovery = sum(1 for r in recommendations if r.get('source') in ['discovery', 'hybrid'])

            # Store in Redis (unless dry run)
            if not self.dry_run:
                success = self.cache.store_recommendations(
                    customer_id=customer_id,
                    recommendations=recommendations,
                    week_key=self.week_key
                )
                if not success:
                    return (customer_id, False, 0, 0, "Failed to store in Redis")

            logger.debug(
                f"✓ Customer {customer_id}: {num_recommendations} recs "
                f"({num_discovery} discovery)"
            )

            return (customer_id, True, num_recommendations, num_discovery, "")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Customer {customer_id} failed: {error_msg}")
            return (customer_id, False, 0, 0, error_msg)

        finally:
            conn.close()

    def run(self, limit: int = None):
        """
        Main worker loop - processes all active clients.

        Args:
            limit: Optional limit for testing
        """
        logger.info("=" * 80)
        logger.info("WEEKLY RECOMMENDATION WORKER")
        logger.info("=" * 80)
        logger.info(f"Week: {self.week_key}")
        logger.info(f"Recommendations per client: {RECOMMENDATIONS_PER_CLIENT}")
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Dry run: {self.dry_run}")
        if limit:
            logger.info(f"Limit: {limit} clients (testing mode)")
        logger.info("=" * 80)

        # Get active clients
        clients = self.get_active_clients(limit=limit)
        self.stats['total_clients'] = len(clients)

        if not clients:
            logger.warning("No active clients found. Exiting.")
            return

        # Process clients in parallel
        logger.info(f"\nProcessing {len(clients)} clients with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_customer = {
                executor.submit(self.process_client, customer_id): customer_id
                for customer_id in clients
            }

            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_customer), 1):
                customer_id, success, num_recs, num_discovery, error = future.result()

                self.stats['processed'] += 1

                if success:
                    self.stats['successful'] += 1
                    self.stats['total_recommendations'] += num_recs
                    self.stats['total_discovery'] += num_discovery
                else:
                    self.stats['failed'] += 1

                # Progress update every 10 clients
                if i % 10 == 0 or i == len(clients):
                    elapsed = time.time() - self.stats['start_time']
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(clients) - i) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {i}/{len(clients)} ({i/len(clients)*100:.1f}%) | "
                        f"Success: {self.stats['successful']} | "
                        f"Failed: {self.stats['failed']} | "
                        f"Rate: {rate:.1f} clients/sec | "
                        f"ETA: {eta/60:.1f} min"
                    )

        self.print_summary()

    def print_summary(self):
        """Print job summary statistics"""
        elapsed = time.time() - self.stats['start_time']

        logger.info("\n" + "=" * 80)
        logger.info("JOB SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Week: {self.week_key}")
        logger.info(f"Total clients: {self.stats['total_clients']}")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Success rate: {self.stats['successful']/max(self.stats['processed'], 1)*100:.1f}%")
        logger.info(f"Total recommendations: {self.stats['total_recommendations']}")
        logger.info(f"Total discovery recommendations: {self.stats['total_discovery']}")
        logger.info(
            f"Avg discovery per client: "
            f"{self.stats['total_discovery']/max(self.stats['successful'], 1):.1f}"
        )
        logger.info(f"Duration: {elapsed/60:.1f} minutes")
        logger.info(f"Rate: {self.stats['processed']/elapsed:.2f} clients/second")
        logger.info("=" * 80)

        if self.stats['successful'] > 0:
            logger.info("✅ JOB COMPLETED SUCCESSFULLY")
        elif self.stats['failed'] == self.stats['processed']:
            logger.error("❌ JOB FAILED (all clients failed)")
        else:
            logger.warning("⚠️  JOB COMPLETED WITH ERRORS")

    def close(self):
        """Cleanup resources"""
        if self.cache:
            self.cache.close()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Generate weekly recommendations for all active clients'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test mode - generate recommendations but don\'t store in Redis'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Process only first N clients (for testing)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})'
    )

    args = parser.parse_args()

    worker = WeeklyRecommendationWorker(
        dry_run=args.dry_run,
        num_workers=args.workers
    )

    try:
        worker.run(limit=args.limit)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Job interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Job failed with error: {e}", exc_info=True)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
