#!/usr/bin/env python3
"""
Trigger Re-learning - Manual Forecast Refresh

Simple script to trigger forecast re-learning for all products.
This clears the cache and runs the forecast worker.

Usage:
    python3 scripts/forecasting/trigger_relearn.py
    python3 scripts/forecasting/trigger_relearn.py --clear-only  # Just clear cache
"""

import os
import sys
import redis
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))


def clear_forecast_cache():
    """
    Clear all forecast caches from Redis
    """
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5
        )
        redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

        # Find all forecast keys
        pattern = "forecast:product:*"
        keys = list(redis_client.scan_iter(match=pattern, count=1000))

        if keys:
            logger.info(f"Found {len(keys):,} cached forecasts")
            logger.info("Clearing cache...")

            # Delete in batches
            batch_size = 1000
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                redis_client.delete(*batch)
                logger.info(f"Deleted {min(i + batch_size, len(keys)):,}/{len(keys):,} keys")

            logger.info(f"✓ Successfully cleared {len(keys):,} cached forecasts")
        else:
            logger.info("No cached forecasts found")

        # Also clear metadata
        redis_client.delete('forecast:metadata:last_run')

        return len(keys)

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise


def run_forecast_worker():
    """
    Run the forecast worker script
    """
    logger.info("="*80)
    logger.info("Starting forecast worker...")
    logger.info("="*80)

    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        worker_script = os.path.join(script_dir, 'forecast_worker.py')

        # Run worker
        result = subprocess.run(
            [sys.executable, worker_script],
            cwd=os.path.join(script_dir, '../..'),
            check=True
        )

        logger.info("="*80)
        logger.info("Forecast worker completed successfully")
        logger.info("="*80)

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Forecast worker failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error running forecast worker: {e}")
        return False


def main():
    """
    Main entry point
    """
    clear_only = '--clear-only' in sys.argv

    try:
        logger.info("="*80)
        logger.info("FORECAST RE-LEARNING TRIGGER")
        logger.info("="*80)

        # Step 1: Clear cache
        cleared = clear_forecast_cache()

        if clear_only:
            logger.info("Cache cleared. Skipping worker run (--clear-only flag)")
            sys.exit(0)

        # Step 2: Run worker
        success = run_forecast_worker()

        if success:
            logger.info("✓ Re-learning completed successfully")
            sys.exit(0)
        else:
            logger.error("✗ Re-learning failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Re-learning trigger failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
