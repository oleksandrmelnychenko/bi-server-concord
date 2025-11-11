import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import redis

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
WEEKLY_TTL = int(os.getenv('WEEKLY_TTL', 604800))  # 7 days (exactly 1 week) - FIXED from 8 days

class WeeklyRecommendationCache:

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB):
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_keepalive=True
            )

            self.client.ping()
            logger.info(f"✓ Connected to Redis at {host}:{port} (DB {db})")
        except Exception as e:
            logger.error(f"✗ Failed to connect to Redis: {e}")
            raise

    @staticmethod
    def get_week_key(date: Optional[datetime] = None) -> str:
        if date is None:
            date = datetime.now()

        year, week, _ = date.isocalendar()
        return f"{year}_W{week:02d}"

    def get_redis_key(self, customer_id: int, week_key: Optional[str] = None) -> str:
        if week_key is None:
            week_key = self.get_week_key()
        return f"weekly_recs:{week_key}:{customer_id}"

    def store_recommendations(
        self,
        customer_id: int,
        recommendations: List[Dict[str, Any]],
        week_key: Optional[str] = None,
        ttl: int = WEEKLY_TTL
    ) -> bool:
        try:
            key = self.get_redis_key(customer_id, week_key)
            value = json.dumps(recommendations)
            self.client.setex(key, ttl, value)
            logger.debug(f"Stored {len(recommendations)} recommendations for customer {customer_id} (key: {key}, TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to store recommendations for customer {customer_id}: {e}")
            return False

    def get_recommendations(
        self,
        customer_id: int,
        week_key: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            key = self.get_redis_key(customer_id, week_key)
            value = self.client.get(key)

            if value is None:
                logger.debug(f"Cache MISS for customer {customer_id} (key: {key})")
                return None

            recommendations = json.loads(value)
            logger.debug(f"Cache HIT for customer {customer_id} (key: {key}, {len(recommendations)} recs)")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to retrieve recommendations for customer {customer_id}: {e}")
            return None

    def delete_recommendations(
        self,
        customer_id: int,
        week_key: Optional[str] = None
    ) -> bool:
        try:
            key = self.get_redis_key(customer_id, week_key)
            deleted = self.client.delete(key)
            logger.debug(f"Deleted recommendations for customer {customer_id} (key: {key})")
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete recommendations for customer {customer_id}: {e}")
            return False

    def clear_week(self, week_key: Optional[str] = None) -> int:
        try:
            if week_key is None:
                week_key = self.get_week_key()

            pattern = f"weekly_recs:{week_key}:*"
            keys = list(self.client.scan_iter(match=pattern, count=100))

            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} recommendations for week {week_key}")
                return deleted
            else:
                logger.info(f"No recommendations found for week {week_key}")
                return 0

        except Exception as e:
            logger.error(f"Failed to clear week {week_key}: {e}")
            return 0

    def get_cached_customer_count(self, week_key: Optional[str] = None) -> int:
        try:
            if week_key is None:
                week_key = self.get_week_key()

            pattern = f"weekly_recs:{week_key}:*"
            count = sum(1 for _ in self.client.scan_iter(match=pattern, count=100))
            logger.debug(f"Found {count} cached customers for week {week_key}")
            return count

        except Exception as e:
            logger.error(f"Failed to count cached customers: {e}")
            return 0

    def get_ttl(self, customer_id: int, week_key: Optional[str] = None) -> Optional[int]:
        try:
            key = self.get_redis_key(customer_id, week_key)
            ttl = self.client.ttl(key)

            if ttl == -2:
                return None
            elif ttl == -1:
                return -1
            else:
                return ttl

        except Exception as e:
            logger.error(f"Failed to get TTL for customer {customer_id}: {e}")
            return None

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except:
            return False

    def close(self):
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

def test_redis_connection():
    try:
        cache = WeeklyRecommendationCache()

        assert cache.ping(), "Redis ping failed"
        print("✓ Redis connection successful")

        week_key = cache.get_week_key()
        print(f"✓ Current week key: {week_key}")

        test_customer = 999999
        test_recs = [
            {"product_id": 123, "score": 85.5, "rank": 1, "source": "repurchase"},
            {"product_id": 456, "score": 72.3, "rank": 2, "source": "discovery"}
        ]

        cache.store_recommendations(test_customer, test_recs, ttl=60)
        print(f"✓ Stored test recommendations for customer {test_customer}")

        retrieved = cache.get_recommendations(test_customer)
        assert retrieved == test_recs, "Retrieved data doesn't match stored data"
        print(f"✓ Retrieved test recommendations successfully")

        ttl = cache.get_ttl(test_customer)
        assert ttl is not None and ttl > 0, "TTL not set correctly"
        print(f"✓ TTL set correctly: {ttl}s remaining")

        cache.delete_recommendations(test_customer)
        retrieved = cache.get_recommendations(test_customer)
        assert retrieved is None, "Recommendations not deleted"
        print(f"✓ Deleted test recommendations successfully")

        cache.close()
        print("\n✅ All Redis tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Redis test failed: {e}")
        return False

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_redis_connection()
