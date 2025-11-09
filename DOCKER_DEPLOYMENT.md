# Docker Deployment Guide - Recommendation System

Complete guide for deploying the V3.2 Recommendation System using Docker with weekly background worker and Redis caching.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  WEEKLY WORKER (Docker Container)                       │
│  - Runs weekly (cron scheduled)                        │
│  - Processes ALL ~429 customers                        │
│  - Takes ~47 minutes (6.5s × 429 customers)            │
│  - Stores in Redis with 8-day TTL                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  REDIS CACHE (Docker Container)                         │
│  Key: recommendations:customer:{id}:{date}             │
│  Value: Full recommendation JSON                       │
│  TTL: 8 days (691,200 seconds)                         │
│  Max Memory: 2GB with LRU eviction                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  FASTAPI (Docker Container × 4 workers)                 │
│  - Reads from Redis cache (< 10ms)                     │
│  - Falls back to real-time if cache miss              │
│  - Handles concurrent requests                         │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Access to MSSQL database (78.152.175.67:1433)

### 2. Environment Setup

Create `.env` file in project root:

```bash
# Database Configuration
DB_HOST=78.152.175.67
DB_PORT=1433
DB_NAME=ConcordDb_v5
DB_USER=sa
DB_PASSWORD=Passw0rd

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Worker Configuration
AS_OF_DATE=2024-07-01           # Optional: Override recommendation date
CRON_SCHEDULE=0 2 * * 0         # Sunday 2 AM (default)
RUN_ON_STARTUP=false            # Set to true for immediate first run
```

### 3. Build and Start Services

```bash
# Build all containers
docker-compose build

# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f redis
```

### 4. Verify Deployment

```bash
# Check service health
docker-compose ps

# Test API health endpoint
curl http://localhost:8000/health

# Test Redis connection
docker-compose exec redis redis-cli ping

# Check worker cron status
docker-compose exec worker crontab -l
```

## Service Details

### FastAPI Service (`api`)

**Port**: 8000
**Workers**: 4
**Health Check**: `http://localhost:8000/health`

**Endpoints**:
- `POST /recommend` - Get recommendations (checks cache first)
- `GET /weekly-recommendations/{customer_id}` - Get weekly pre-computed recs
- `GET /health` - Service health check
- `GET /metrics` - Performance metrics

**Cache Logic**:
1. Checks worker cache first (fast, pre-computed)
2. Falls back to ad-hoc cache
3. Generates real-time if cache miss

### Redis Service (`redis`)

**Port**: 6379
**Max Memory**: 2GB
**Eviction Policy**: allkeys-lru
**Persistence**: AOF enabled

**Data Structure**:
```
recommendations:customer:410376:2024-07-01 → {
  "customer_id": 410376,
  "recommendations": [...],
  "count": 25,
  "discovery_count": 5,
  "generated_at": "2025-11-09T20:30:00"
}
```

### Worker Service (`worker`)

**Cron Schedule**: Sunday 2 AM (configurable)
**Execution Time**: ~47 minutes for all customers
**Log Location**: `workers/logs/`

**Manual Execution**:
```bash
# Run worker manually (all customers)
docker-compose exec worker python3 workers/weekly_recommendation_worker.py

# Run with custom date
docker-compose exec -e AS_OF_DATE=2024-08-01 worker \
  python3 workers/weekly_recommendation_worker.py
```

## Configuration Options

### Cron Schedule Format

```bash
# Format: minute hour day month weekday
CRON_SCHEDULE=0 2 * * 0    # Sunday 2 AM (default)
CRON_SCHEDULE=0 3 * * 1    # Monday 3 AM
CRON_SCHEDULE=0 0 * * *    # Daily midnight
CRON_SCHEDULE=0 */6 * * *  # Every 6 hours
```

### Worker Startup Options

```bash
# Run worker immediately on container start, then follow cron schedule
RUN_ON_STARTUP=true docker-compose up -d worker
```

## Monitoring

### View Worker Logs

```bash
# Real-time logs
docker-compose logs -f worker

# Cron execution logs
docker-compose exec worker tail -f /app/workers/logs/cron.log

# Worker statistics
docker-compose exec worker cat /app/workers/worker_stats_*.json
```

### Redis Monitoring

```bash
# Connect to Redis CLI
docker-compose exec redis redis-cli

# Check memory usage
docker-compose exec redis redis-cli INFO memory

# Count cached recommendations
docker-compose exec redis redis-cli KEYS "recommendations:customer:*" | wc -l

# View specific customer cache
docker-compose exec redis redis-cli GET "recommendations:customer:410376:2024-07-01"

# Check cache TTL
docker-compose exec redis redis-cli TTL "recommendations:customer:410376:2024-07-01"
```

### API Metrics

```bash
# Get performance metrics
curl http://localhost:8000/metrics

# Example response:
{
  "requests": 1500,
  "cache_hits": 1450,
  "cache_misses": 50,
  "errors": 0,
  "cache_hit_rate": 0.967,
  "error_rate": 0.0,
  "avg_latency_ms": 8.5
}
```

## Production Deployment

### Scaling API Workers

```yaml
# In docker-compose.yml, update api service:
api:
  deploy:
    replicas: 4  # Run 4 API instances
    resources:
      limits:
        cpus: '2.0'
        memory: 2G
```

### Resource Limits

```yaml
services:
  redis:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G

  worker:
    deploy:
      resources:
        limits:
          cpus: '4.0'  # Worker is CPU-intensive
          memory: 4G
```

### Persistent Volumes

Data is automatically persisted in Docker volumes:

```bash
# List volumes
docker volume ls

# Inspect Redis data
docker volume inspect concord-bi-server_redis_data

# Backup Redis data
docker run --rm -v concord-bi-server_redis_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/redis_backup.tar.gz /data

# Restore Redis data
docker run --rm -v concord-bi-server_redis_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/redis_backup.tar.gz -C /
```

## Troubleshooting

### Worker Not Running

```bash
# Check worker status
docker-compose exec worker ps aux | grep cron

# Check cron configuration
docker-compose exec worker crontab -l

# Manually trigger worker
docker-compose exec worker python3 workers/weekly_recommendation_worker.py
```

### Redis Connection Issues

```bash
# Test Redis connectivity
docker-compose exec api python3 -c "import redis; r=redis.Redis(host='redis'); print(r.ping())"

# Check Redis logs
docker-compose logs redis

# Restart Redis
docker-compose restart redis
```

### Database Connection Issues

```bash
# Test database connectivity from API
docker-compose exec api python3 -c "from api.db_pool import get_connection; conn=get_connection(); print('Connected!')"

# Check database credentials in .env
docker-compose exec api env | grep DB_
```

### Out of Memory Issues

```bash
# Check Redis memory usage
docker-compose exec redis redis-cli INFO memory | grep used_memory_human

# Check container memory
docker stats

# Clear Redis cache (WARNING: deletes all recommendations)
docker-compose exec redis redis-cli FLUSHDB
```

## Maintenance

### Update Recommendation Engine

```bash
# 1. Update code
git pull origin recommendation-system

# 2. Rebuild containers
docker-compose build

# 3. Restart services (zero-downtime)
docker-compose up -d --no-deps --build api
docker-compose up -d --no-deps --build worker

# 4. Verify
curl http://localhost:8000/health
```

### Clean Up Old Data

```bash
# Remove old worker logs
docker-compose exec worker find /app/workers -name "*.log" -mtime +30 -delete

# Remove old stat files
docker-compose exec worker find /app/workers -name "worker_stats_*.json" -mtime +30 -delete
```

### Backup Configuration

```bash
# Backup docker-compose and env files
tar czf recommendation-system-config.tar.gz \
  docker-compose.yml \
  Dockerfile.api \
  Dockerfile.worker \
  .env
```

## Performance Benchmarks

### Expected Performance

| Metric | Target | Actual (20-client test) |
|--------|--------|------------------------|
| API Latency (cached) | < 10ms | ~8ms |
| API Latency (uncached) | < 7s | ~6.5s |
| Worker Time (all customers) | ~47 min | TBD |
| Cache Hit Rate | > 95% | TBD |
| Discovery Rate | 15-20% | 16.4% |

### Load Testing

```bash
# Install Apache Bench
apt-get install apache2-utils

# Test 1000 requests, 10 concurrent
ab -n 1000 -c 10 -p request.json -T application/json \
  http://localhost:8000/recommend

# request.json:
{
  "customer_id": 410376,
  "as_of_date": "2024-07-01",
  "top_n": 50,
  "use_cache": true,
  "include_discovery": true
}
```

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review worker stats: `cat workers/worker_stats_*.json`
- Verify cache: `docker-compose exec redis redis-cli KEYS "*"`
- Test API: `curl http://localhost:8000/health`
