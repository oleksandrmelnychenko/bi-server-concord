# Concord BI Server - Deployment Guide

Complete guide for deploying the B2B Product Recommendation API.

## 🚀 Quick Start (New Users)

```bash
# One-command setup and deployment
./quickstart.sh
```

The quickstart script will:
1. ✓ Create and configure `.env` file
2. ✓ Validate your environment
3. ✓ Guide you through deployment mode selection
4. ✓ Deploy the services
5. ✓ Provide testing instructions

---

## 📋 Deployment Scripts

### 1. `quickstart.sh` - Interactive Setup
**Best for:** First-time users, quick demos

```bash
./quickstart.sh
```

Interactive wizard that handles everything automatically.

### 2. `deploy.sh` - Flexible Deployment
**Best for:** Production deployments, CI/CD pipelines

```bash
# Minimal deployment (API + Redis only)
./deploy.sh api-only

# Core services (API + Redis + Postgres + MLflow)
./deploy.sh core

# Full stack (all services)
./deploy.sh full

# Local development (no Docker)
./deploy.sh local

# Management commands
./deploy.sh stop          # Stop all services
./deploy.sh restart       # Restart all services
./deploy.sh status        # Show service status
./deploy.sh logs          # Follow API logs
./deploy.sh clean         # Remove everything (WARNING: deletes data)
```

**Options:**
- `--build` - Force rebuild Docker images
- `--logs` - Follow logs after deployment
- `--no-cache` - Build without cache

**Examples:**
```bash
# Deploy with rebuild and follow logs
./deploy.sh full --build --logs

# Quick API-only deployment
./deploy.sh api-only

# Stop everything
./deploy.sh stop
```

### 3. `validate-env.sh` - Environment Validation
**Best for:** Pre-deployment checks, troubleshooting

```bash
./validate-env.sh
```

Validates:
- ✓ System requirements (CPU, RAM, disk)
- ✓ Required software (Docker, Docker Compose, Python)
- ✓ Configuration files (`.env`, `docker-compose.yml`)
- ✓ Project structure
- ✓ Network connectivity
- ✓ Port availability
- ✓ Docker resources

---

## 🎯 Deployment Modes Comparison

| Mode | Services | Startup Time | RAM Usage | Use Case |
|------|----------|--------------|-----------|----------|
| **API Only** | API + Redis | ~30 sec | ~500MB | Development, Testing |
| **Core** | + Postgres + MLflow | ~2 min | ~2GB | ML Experiments |
| **Full** | + DataHub + Grafana | ~5 min | ~8GB | Production |
| **Local** | Python process | Instant | ~200MB | Quick testing |

---

## 📦 Prerequisites

### Minimum Requirements
- **CPU:** 2 cores
- **RAM:** 4GB (8GB recommended)
- **Disk:** 20GB free space
- **OS:** Linux, macOS, or Windows with WSL2

### Required Software
- Docker 20.10+
- Docker Compose 2.0+
- (Optional) Python 3.11+ for local mode
- (Optional) curl for health checks

### Installation
```bash
# macOS
brew install docker docker-compose

# Ubuntu/Debian
sudo apt install docker.io docker-compose

# Verify installation
docker --version
docker-compose --version
```

---

## ⚙️ Configuration

### 1. Environment Variables

Copy and configure `.env`:

```bash
cp .env.example .env
nano .env  # or vim, code, etc.
```

**Required settings:**
```bash
# Database Configuration
DB_SERVER=78.152.175.67
DB_PORT=1433
DB_NAME=ConcordDb_v5
DB_USER=your_username
DB_PASSWORD=your_password

# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600

# API Configuration
API_PORT=8000
API_WORKERS=4
```

### 2. Database Connection

Ensure your database server is accessible:

```bash
# Test connection
telnet <DB_SERVER> <DB_PORT>

# Or using netcat
nc -zv <DB_SERVER> <DB_PORT>
```

---

## 🚢 Deployment Steps

### Option A: Quick Start (Recommended for First Time)

```bash
# Run interactive setup
./quickstart.sh

# Follow the prompts to:
# 1. Configure database credentials
# 2. Choose deployment mode
# 3. Validate environment
# 4. Deploy services
```

### Option B: Manual Deployment

```bash
# Step 1: Validate environment
./validate-env.sh

# Step 2: Choose deployment mode
./deploy.sh api-only     # Minimal
./deploy.sh core         # Standard
./deploy.sh full         # Complete

# Step 3: Wait for services to start (~30 sec - 5 min)

# Step 4: Verify deployment
curl http://localhost:8000/health
```

### Option C: Using Make (Alternative)

```bash
# Initialize project
make init

# Start API only
make up-api

# View logs
make logs-api

# Check status
make status
```

---

## 🧪 Testing Your Deployment

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "redis_connected": true,
  "model_version": "improved_hybrid_v3_75.4pct_pooled"
}
```

### 2. API Documentation

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Test Recommendation

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 410376,
    "top_n": 50,
    "use_cache": true
  }'
```

Expected response:
```json
{
  "customer_id": 410376,
  "recommendations": [...],
  "count": 50,
  "precision_estimate": 0.754,
  "latency_ms": 245.32,
  "cached": false,
  "timestamp": "2025-11-06T10:30:00"
}
```

### 4. Performance Test

```bash
# Install Apache Bench (if not installed)
# macOS: brew install apache-bench
# Ubuntu: sudo apt install apache2-utils

# Run 100 requests with 10 concurrent connections
ab -n 100 -c 10 -p request.json -T application/json \
  http://localhost:8000/recommend
```

---

## 🔍 Monitoring & Logs

### View Logs

```bash
# All services
docker-compose logs -f

# API only
docker-compose logs -f api

# Specific service
docker-compose logs -f redis

# Last 100 lines
docker-compose logs --tail=100 api
```

### Check Service Status

```bash
# Using deployment script
./deploy.sh status

# Using Docker Compose
docker-compose ps

# Using Make
make status
```

### Access URLs

| Service | URL | Purpose |
|---------|-----|---------|
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| API Health | http://localhost:8000/health | Health check endpoint |
| API Metrics | http://localhost:8000/metrics | Performance metrics |
| MLflow | http://localhost:5001 | ML experiment tracking |
| DataHub | http://localhost:9002 | Data catalog |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| MinIO | http://localhost:9001 | Object storage console |

---

## 🛠️ Troubleshooting

### Port Already in Use

```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port in .env
API_PORT=8001
```

### Docker Daemon Not Running

```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker

# Verify
docker info
```

### Out of Memory

```bash
# Check Docker memory limit
docker info | grep Memory

# Increase Docker memory in Docker Desktop:
# Preferences → Resources → Memory (set to 8GB+)

# Or use API-only mode
./deploy.sh api-only
```

### Database Connection Failed

```bash
# Test database connectivity
telnet <DB_SERVER> <DB_PORT>

# Check credentials in .env
cat .env | grep DB_

# Verify firewall rules allow connection to DB_SERVER:DB_PORT
```

### Services Won't Start

```bash
# Check logs for errors
docker-compose logs

# Remove and recreate
./deploy.sh clean
./deploy.sh api-only

# Rebuild from scratch
./deploy.sh api-only --build --no-cache
```

### Redis Connection Failed

```bash
# Check if Redis is running
docker-compose ps redis

# Restart Redis
docker-compose restart redis

# Check Redis logs
docker-compose logs redis
```

---

## 🔄 Updating & Maintenance

### Update Application Code

```bash
# Pull latest changes (if using git)
git pull

# Rebuild and restart
./deploy.sh api-only --build

# Or using Make
make restart-api
```

### Clear Cache

```bash
# Clear all cached recommendations
curl -X POST http://localhost:8000/cache/clear-all

# Clear specific customer cache
curl -X DELETE http://localhost:8000/cache/410376
```

### Backup Database (Postgres metadata)

```bash
# Backup
docker-compose exec postgres pg_dump -U concord concord_bi > backup.sql

# Restore
docker-compose exec -T postgres psql -U concord concord_bi < backup.sql
```

---

## 🔒 Security Considerations

### Before Production

- [ ] Change default passwords in `.env`
- [ ] Enable HTTPS/TLS (use reverse proxy like nginx)
- [ ] Implement API authentication (JWT, OAuth2, API keys)
- [ ] Enable rate limiting
- [ ] Configure IP whitelisting
- [ ] Set up audit logging
- [ ] Use secret management (AWS Secrets Manager, HashiCorp Vault)
- [ ] Regular security updates

### Network Security

```bash
# Restrict Redis to localhost only
REDIS_HOST=127.0.0.1

# Use Docker internal network (already configured)
# Services communicate via concord-network
```

---

## 📊 Performance Tuning

### API Workers

```bash
# Adjust in .env or docker-compose.yml
API_WORKERS=4  # Default: 4 (set to CPU cores)
```

### Redis Cache TTL

```bash
# Adjust in .env
CACHE_TTL=3600  # 1 hour in seconds
```

### Connection Pool Size

Edit `api/db_pool.py`:
```python
MAX_CONNECTIONS = 20  # Increase for higher concurrency
MAX_OVERFLOW = 10     # Additional connections when pool is full
```

---

## 🚀 Production Deployment

### Docker Swarm (Recommended)

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml concord-bi

# Scale API service
docker service scale concord-bi_api=3
```

### Kubernetes

```bash
# Coming soon - K8s manifests in /k8s directory
kubectl apply -f k8s/
```

### Cloud Deployment

#### AWS ECS
- Use `deployment-package/` with ECS task definitions
- Configure Application Load Balancer
- Use RDS for Postgres, ElastiCache for Redis

#### Google Cloud Run
- Build container: `docker build -t gcr.io/PROJECT/concord-api .`
- Deploy: `gcloud run deploy concord-api --image gcr.io/PROJECT/concord-api`

#### Azure Container Instances
- Use Azure Container Registry
- Deploy with `az container create`

---

## 📚 Additional Resources

### Documentation
- [API Specification](API_SPECIFICATION.md) - Full API documentation
- [Technical Stack](TECHNICAL_STACK.md) - Technology details
- [README](README.md) - Project overview

### Scripts
- `deploy.sh` - Main deployment script
- `validate-env.sh` - Environment validation
- `quickstart.sh` - Interactive setup
- `Makefile` - Make commands

### Support
- Check logs: `./deploy.sh logs`
- Validate environment: `./validate-env.sh`
- Check status: `./deploy.sh status`

---

## 🎓 Common Use Cases

### Development
```bash
./deploy.sh local         # No Docker, instant startup
# or
./deploy.sh api-only      # Minimal Docker setup
```

### Testing
```bash
./deploy.sh core --build  # Rebuild with latest changes
./deploy.sh logs          # Watch logs
```

### Production
```bash
./validate-env.sh         # Pre-flight check
./deploy.sh full          # Full stack
# Monitor at http://localhost:3000 (Grafana)
```

### CI/CD Pipeline
```bash
./validate-env.sh         # Validation step
./deploy.sh api-only --build --no-cache  # Build & deploy
# Run integration tests
# Deploy to production
```

---

## ✅ Deployment Checklist

- [ ] Install Docker and Docker Compose
- [ ] Clone/download project
- [ ] Configure `.env` file with database credentials
- [ ] Run `./validate-env.sh` to check environment
- [ ] Run `./quickstart.sh` or `./deploy.sh api-only`
- [ ] Wait for services to start (~30 sec)
- [ ] Test health check: `curl http://localhost:8000/health`
- [ ] Open API docs: http://localhost:8000/docs
- [ ] Test recommendation endpoint
- [ ] Monitor logs: `./deploy.sh logs`

---

**Version:** 3.0.0
**Last Updated:** November 6, 2025
**Performance:** 75.4% precision@50, <400ms P99 latency

🚀 **Ready to deploy!** Start with `./quickstart.sh`
