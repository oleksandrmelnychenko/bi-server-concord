# Recommendation API Deployment

## Prerequisites

### System Requirements
- Docker Desktop 4.0+ (with Docker Compose)
- 4GB RAM minimum (8GB recommended)
- 10GB disk space
- Network access to SQL Server database

### Required Software
| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Container orchestration |

## Quick Start

```bash
# 1. Navigate to deployment folder
cd Deployment/Recommendation

# 2. Make script executable
chmod +x deploy.sh

# 3. Run deployment
./deploy.sh
```

## Deployment Options

```bash
# Full deployment (build + start)
./deploy.sh

# Force rebuild containers
./deploy.sh --rebuild

# Stop all services
./deploy.sh --stop

# Check status
./deploy.sh --status
```

## Configuration

### Environment Variables (.env)

Create or edit `.env` in the project root with:

```env
# Database Configuration
DB_SERVER=your-sql-server.database.windows.net
DB_PORT=1433
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# Redis Configuration (optional - uses defaults)
REDIS_HOST=redis
REDIS_PORT=6379

# API Configuration
API_PORT=8000
API_WORKERS=4
```

### ANN Collaborative Filtering

The ANN (Approximate Nearest Neighbors) index enables collaborative filtering recommendations:

```bash
# Generate ANN index manually (if needed)
docker exec recommendation-api python3 /app/scripts/build_ann_neighbors.py
```

The index file is stored at `data/ann_neighbors.json` (~35MB for 3000+ products).

## Services

| Service | Port | Description |
|---------|------|-------------|
| recommendation-api | 8000 | Main API server |
| recommendation-worker | - | Background job processor |
| recommendation-redis | 6379 | Cache storage |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/recommendations/{customer_id}` | GET | Get recommendations |
| `/forecast/{product_id}` | GET | Get product forecast |
| `/docs` | GET | Swagger documentation |

## Verification

```bash
# Check health
curl http://localhost:8000/health

# Test recommendations
curl http://localhost:8000/recommendations/1234

# View logs
docker logs recommendation-api
docker logs recommendation-worker
```

## Troubleshooting

### Docker not starting
```bash
# Check Docker daemon
docker info

# Restart Docker Desktop (macOS)
osascript -e 'quit app "Docker"' && open -a Docker
```

### API errors
```bash
# Check logs
docker logs recommendation-api --tail 100

# Restart API
docker-compose restart api
```

### Database connection issues
1. Verify `.env` credentials
2. Check network/firewall rules
3. Ensure SQL Server allows connections

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  recommendation │  │  recommendation │  │   redis     │ │
│  │      -api       │  │    -worker      │  │  (cache)    │ │
│  │   Port: 8000    │  │   (cron jobs)   │  │ Port: 6379  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           └────────────────────┴───────────────────┘        │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   SQL Server DB     │
                    │  (External/Azure)   │
                    └─────────────────────┘
```

## Files Structure

```
Deployment/Recommendation/
├── deploy.sh           # Main deployment script
├── README.md           # This file
└── requirements.txt    # Python dependencies reference

Project Root/
├── docker-compose.yml  # Service definitions
├── Dockerfile.api      # API container
├── Dockerfile.worker   # Worker container
├── requirements.txt    # Python packages
├── .env               # Environment config
└── data/
    └── ann_neighbors.json  # ANN index (generated)
```
