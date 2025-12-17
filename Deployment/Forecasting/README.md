# Forecasting API Deployment

## Overview

The Forecasting service predicts future product demand based on historical purchase patterns using RFM (Recency, Frequency, Monetary) analysis and time-series forecasting.

## Prerequisites

### System Requirements
- Docker Desktop 4.0+ (with Docker Compose)
- 4GB RAM minimum (8GB recommended)
- 5GB disk space
- Network access to SQL Server database

### Required Software
| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Container orchestration |

## Quick Start

### Linux/macOS
```bash
cd Deployment/Forecasting
chmod +x deploy.sh
./deploy.sh
```

### Windows
```powershell
cd Deployment\Forecasting
.\deploy.ps1
```

## Deployment Options

### Linux/macOS
```bash
./deploy.sh              # Full deployment
./deploy.sh --rebuild    # Force rebuild containers
./deploy.sh --stop       # Stop all services
./deploy.sh --status     # Check status
./deploy.sh --run-now    # Generate forecasts immediately
```

### Windows
```powershell
.\deploy.ps1              # Full deployment
.\deploy.ps1 -Rebuild     # Force rebuild
.\deploy.ps1 -Stop        # Stop services
.\deploy.ps1 -Status      # Check status
.\deploy.ps1 -RunNow      # Generate forecasts immediately
```

## Configuration

### Environment Variables (.env)

```env
# Database Configuration
DB_SERVER=your-sql-server.database.windows.net
DB_PORT=1433
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# Forecast Configuration
FORECAST_CACHE_TTL=604800          # Cache TTL in seconds (7 days)
FORECAST_WORKERS=10                 # Parallel workers
FORECAST_MIN_CUSTOMERS=2            # Min customers for product
FORECAST_MIN_ORDERS=3               # Min orders for product
FORECAST_CRON_SCHEDULE=0 3 * * 0    # Weekly Sunday 3 AM
FORECAST_RUN_ON_STARTUP=false       # Run on container start
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/forecast/{product_id}` | GET | Get product forecast |
| `/docs` | GET | Swagger documentation |

### Forecast Response Example

```json
{
  "product_id": 12345,
  "forecast": {
    "next_week": 150,
    "next_month": 580,
    "next_quarter": 1650,
    "confidence": 0.85
  },
  "metrics": {
    "avg_weekly_sales": 145,
    "trend": "increasing",
    "seasonality": "low"
  },
  "generated_at": "2025-12-16T10:00:00Z"
}
```

## Scheduled Jobs

| Schedule | Job | Description |
|----------|-----|-------------|
| **Weekly Sunday 3:00 AM** | Forecast Generation | Regenerates forecasts for all products |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  recommendation │  │    forecast     │  │   redis     │ │
│  │      -api       │  │    -worker      │  │  (cache)    │ │
│  │   Port: 8100    │  │  (scheduled)    │  │ Port: 6380  │ │
│  │                 │  │                 │  │             │ │
│  │  /forecast/{id} │  │  Generates      │  │  Stores     │ │
│  │  endpoint       │  │  forecasts      │  │  forecasts  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           └────────────────────┴───────────────────┘        │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   SQL Server DB     │
                    │  (Order History)    │
                    └─────────────────────┘
```

## Forecasting Algorithm

The forecasting system uses:

1. **RFM Analysis**
   - Recency: Days since last purchase
   - Frequency: Purchase count
   - Monetary: Total spend

2. **Time Series**
   - Moving averages
   - Trend detection
   - Seasonal adjustment

3. **Aggregation**
   - Product-level forecasts
   - Customer segment weighting
   - Confidence intervals

## Verification

```bash
# Check health
curl http://localhost:8100/health

# Get product forecast
curl http://localhost:8100/forecast/12345

# View worker logs
docker logs forecast-worker

# Check cache
docker exec recommendation-redis redis-cli KEYS "forecast:*"
```

## Troubleshooting

### No forecasts generated
```bash
# Check worker logs
docker logs forecast-worker --tail 100

# Run manually
docker compose run --rm forecast-worker python3 scripts/forecasting/forecast_worker.py
```

### Database connection issues
1. Verify `.env` credentials
2. Check network/firewall rules
3. Ensure SQL Server allows connections

### Cache issues
```bash
# Clear forecast cache
docker exec recommendation-redis redis-cli KEYS "forecast:*" | xargs docker exec recommendation-redis redis-cli DEL

# Regenerate forecasts
./deploy.sh --run-now
```

## Files Structure

```
Deployment/Forecasting/
├── deploy.sh           # Linux/macOS deployment script
├── deploy.ps1          # Windows deployment script
├── README.md           # This file
└── requirements.txt    # Python dependencies reference

scripts/forecasting/
├── __init__.py
├── forecast_worker.py  # Main worker script
├── trigger_relearn.py  # Manual trigger
├── core/
│   ├── forecast_engine.py
│   ├── pattern_analyzer.py
│   └── product_aggregator.py
└── utils/
```

## Integration with Recommendations

The Forecasting service works alongside the Recommendation API:

- **Recommendations** → What products to suggest to customers
- **Forecasting** → How much of each product will be needed

Both services share:
- Same Redis cache (port 6380)
- Same SQL Server database
- Same Docker network
