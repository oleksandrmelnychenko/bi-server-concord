# B2B Product Recommendation API - Docker Deployment Package

## Quick Start

1. **Configure Environment:**
```bash
cp .env.example .env
# Edit .env with your database credentials
```

2. **Build and Run:**
```bash
docker-compose up -d
```

3. **Access API:**
- Swagger UI: http://localhost:8000/docs
- API: http://localhost:8000
- Health: http://localhost:8000/health

## Configuration

Edit `.env`:
```
DB_SERVER=your-db-server
DB_PORT=1433
DB_NAME=ConcordDb_v5
DB_USER=your-username
DB_PASSWORD=your-password
```

## Performance

- Model: V3 Hybrid Recommender
- Overall Precision: 29.2%
- HEAVY users: 57.7%
- REGULAR users: 40.0%
- LIGHT users: 16.1%

## API Endpoints

- `POST /recommend` - Get recommendations
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `GET /docs` - Swagger documentation

##  Stopping

```bash
docker-compose down
```

## Logs

```bash
docker-compose logs -f api
```

For detailed usage examples, see `API_USAGE_EXAMPLES.md`
