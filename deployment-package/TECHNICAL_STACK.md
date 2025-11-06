# B2B Product Recommendation API - Complete Technical Stack

## Overview

Production-ready B2B product recommendation system achieving 29.2% precision@50 using rule-based hybrid algorithms optimized for deterministic purchasing patterns in the truck parts domain.

---

## Core Technologies

### Programming Language

**Python 3.11**
- Primary language for all backend services
- Type hints for better code quality
- Async/await support for concurrent operations

---

## Web Framework & API

### FastAPI 0.104.1
**Why FastAPI:**
- Automatic OpenAPI/Swagger documentation generation
- Native async support for high-performance APIs
- Pydantic integration for request/response validation
- Built-in dependency injection
- Type safety with Python type hints

**Features Used:**
- Automatic interactive API docs (Swagger UI)
- ReDoc alternative documentation
- Request validation with Pydantic models
- Dependency injection for database connections
- CORS middleware
- Custom middleware for timing and metrics
- Health check endpoints
- Lifespan events for startup/shutdown

### Uvicorn 0.24.0
**ASGI Server:**
- Production-grade ASGI server
- Multi-worker support (4 workers in production)
- WebSocket support (for future real-time features)
- Graceful shutdown handling

---

## Database Layer

### Microsoft SQL Server
**Production Database:**
- Version: SQL Server (ConcordDb_v5)
- Connection: TCP/IP on port 1433
- Tables used:
  - `Client` - Customer information
  - `ClientAgreement` - Customer agreements
  - `Order` - Order transactions
  - `OrderItem` - Line items with product details
  - `Product` - Product catalog
  - `ProductCarBrand` - Product-to-brand mapping (261,800 records)
  - `CarBrand` - Vehicle brand information

### pymssql 2.2.10
**Database Connector:**
- Python DB-API 2.0 compliant interface
- Native Microsoft SQL Server driver
- Efficient connection pooling
- Transaction support
- Parameterized queries for SQL injection prevention

**Connection Pooling:**
- Custom implementation using `queue.Queue`
- Pool size: 20 connections
- Max overflow: 10 connections
- Idle timeout: 300 seconds
- Connection validation on checkout

---

## Caching Layer

### Redis 7-alpine
**In-Memory Cache:**
- High-performance key-value store
- Used for caching recommendation results
- TTL support for automatic expiration
- Pub/Sub capabilities (for future features)
- Persistence to disk for reliability

**Configuration:**
- Host: redis (Docker network) / localhost (local dev)
- Port: 6379
- DB: 0
- Memory policy: allkeys-lru

### redis-py 5.0.1
**Python Redis Client:**
- Async and sync support
- Connection pooling
- Pipeline support for batch operations
- Serialization with JSON

---

## Data Validation & Schemas

### Pydantic 2.4.2
**Data Validation:**
- Request/response model validation
- Type coercion and validation
- Custom validators for business logic
- JSON schema generation
- Environment variable parsing with pydantic-settings

**Models Defined:**
- `RecommendationRequest` - API input validation
- `RecommendationResponse` - API output schema
- `RecommendationItem` - Individual recommendation structure
- Configuration models for environment variables

---

## Recommendation Engine

### Custom V3 Hybrid Recommender
**Architecture:** Rule-Based with Weighted Scoring

**Core Algorithm:**
```
1. Customer Segmentation:
   - HEAVY: 500+ orders
   - REGULAR: 100-500 orders (CONSISTENT vs EXPLORATORY)
   - LIGHT: <100 orders

2. Scoring Components:
   - Frequency Score: Purchase count weighting
   - Recency Score: Time decay function
   - Monetary Score: Purchase value (optional)

3. Segment-Specific Weights:
   - HEAVY: 60% frequency, 25% recency, 15% other
   - REGULAR-CONSISTENT: 50% frequency, 35% recency, 15% other
   - REGULAR-EXPLORATORY: 25% frequency, 50% recency, 25% other
   - LIGHT: 70% frequency, 30% recency
```

**Performance Metrics:**
- Overall Precision@50: 29.2%
- HEAVY users: 57.7%
- REGULAR users: 40.0%
- LIGHT users: 16.1%

**Data Processing:**
- SQL-based feature extraction
- In-memory weighted scoring
- NumPy for numerical operations
- Custom sorting and ranking algorithms

---

## Machine Learning Experiments (Not in Production)

**Attempted Approaches (All Failed):**

### PyTorch 2.1.0
- Neural Collaborative Filtering
- Deep learning embeddings
- Custom loss functions
- **Result:** Overfitting, poor generalization

### PyTorch Geometric 2.4.0
- Graph Neural Networks (GNN)
- LightGCN architecture
- Node embeddings for customers and products
- BPR (Bayesian Personalized Ranking) loss
- **Result:** 14-27% precision (vs 29.2% baseline)

### LightFM
- Hybrid collaborative filtering
- Content-based features
- WARP loss function
- **Result:** Failed to converge

**Why ML Failed:**
- B2B domain is too deterministic
- Purchasing follows maintenance cycles, not exploration
- ML learns collaborative patterns (customer similarity)
- Domain needs temporal patterns (maintenance schedules)
- Simple rules outperform complex ML models

---

## HTTP & Networking

### httpx 0.25.1
**Async HTTP Client:**
- Used for external API calls
- Connection pooling
- Timeout configuration
- HTTP/2 support

### requests 2.31.0
**Synchronous HTTP Client:**
- Used in validation scripts
- Simple API for testing
- Session management

---

## Environment & Configuration

### python-dotenv 1.0.0
**Environment Management:**
- Load `.env` files
- Environment variable parsing
- Separate configs for dev/staging/prod

**Environment Variables:**
```bash
# Database
DB_SERVER=78.152.175.67
DB_PORT=1433
DB_NAME=ConcordDb_v5
DB_USER=ef_migrator
DB_PASSWORD=***

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# API
API_PORT=8000
API_WORKERS=4
```

---

## Containerization

### Docker 24.x
**Container Runtime:**
- Multi-stage builds (not used, single-stage for simplicity)
- Non-root user for security
- Health checks built-in
- Volume mounts for persistence

**Dockerfile Highlights:**
```dockerfile
FROM python:3.11-slim
- System dependencies (freetds-dev for SQL Server)
- Python dependencies from requirements.txt
- Non-root user (appuser, UID 1000)
- Health check endpoint
- 4 Uvicorn workers
```

### Docker Compose 3.8
**Multi-Container Orchestration:**
- API service
- Redis service
- Shared network
- Volume persistence
- Automatic restarts
- Environment variable injection

---

## Logging & Monitoring

### Python logging
**Standard Library:**
- Structured logging
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Request timing middleware
- Performance metrics

### Custom Metrics
**API Metrics:**
- Request count
- Cache hit rate
- Error rate
- Average latency
- P50, P95, P99 percentiles

---

## Development Tools

### Type Checking
- Python type hints throughout codebase
- mypy for static type checking (optional)

### Code Quality
- PEP 8 style guide
- Black formatter (optional)
- isort for import sorting

### Testing (Validation)
- Custom validation scripts
- Temporal validation (point-in-time split)
- Precision@50 metric
- Customer segmentation analysis

---

## System Dependencies

### FreeTDS
**SQL Server Protocol:**
- Required for pymssql
- Handles TDS (Tabular Data Stream) protocol
- Version 1.x compatible with SQL Server

### System Libraries
```
gcc, g++ - C compilers for building Python extensions
freetds-dev - SQL Server client library headers
freetds-bin - SQL Server client binaries
unixodbc-dev - ODBC headers
```

---

## Data Flow Architecture

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │ HTTP POST /recommend
       ▼
┌─────────────┐
│   FastAPI   │◄──────┐
│   Uvicorn   │       │ Health Check
└──────┬──────┘       │ Metrics
       │              │
       ├──────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│    Redis    │◄────┤ Cache Layer │
│   (Cache)   │     └─────────────┘
└─────────────┘
       │
       │ Cache Miss
       ▼
┌─────────────┐
│ Connection  │
│    Pool     │
│  (20 conn)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ V3 Hybrid   │
│ Recommender │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   SQL       │
│  Queries    │
│  (5-10)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  SQL Server │
│ ConcordDb   │
└─────────────┘
       │
       │ Product IDs,
       │ Purchase History,
       │ Frequencies
       ▼
┌─────────────┐
│  Weighted   │
│  Scoring    │
│  Engine     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Top 50    │
│   Products  │
│   (JSON)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Response   │
│   to User   │
└─────────────┘
```

---

## Performance Characteristics

### API Response Times
- Average: ~800ms
- P50: ~500ms
- P95: ~1500ms
- P99: ~2000ms

**Bottlenecks:**
- Database queries (primary)
- Scoring computation (secondary)
- Network latency (tertiary)

### Database Query Performance
- Customer profile: 50-100ms
- Purchase history: 100-300ms
- Product details: 50-100ms
- Total: 200-500ms

### Scalability
- **Horizontal:** Add more Uvicorn workers
- **Vertical:** Increase connection pool size
- **Caching:** Redis reduces load by 70%

---

## Production Deployment

### Infrastructure Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB
- Network: 100 Mbps

**Recommended:**
- CPU: 4+ cores
- RAM: 8 GB
- Disk: 20 GB SSD
- Network: 1 Gbps

### Docker Deployment
```bash
# Build
docker-compose build

# Start
docker-compose up -d

# Scale workers
docker-compose up -d --scale api=4

# Monitor
docker-compose logs -f api

# Stop
docker-compose down
```

---

## Security Considerations

### Implemented
- Non-root Docker user
- Parameterized SQL queries (SQL injection prevention)
- Connection pooling (prevents connection exhaustion)
- Environment variables for secrets
- CORS middleware

### TODO (Before Production)
- [ ] API authentication (JWT, OAuth2, API keys)
- [ ] Rate limiting
- [ ] HTTPS/TLS
- [ ] IP whitelisting
- [ ] Input sanitization
- [ ] Audit logging
- [ ] Secret management (Vault, AWS Secrets Manager)

---

## Version History

### v3.0.0 (Current - Production)
- Rule-based hybrid recommender
- 29.2% overall precision
- FastAPI with Swagger
- Docker deployment
- Redis caching

### v2.x (Deprecated)
- Earlier recommendation approaches

### v4.0 (Attempted - Failed)
- Collaborative filtering
- 21.2% precision
- Abandoned

### v3.5 (Attempted - Failed)
- Brand affinity boosting
- 12.6% precision for LIGHT users
- Abandoned

---

## Future Enhancements

### Planned
1. Real-time recommendations via WebSockets
2. A/B testing framework
3. Product catalog integration
4. Multi-tenancy support
5. GraphQL API
6. Kubernetes deployment

### Under Consideration
1. Hybrid approach (V3 + lightweight ML for specific segments)
2. Time-series forecasting for maintenance cycles
3. Product complementarity detection
4. Customer lifetime value prediction

---

## License & Credits

**Technology Stack:**
- FastAPI: MIT License
- Python: PSF License
- Redis: BSD License
- Docker: Apache 2.0 License

**Developed By:** Concord BI Team
**Date:** November 2025
**Version:** 3.0.0

---

## References

### Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Redis: https://redis.io/documentation
- Docker: https://docs.docker.com/
- PyMSSQL: http://pymssql.org/

### Papers & Articles
- "Hybrid Recommendation Systems" (2007)
- "Temporal Collaborative Filtering" (2010)
- "B2B vs B2C Recommendation Systems" (2018)

---

## Contact & Support

For technical questions or deployment assistance:
- API Documentation: http://localhost:8000/docs
- Repository: [Link to repo]
- Email: [support@example.com]

---

**Last Updated:** November 4, 2025
**Document Version:** 1.0
