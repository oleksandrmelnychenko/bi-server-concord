# Concord BI Server

Enterprise AI-powered Business Intelligence platform for on-premises deployment.

## Features

- **Automated Data Discovery**: DataHub automatically catalogs 300+ database tables
- **Delta Lake Storage**: Versioned, ACID-compliant data lake
- **AI/ML Capabilities**:
  - Personalized product recommendations
  - Sales forecasting
  - Natural language query interface
- **MLOps**: Experiment tracking, model versioning, automated deployment
- **REST API**: FastAPI-based model serving
- **Monitoring**: Grafana dashboards for observability

## Architecture

```
MSSQL (313 tables, 300GB)
    ↓
[Dagster Pipelines]
    ↓
Delta Lake (Parquet + versioning)
    ↓
┌─────────┬──────────┬─────────┐
│ DataHub │   dbt    │  Feast  │
│ Catalog │Transform │Features │
└─────────┴──────────┴─────────┘
    ↓
┌──────────────────────────────┐
│       ML Models              │
│  - Recommendations           │
│  - Sales Forecasting         │
│  - NL Query (LLM)           │
└──────────────────────────────┘
    ↓
FastAPI + Redis
    ↓
Production Applications
```

## Technology Stack

- **Orchestration**: Dagster
- **Data Lake**: Delta Lake (delta-rs)
- **Data Catalog**: DataHub
- **Transformations**: dbt
- **ML Framework**: scikit-learn, PyTorch, Prophet, LightFM
- **LLM**: Llama 3.1 (via Ollama)
- **API**: FastAPI
- **Storage**: MinIO (S3-compatible)
- **Cache**: Redis
- **Monitoring**: Grafana + Prometheus
- **Deployment**: Docker + Docker Compose

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 32+ GB RAM
- 100+ GB available disk space
- NVIDIA GPU (optional, for LLM acceleration)

### Installation

1. Clone and configure:
```bash
cd /Users/oleksandrmelnychenko/Projects/Concord-BI-Server
cp .env.example .env
# Edit .env with your MSSQL credentials
```

2. Start core services:
```bash
docker-compose up -d postgres redis minio
```

3. Start DataHub:
```bash
docker-compose up -d datahub-gms datahub-frontend
```

4. Start MLflow:
```bash
docker-compose up -d mlflow
```

5. Start API:
```bash
docker-compose up -d api
```

### Access Services

- **DataHub UI**: http://localhost:9002
- **MLflow UI**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000

## Project Structure

```
Concord-BI-Server/
├── services/          # Dockerized services
│   ├── api/          # FastAPI service
│   ├── datahub/      # DataHub configuration
│   ├── dagster/      # Workflow orchestration
│   ├── mlflow/       # ML experiment tracking
│   └── models/       # Model training scripts
├── pipelines/        # Data pipelines
│   ├── ingestion/    # MSSQL → Delta Lake
│   ├── dbt/          # SQL transformations
│   └── features/     # Feature engineering
├── models/           # ML model implementations
│   ├── recommendations/
│   ├── forecasting/
│   └── nlq/          # Natural language queries
├── data/             # Data storage (gitignored)
│   ├── raw/
│   ├── delta/
│   └── features/
└── docs/             # Documentation

```

## Development

### Running Pipelines

```bash
# Ingest data from MSSQL
docker-compose run dagster dagster pipeline execute -p mssql_ingestion

# Run dbt transformations
docker-compose run dbt dbt run

# Train recommendation model
docker-compose run models python -m models.recommendations.train
```

### Adding New Models

See `docs/adding-models.md`

## Production Deployment

See `docs/deployment.md`

## License

Proprietary - Concord Enterprise
