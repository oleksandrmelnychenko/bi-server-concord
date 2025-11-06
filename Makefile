.PHONY: help build up down logs restart clean test ingest-data catalog-data

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker images
	docker-compose build

up: ## Start all services
	@echo "Starting Concord BI Server..."
	docker-compose up -d postgres redis minio
	@echo "Waiting for core services..."
	sleep 10
	docker-compose up -d minio-init mlflow ollama
	@echo "Starting data services..."
	sleep 10
	docker-compose up -d elasticsearch zookeeper broker
	sleep 30
	docker-compose up -d datahub-gms datahub-frontend
	@echo "Starting application services..."
	docker-compose up -d api dagster grafana
	@echo "All services started!"
	@make status

up-core: ## Start only core services (Postgres, Redis, MinIO, MLflow)
	docker-compose up -d postgres redis minio minio-init mlflow

up-api: ## Start API service only
	docker-compose up -d api

down: ## Stop all services
	docker-compose down

down-volumes: ## Stop all services and remove volumes (WARNING: deletes data)
	docker-compose down -v

logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show API logs
	docker-compose logs -f api

logs-dagster: ## Show Dagster logs
	docker-compose logs -f dagster

logs-datahub: ## Show DataHub logs
	docker-compose logs -f datahub-gms datahub-frontend

status: ## Show status of all services
	@echo "\n=== Service Status ===\n"
	@docker-compose ps
	@echo "\n=== Access URLs ===\n"
	@echo "  API Documentation:  http://localhost:8000/docs"
	@echo "  DataHub:            http://localhost:9002"
	@echo "  MLflow:             http://localhost:5000"
	@echo "  Dagster:            http://localhost:3001"
	@echo "  Grafana:            http://localhost:3000"
	@echo "  MinIO Console:      http://localhost:9001"
	@echo ""

restart: ## Restart all services
	docker-compose restart

restart-api: ## Restart API service
	docker-compose restart api

clean: ## Remove stopped containers
	docker-compose rm -f

ingest-data: ## Run MSSQL to Delta Lake ingestion
	@echo "Starting data ingestion from MSSQL to Delta Lake..."
	docker-compose exec dagster python /opt/dagster/app/pipelines/ingestion/mssql_to_delta.py

catalog-data: ## Run DataHub ingestion to catalog MSSQL tables
	@echo "Starting DataHub ingestion..."
	docker run --rm -it --network concord-bi-server_concord-network \
		-v $(PWD)/services/datahub:/opt/datahub \
		--env-file .env \
		-e DATAHUB_GMS_HOST=datahub-gms \
		-e DATAHUB_GMS_PORT=8080 \
		acryldata/datahub-ingestion:latest \
		datahub ingest -c /opt/datahub/mssql-recipe.yml

pull-llm: ## Pull Llama 3.1 model for NL queries
	@echo "Pulling Llama 3.1 8B model..."
	docker-compose exec ollama ollama pull llama3.1:8b

pull-llm-large: ## Pull Llama 3.1 70B model (requires GPU)
	@echo "Pulling Llama 3.1 70B model..."
	docker-compose exec ollama ollama pull llama3.1:70b

test-api: ## Test API endpoints
	@echo "Testing API health..."
	curl -s http://localhost:8000/api/v1/health | jq .
	@echo "\nTesting recommendations..."
	curl -s -X POST http://localhost:8000/api/v1/recommendations/predict \
		-H "Content-Type: application/json" \
		-d '{"customer_id": "CUST-001", "n_recommendations": 5}' | jq .
	@echo "\nTesting forecasting..."
	curl -s -X POST http://localhost:8000/api/v1/forecasting/predict \
		-H "Content-Type: application/json" \
		-d '{"periods": 7}' | jq .

shell-api: ## Open shell in API container
	docker-compose exec api /bin/bash

shell-dagster: ## Open shell in Dagster container
	docker-compose exec dagster /bin/bash

db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U concord -d concord_bi

init: ## Initialize the project (first time setup)
	@echo "Initializing Concord BI Server..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your MSSQL credentials."; \
		echo "Run 'make up' after configuring .env"; \
	else \
		echo ".env file already exists"; \
	fi
	@mkdir -p data/raw data/delta data/features
	@echo "Project initialized!"

backup-db: ## Backup PostgreSQL database
	@echo "Backing up PostgreSQL..."
	docker-compose exec -T postgres pg_dump -U concord concord_bi > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup complete!"

restore-db: ## Restore PostgreSQL database (requires BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Error: Please specify BACKUP_FILE=path/to/backup.sql"; \
		exit 1; \
	fi
	docker-compose exec -T postgres psql -U concord -d concord_bi < $(BACKUP_FILE)

health: ## Check health of all services
	@echo "Checking service health..."
	@echo "\nPostgreSQL:"
	@docker-compose exec postgres pg_isready -U concord || echo "  ❌ Not ready"
	@echo "\nRedis:"
	@docker-compose exec redis redis-cli ping || echo "  ❌ Not ready"
	@echo "\nAPI:"
	@curl -s http://localhost:8000/api/v1/health || echo "  ❌ Not ready"
	@echo "\nMLflow:"
	@curl -s http://localhost:5000/health || echo "  ❌ Not ready"

dev: ## Start in development mode with hot-reload
	docker-compose up api dagster

prod: ## Start in production mode
	docker-compose -f docker-compose.yml up -d

monitor: ## Show resource usage
	docker stats $(shell docker-compose ps -q)
