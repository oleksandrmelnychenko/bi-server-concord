#!/bin/bash

###############################################################################
# Concord BI Server - Automated Deployment Script
#
# This script automates the deployment of the Concord BI recommendation API
# with support for different deployment modes and environments.
#
# Usage:
#   ./deploy.sh [mode] [options]
#
# Modes:
#   api-only    - Deploy only API + Redis (minimal, fastest)
#   core        - Deploy core services (API, Redis, Postgres, MLflow)
#   full        - Deploy full stack (all services including DataHub, Grafana)
#   local       - Run API locally without Docker
#   stop        - Stop all running services
#   restart     - Restart all services
#   clean       - Stop and remove all containers and volumes
#
# Options:
#   --build     - Force rebuild of Docker images
#   --logs      - Follow logs after deployment
#   --no-cache  - Build without cache
#
# Examples:
#   ./deploy.sh api-only
#   ./deploy.sh full --build --logs
#   ./deploy.sh local
#   ./deploy.sh stop
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_dependencies() {
    print_header "Checking Dependencies"

    local missing_deps=0

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        missing_deps=1
    else
        print_success "Docker $(docker --version | cut -d' ' -f3) found"
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        missing_deps=1
    else
        print_success "Docker Compose $(docker-compose --version | cut -d' ' -f3) found"
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        missing_deps=1
    else
        print_success "Docker daemon is running"
    fi

    if [ $missing_deps -eq 1 ]; then
        print_error "Missing dependencies. Please install required tools."
        exit 1
    fi

    echo ""
}

check_env_file() {
    print_header "Checking Environment Configuration"

    if [ ! -f .env ]; then
        print_warning ".env file not found"
        print_info "Creating .env from .env.example..."
        cp .env.example .env
        print_success ".env file created"
        print_warning "Please edit .env file with your database credentials"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to exit and edit .env..."
    else
        print_success ".env file exists"
    fi

    # Check if critical env vars are set
    if grep -q "DB_PASSWORD=your_password_here" .env 2>/dev/null; then
        print_warning "Database password is still set to default"
        print_info "Consider updating DB_PASSWORD in .env file"
    fi

    echo ""
}

create_directories() {
    print_header "Creating Data Directories"

    mkdir -p data/raw
    mkdir -p data/delta
    mkdir -p data/features
    mkdir -p data/ml_features
    mkdir -p data/schema
    mkdir -p results
    mkdir -p models

    print_success "Data directories created"
    echo ""
}

###############################################################################
# Deployment Functions
###############################################################################

deploy_api_only() {
    print_header "Deploying API + Redis (Minimal Mode)"

    print_info "Starting Redis..."
    docker-compose up -d redis

    sleep 3

    print_info "Starting API..."
    if [ "$BUILD" = true ]; then
        docker-compose up -d --build api
    else
        docker-compose up -d api
    fi

    print_success "API + Redis deployed successfully!"
    echo ""

    show_api_status
}

deploy_core() {
    print_header "Deploying Core Services (API, Redis, Postgres, MLflow)"

    print_info "Starting Postgres..."
    docker-compose up -d postgres

    print_info "Starting Redis..."
    docker-compose up -d redis

    print_info "Starting MinIO..."
    docker-compose up -d minio

    sleep 5

    print_info "Initializing MinIO buckets..."
    docker-compose up -d minio-init

    sleep 3

    print_info "Starting MLflow..."
    docker-compose up -d mlflow

    sleep 5

    print_info "Starting API..."
    if [ "$BUILD" = true ]; then
        docker-compose up -d --build api
    else
        docker-compose up -d api
    fi

    print_success "Core services deployed successfully!"
    echo ""

    show_status
}

deploy_full() {
    print_header "Deploying Full Stack (All Services)"

    print_info "Starting core infrastructure..."
    docker-compose up -d postgres redis minio

    sleep 10

    print_info "Initializing storage..."
    docker-compose up -d minio-init

    sleep 5

    print_info "Starting ML services..."
    docker-compose up -d mlflow ollama

    sleep 10

    print_info "Starting data infrastructure..."
    docker-compose up -d elasticsearch zookeeper

    sleep 10

    docker-compose up -d broker

    sleep 15

    print_info "Starting DataHub..."
    docker-compose up -d datahub-gms datahub-frontend

    sleep 10

    print_info "Starting application services..."
    if [ "$BUILD" = true ]; then
        docker-compose up -d --build api dagster grafana
    else
        docker-compose up -d api dagster grafana
    fi

    print_success "Full stack deployed successfully!"
    echo ""

    show_status
}

deploy_local() {
    print_header "Running API Locally (Without Docker)"

    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi

    print_success "Python $(python3 --version | cut -d' ' -f2) found"

    # Check if Redis is running
    print_info "Checking if Redis is available..."
    if docker ps | grep -q concord-redis; then
        print_success "Redis container is running"
    else
        print_info "Starting Redis container..."
        docker-compose up -d redis
        sleep 3
    fi

    # Install dependencies
    print_info "Installing Python dependencies..."
    if [ -f "deployment-package/requirements.txt" ]; then
        pip3 install -q -r deployment-package/requirements.txt
    else
        print_warning "requirements.txt not found in deployment-package/"
    fi

    print_success "Dependencies installed"
    echo ""

    print_header "Starting API Server"
    print_info "API will be available at: http://localhost:8000"
    print_info "API docs will be at: http://localhost:8000/docs"
    print_info "Press Ctrl+C to stop"
    echo ""

    # Run the API
    cd "$SCRIPT_DIR"
    python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
}

stop_services() {
    print_header "Stopping All Services"

    docker-compose down

    print_success "All services stopped"
    echo ""
}

restart_services() {
    print_header "Restarting All Services"

    docker-compose restart

    print_success "All services restarted"
    echo ""

    show_status
}

clean_all() {
    print_header "Cleaning All Containers and Volumes"

    print_warning "This will remove all containers, networks, and volumes"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        print_info "Cleanup cancelled"
        exit 0
    fi

    docker-compose down -v

    print_success "All containers, networks, and volumes removed"
    echo ""
}

###############################################################################
# Status Display Functions
###############################################################################

show_api_status() {
    print_header "Service Status"

    docker-compose ps

    echo ""
    print_header "Access URLs"
    echo ""
    echo -e "  ${GREEN}API Documentation (Swagger):${NC}  http://localhost:8000/docs"
    echo -e "  ${GREEN}API ReDoc:${NC}                   http://localhost:8000/redoc"
    echo -e "  ${GREEN}API Health Check:${NC}            http://localhost:8000/health"
    echo -e "  ${GREEN}API Metrics:${NC}                 http://localhost:8000/metrics"
    echo ""

    # Wait for API to be ready
    print_info "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API is ready!"
            break
        fi
        sleep 1
    done
    echo ""
}

show_status() {
    print_header "Service Status"

    docker-compose ps

    echo ""
    print_header "Access URLs"
    echo ""
    echo -e "  ${GREEN}API Documentation:${NC}  http://localhost:8000/docs"
    echo -e "  ${GREEN}MLflow UI:${NC}          http://localhost:5001"
    echo -e "  ${GREEN}DataHub:${NC}            http://localhost:9002"
    echo -e "  ${GREEN}Grafana:${NC}            http://localhost:3000"
    echo -e "  ${GREEN}MinIO Console:${NC}      http://localhost:9001"
    echo -e "  ${GREEN}Dagster:${NC}            http://localhost:3001"
    echo ""

    # Quick health check
    print_info "Performing health checks..."

    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy"
    else
        print_warning "API is not responding yet"
    fi
    echo ""
}

show_logs() {
    print_header "Following Logs"
    print_info "Press Ctrl+C to stop following logs"
    echo ""
    docker-compose logs -f api
}

###############################################################################
# Main Script Logic
###############################################################################

show_usage() {
    cat << EOF
Usage: $0 [mode] [options]

Deployment Modes:
  api-only    Deploy only API + Redis (minimal, fastest)
  core        Deploy core services (API, Redis, Postgres, MLflow)
  full        Deploy full stack (all services)
  local       Run API locally without Docker
  stop        Stop all running services
  restart     Restart all services
  clean       Stop and remove all containers and volumes
  status      Show status of all services
  logs        Follow API logs

Options:
  --build     Force rebuild of Docker images
  --logs      Follow logs after deployment
  --no-cache  Build without using cache
  --help      Show this help message

Examples:
  $0 api-only
  $0 full --build --logs
  $0 local
  $0 stop
  $0 status

EOF
    exit 0
}

# Parse arguments
MODE="${1:-api-only}"
BUILD=false
FOLLOW_LOGS=false
NO_CACHE=""

shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --logs)
            FOLLOW_LOGS=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Build with no-cache if requested
if [ "$BUILD" = true ] && [ -n "$NO_CACHE" ]; then
    export COMPOSE_DOCKER_CLI_BUILD=1
    export DOCKER_BUILDKIT=1
fi

###############################################################################
# Execute Deployment
###############################################################################

print_header "Concord BI Server - Automated Deployment"
echo ""

# Pre-flight checks (skip for stop/clean commands)
if [ "$MODE" != "stop" ] && [ "$MODE" != "clean" ] && [ "$MODE" != "status" ] && [ "$MODE" != "logs" ]; then
    check_dependencies
    check_env_file
    create_directories
fi

# Execute requested mode
case $MODE in
    api-only)
        deploy_api_only
        ;;
    core)
        deploy_core
        ;;
    full)
        deploy_full
        ;;
    local)
        deploy_local
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    clean)
        clean_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        print_error "Unknown mode: $MODE"
        show_usage
        ;;
esac

# Follow logs if requested
if [ "$FOLLOW_LOGS" = true ] && [ "$MODE" != "local" ] && [ "$MODE" != "logs" ]; then
    show_logs
fi

print_header "Deployment Complete!"
print_success "Concord BI Server is ready to use"
echo ""
