#!/bin/bash
#
# Forecasting API Deployment Script
# ==================================
# This script deploys the Forecasting API with all dependencies.
#
# Usage:
#   ./deploy.sh              # Full deployment
#   ./deploy.sh --rebuild    # Force rebuild containers
#   ./deploy.sh --stop       # Stop all services
#   ./deploy.sh --status     # Check service status
#   ./deploy.sh --run-worker # Run forecast worker manually
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Forecasting API Deployment Script${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    log_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    fi

    log_info "Docker is installed and running ✓"
}

check_docker_compose() {
    log_info "Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed."
        exit 1
    fi
    log_info "Docker Compose is available ✓"
}

setup_dockerignore() {
    log_info "Setting up .dockerignore..."
    cat > "$PROJECT_ROOT/.dockerignore" << 'EOF'
db-ai-api/
db-rag-system/
*.log
__pycache__/
.git/
*.pyc
.DS_Store
.env.example
*.md
docs/
sample_forecasts/
Deployment/
EOF
    log_info ".dockerignore configured ✓"
}

build_containers() {
    log_info "Building Docker containers..."
    cd "$SCRIPT_DIR"

    # Fix Docker credentials if needed
    if [ -f ~/.docker/config.json ]; then
        if grep -q "credsStore" ~/.docker/config.json 2>/dev/null; then
            echo '{}' > ~/.docker/config.json
            log_info "Fixed Docker credentials config"
        fi
    fi

    if [ "$1" == "--rebuild" ]; then
        log_info "Force rebuilding containers (--no-cache)..."
        docker-compose -f "$COMPOSE_FILE" build --no-cache
    else
        docker-compose -f "$COMPOSE_FILE" build
    fi
    log_info "Containers built successfully ✓"
    log_info "  - redis: Cache store (port 6381)"
    log_info "  - api: FastAPI forecasting service (port 8101)"
    log_info "  - forecast-worker: Batch forecast generator"
}

start_services() {
    log_info "Starting services..."
    cd "$SCRIPT_DIR"

    # Start redis and api (not forecast-worker which runs on-demand)
    docker-compose -f "$COMPOSE_FILE" up -d redis api
    log_info "Services starting..."
    log_info "  - redis: Starting cache store"
    log_info "  - api: Starting forecasting API"

    # Wait for health checks
    log_info "Waiting for services to be healthy..."
    sleep 10

    # Check health
    for i in {1..30}; do
        if curl -s http://localhost:8101/health | grep -q '"status":"healthy"'; then
            log_info "API is healthy ✓"
            return 0
        fi
        sleep 2
    done

    log_warn "API health check timed out. Check logs with: docker logs forecast-api"
}

run_forecast_worker() {
    log_info "Running forecast worker..."
    cd "$SCRIPT_DIR"

    # Ensure redis is running
    docker-compose -f "$COMPOSE_FILE" up -d redis
    sleep 5

    log_info "Starting forecast worker (this may take 20-40 minutes for all products)..."
    docker-compose -f "$COMPOSE_FILE" run --rm forecast-worker

    log_info "Forecast worker completed ✓"
}

stop_services() {
    log_info "Stopping services..."
    cd "$SCRIPT_DIR"
    docker-compose -f "$COMPOSE_FILE" down
    log_info "Services stopped ✓"
}

show_status() {
    echo ""
    echo -e "${BLUE}=== Service Status ===${NC}"
    cd "$SCRIPT_DIR"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""

    echo -e "${BLUE}=== API Health ===${NC}"
    if curl -s http://localhost:8101/health 2>/dev/null | python3 -m json.tool 2>/dev/null; then
        echo ""
    else
        log_warn "API is not responding"
    fi

    echo -e "${BLUE}=== Cached Forecasts ===${NC}"
    FORECAST_COUNT=$(docker exec forecast-redis redis-cli keys "forecast:product:*" 2>/dev/null | wc -l || echo "0")
    log_info "Cached forecasts in Redis: $FORECAST_COUNT"
}

show_endpoints() {
    echo ""
    echo -e "${BLUE}=== API Endpoints ===${NC}"
    echo "  Health Check:     http://localhost:8101/health"
    echo "  Forecast:         http://localhost:8101/forecast/{product_id}"
    echo "  API Docs:         http://localhost:8101/docs"
    echo "  Redis:            localhost:6381"
    echo ""
    echo -e "${BLUE}=== Services ===${NC}"
    echo "  redis:           Cache store for forecasts (7-day TTL)"
    echo "  api:             FastAPI serving product forecasts"
    echo "  forecast-worker: Batch worker (runs on-demand)"
    echo ""
    echo -e "${BLUE}=== Run Forecast Worker ===${NC}"
    echo "  ./deploy.sh --run-worker    # Generate forecasts for all products"
    echo ""
}

# Main execution
case "$1" in
    --stop)
        stop_services
        ;;
    --status)
        show_status
        ;;
    --run-worker)
        run_forecast_worker
        ;;
    --rebuild)
        check_docker
        check_docker_compose
        setup_dockerignore
        build_containers --rebuild
        start_services
        show_status
        show_endpoints
        ;;
    *)
        check_docker
        check_docker_compose
        setup_dockerignore
        build_containers
        start_services
        show_status
        show_endpoints
        ;;
esac

echo -e "${GREEN}Deployment complete!${NC}"
