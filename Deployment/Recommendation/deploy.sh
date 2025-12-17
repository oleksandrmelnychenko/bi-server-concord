#!/bin/bash
#
# Recommendation API Deployment Script
# =====================================
# This script deploys the Recommendation API with all dependencies.
#
# Usage:
#   ./deploy.sh              # Full deployment
#   ./deploy.sh --rebuild    # Force rebuild containers
#   ./deploy.sh --stop       # Stop all services
#   ./deploy.sh --status     # Check service status
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
DATA_DIR="$PROJECT_ROOT/data"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Recommendation API Deployment Script${NC}"
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
        echo "  - macOS: https://docs.docker.com/desktop/install/mac-install/"
        echo "  - Linux: https://docs.docker.com/engine/install/"
        echo "  - Windows: https://docs.docker.com/desktop/install/windows-install/"
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

check_env_file() {
    log_info "Checking environment configuration..."
    if [ ! -f "$ENV_FILE" ]; then
        log_error ".env file not found at $ENV_FILE"
        log_info "Creating .env from template..."
        if [ -f "$PROJECT_ROOT/.env.example" ]; then
            cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
            log_warn "Please edit $ENV_FILE with your database credentials"
            exit 1
        else
            log_error "No .env.example found. Please create .env manually."
            exit 1
        fi
    fi
    log_info "Environment file exists ✓"
}

setup_data_directory() {
    log_info "Setting up data directory..."
    mkdir -p "$DATA_DIR"
    log_info "Data directory ready: $DATA_DIR ✓"
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
    log_info "  - redis: Cache store (port 6380)"
    log_info "  - api: FastAPI recommendation service (port 8100)"
    log_info "  - worker: Scheduled jobs (daily ANN learning + weekly batch)"
}

generate_ann_index() {
    log_info "Checking ANN neighbors index..."
    if [ ! -f "$DATA_DIR/ann_neighbors.json" ]; then
        log_warn "ANN neighbors file not found. Generating..."

        # Start containers temporarily to generate ANN
        docker-compose -f "$COMPOSE_FILE" up -d redis
        sleep 5

        docker-compose -f "$COMPOSE_FILE" run --rm api python3 /app/scripts/build_ann_neighbors.py

        if [ -f "$DATA_DIR/ann_neighbors.json" ]; then
            log_info "ANN neighbors generated successfully ✓"
        else
            log_warn "ANN generation may have failed. Check logs."
        fi
    else
        log_info "ANN neighbors file exists ✓"
    fi
}

start_services() {
    log_info "Starting services..."
    cd "$SCRIPT_DIR"

    # Start all core services
    docker-compose -f "$COMPOSE_FILE" up -d
    log_info "Services starting..."
    log_info "  - redis: Starting cache store"
    log_info "  - api: Starting recommendation API"
    log_info "  - worker: Starting scheduled worker"

    # Wait for health checks
    log_info "Waiting for services to be healthy..."
    sleep 10

    # Check health
    for i in {1..30}; do
        if curl -s http://localhost:8100/health | grep -q '"status":"healthy"'; then
            log_info "API is healthy ✓"
            return 0
        fi
        sleep 2
    done

    log_warn "API health check timed out. Check logs with: docker logs recommendation-api"
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
    if curl -s http://localhost:8100/health 2>/dev/null | python3 -m json.tool 2>/dev/null; then
        echo ""
    else
        log_warn "API is not responding"
    fi

    echo -e "${BLUE}=== ANN Index ===${NC}"
    if [ -f "$DATA_DIR/ann_neighbors.json" ]; then
        SIZE=$(du -h "$DATA_DIR/ann_neighbors.json" | cut -f1)
        log_info "ANN file exists: $SIZE"
    else
        log_warn "ANN file not found"
    fi
}

show_endpoints() {
    echo ""
    echo -e "${BLUE}=== API Endpoints ===${NC}"
    echo "  Health Check:     http://localhost:8100/health"
    echo "  Recommendations:  http://localhost:8100/recommendations/{customer_id}"
    echo "  API Docs:         http://localhost:8100/docs"
    echo "  Redis:            localhost:6380"
    echo ""
    echo -e "${BLUE}=== Worker Schedules ===${NC}"
    echo "  Daily (2:00 AM):    ANN index regeneration - learns from new purchase data"
    echo "  Weekly (Sun 3 AM):  Batch recommendations - pre-computes for all customers"
    echo ""
    echo -e "${BLUE}=== Services ===${NC}"
    echo "  redis:  Cache store for recommendations"
    echo "  api:    FastAPI serving recommendations"
    echo "  worker: Background worker with scheduled jobs"
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
    --rebuild)
        check_docker
        check_docker_compose
        check_env_file
        setup_data_directory
        setup_dockerignore
        build_containers --rebuild
        start_services
        show_status
        show_endpoints
        ;;
    *)
        check_docker
        check_docker_compose
        check_env_file
        setup_data_directory
        setup_dockerignore
        build_containers
        start_services
        show_status
        show_endpoints
        ;;
esac

echo -e "${GREEN}Deployment complete!${NC}"
