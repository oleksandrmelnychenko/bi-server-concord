#!/bin/bash

###############################################################################
# Environment Validation Script
#
# Validates the deployment environment and configuration before deployment
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS+1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS+1))
}

print_header "Concord BI Server - Environment Validation"
echo ""

###############################################################################
# Check System Requirements
###############################################################################

print_header "System Requirements"

# Check OS
OS=$(uname -s)
check_pass "Operating System: $OS"

# Check CPU cores
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "unknown")
if [ "$CPU_CORES" != "unknown" ]; then
    if [ "$CPU_CORES" -ge 2 ]; then
        check_pass "CPU Cores: $CPU_CORES (minimum: 2)"
    else
        check_warn "CPU Cores: $CPU_CORES (recommended: 4+)"
    fi
else
    check_warn "Could not detect CPU cores"
fi

# Check available memory
if [ "$OS" = "Darwin" ]; then
    TOTAL_MEM_MB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
else
    TOTAL_MEM_MB=$(( $(free -m | awk '/^Mem:/{print $2}') ))
fi

if [ "$TOTAL_MEM_MB" -ge 4096 ]; then
    check_pass "Memory: ${TOTAL_MEM_MB}MB (minimum: 4GB)"
else
    check_fail "Memory: ${TOTAL_MEM_MB}MB (minimum required: 4GB)"
fi

# Check available disk space
DISK_AVAIL_GB=$(df -h . | awk 'NR==2 {print $4}' | sed 's/Gi*//')
check_pass "Available Disk Space: ${DISK_AVAIL_GB}GB"

echo ""

###############################################################################
# Check Required Software
###############################################################################

print_header "Required Software"

# Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | sed 's/,//')
    check_pass "Docker: $DOCKER_VERSION"

    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        check_pass "Docker daemon is running"

        # Check Docker resources
        DOCKER_MEM=$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
        if [ -n "$DOCKER_MEM" ]; then
            if [ "$DOCKER_MEM" -ge 4 ]; then
                check_pass "Docker Memory: ${DOCKER_MEM}GB"
            else
                check_warn "Docker Memory: ${DOCKER_MEM}GB (recommended: 8GB+)"
            fi
        fi
    else
        check_fail "Docker daemon is not running"
    fi
else
    check_fail "Docker is not installed"
fi

# Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')
    check_pass "Docker Compose: $COMPOSE_VERSION"
else
    check_fail "Docker Compose is not installed"
fi

# Python 3
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    check_pass "Python: $PYTHON_VERSION"
else
    check_warn "Python 3 is not installed (required for local mode)"
fi

# Git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    check_pass "Git: $GIT_VERSION"
else
    check_warn "Git is not installed"
fi

# curl
if command -v curl &> /dev/null; then
    check_pass "curl is installed"
else
    check_warn "curl is not installed (needed for health checks)"
fi

echo ""

###############################################################################
# Check Configuration Files
###############################################################################

print_header "Configuration Files"

# Check .env file
if [ -f .env ]; then
    check_pass ".env file exists"

    # Validate critical environment variables
    if grep -q "^DB_SERVER=" .env; then
        DB_SERVER=$(grep "^DB_SERVER=" .env | cut -d'=' -f2)
        check_pass "DB_SERVER is set: $DB_SERVER"
    else
        check_fail "DB_SERVER is not set in .env"
    fi

    if grep -q "^DB_NAME=" .env; then
        DB_NAME=$(grep "^DB_NAME=" .env | cut -d'=' -f2)
        check_pass "DB_NAME is set: $DB_NAME"
    else
        check_fail "DB_NAME is not set in .env"
    fi

    if grep -q "^DB_USER=" .env; then
        check_pass "DB_USER is set"
    else
        check_fail "DB_USER is not set in .env"
    fi

    if grep -q "^DB_PASSWORD=" .env; then
        DB_PASSWORD=$(grep "^DB_PASSWORD=" .env | cut -d'=' -f2)
        if [ "$DB_PASSWORD" = "your_password_here" ] || [ -z "$DB_PASSWORD" ]; then
            check_warn "DB_PASSWORD is not configured (still default)"
        else
            check_pass "DB_PASSWORD is configured"
        fi
    else
        check_fail "DB_PASSWORD is not set in .env"
    fi

else
    check_fail ".env file not found"
    echo "  Run: cp .env.example .env"
fi

# Check docker-compose.yml
if [ -f docker-compose.yml ]; then
    check_pass "docker-compose.yml exists"
else
    check_fail "docker-compose.yml not found"
fi

# Check Makefile
if [ -f Makefile ]; then
    check_pass "Makefile exists"
else
    check_warn "Makefile not found (optional)"
fi

echo ""

###############################################################################
# Check Project Structure
###############################################################################

print_header "Project Structure"

# Check critical directories
for dir in api scripts config data deployment-package; do
    if [ -d "$dir" ]; then
        check_pass "Directory exists: $dir/"
    else
        check_fail "Missing directory: $dir/"
    fi
done

# Check API files
if [ -f api/main.py ]; then
    check_pass "API main file exists: api/main.py"
else
    check_fail "Missing API main file: api/main.py"
fi

# Check requirements
if [ -f deployment-package/requirements.txt ]; then
    check_pass "Requirements file exists"
else
    check_warn "Missing requirements.txt in deployment-package/"
fi

echo ""

###############################################################################
# Check Network Connectivity
###############################################################################

print_header "Network Connectivity"

# Check database connectivity
if [ -f .env ]; then
    DB_SERVER=$(grep "^DB_SERVER=" .env | cut -d'=' -f2 2>/dev/null || echo "")
    DB_PORT=$(grep "^DB_PORT=" .env | cut -d'=' -f2 2>/dev/null || echo "1433")

    if [ -n "$DB_SERVER" ] && [ "$DB_SERVER" != "your_db_server" ]; then
        if command -v nc &> /dev/null; then
            if timeout 3 nc -z "$DB_SERVER" "$DB_PORT" 2>/dev/null; then
                check_pass "Database server is reachable: $DB_SERVER:$DB_PORT"
            else
                check_warn "Database server is not reachable: $DB_SERVER:$DB_PORT"
            fi
        else
            check_warn "netcat (nc) not available - skipping database connectivity test"
        fi
    else
        check_warn "Database server not configured yet"
    fi
fi

# Check if ports are available
for port in 8000 6379 5433 5001 9002 3000; do
    if command -v lsof &> /dev/null; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            check_warn "Port $port is already in use"
        else
            check_pass "Port $port is available"
        fi
    fi
done

echo ""

###############################################################################
# Check Docker Resources
###############################################################################

if docker info &> /dev/null 2>&1; then
    print_header "Docker Resources"

    # Check running containers
    RUNNING_CONTAINERS=$(docker ps -q | wc -l)
    check_pass "Running containers: $RUNNING_CONTAINERS"

    # Check Docker images
    IMAGE_COUNT=$(docker images -q | wc -l)
    check_pass "Docker images: $IMAGE_COUNT"

    # Check Docker volumes
    VOLUME_COUNT=$(docker volume ls -q | wc -l)
    check_pass "Docker volumes: $VOLUME_COUNT"

    echo ""
fi

###############################################################################
# Summary
###############################################################################

print_header "Validation Summary"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "${GREEN}✓ Environment is ready for deployment${NC}"
    echo ""
    echo "You can now run:"
    echo "  ./deploy.sh api-only    # For minimal deployment"
    echo "  ./deploy.sh full        # For full stack deployment"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Validation completed with $WARNINGS warning(s)${NC}"
    echo -e "${GREEN}✓ Environment is ready for deployment${NC}"
    echo ""
    echo "Review warnings above, then run:"
    echo "  ./deploy.sh api-only"
else
    echo -e "${RED}✗ Validation failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before deploying."
    exit 1
fi

echo ""
