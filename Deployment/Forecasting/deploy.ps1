# Forecasting API Deployment Script for Windows
# ==============================================
#
# Usage:
#   .\deploy.ps1              # Full deployment
#   .\deploy.ps1 -Rebuild     # Force rebuild containers
#   .\deploy.ps1 -Stop        # Stop all services
#   .\deploy.ps1 -Status      # Check service status
#   .\deploy.ps1 -RunWorker   # Run forecast worker manually
#

param(
    [switch]$Rebuild,
    [switch]$Stop,
    [switch]$Status,
    [switch]$RunWorker
)

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName
$EnvFile = "$ProjectRoot\.env"
$ComposeFile = "$ScriptDir\docker-compose.yml"

Write-Host "============================================" -ForegroundColor Blue
Write-Host "   Forecasting API Deployment (Windows)" -ForegroundColor Blue
Write-Host "============================================" -ForegroundColor Blue
Write-Host ""

function Write-Info($message) {
    Write-Host "[INFO] $message" -ForegroundColor Green
}

function Write-Warn($message) {
    Write-Host "[WARN] $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "[ERROR] $message" -ForegroundColor Red
}

function Test-Docker {
    Write-Info "Checking Docker installation..."

    try {
        $dockerVersion = docker --version
        Write-Info "Docker installed: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed or not in PATH."
        exit 1
    }

    try {
        docker info | Out-Null
        Write-Info "Docker daemon is running"
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }
}

function Test-DockerCompose {
    Write-Info "Checking Docker Compose..."

    try {
        docker compose version | Out-Null
        Write-Info "Docker Compose is available"
    }
    catch {
        Write-Error "Docker Compose is not available."
        exit 1
    }
}

function Set-DockerIgnore {
    Write-Info "Setting up .dockerignore..."

    $dockerignore = @"
db-ai-api/
db-rag-system/
*.log
__pycache__/
.git/
*.pyc
.DS_Store
.env.example
docs/
sample_forecasts/
Deployment/
"@

    $dockerignore | Out-File -FilePath "$ProjectRoot\.dockerignore" -Encoding UTF8
    Write-Info ".dockerignore configured"
}

function Build-Containers {
    param([switch]$ForceRebuild)

    Write-Info "Building Docker containers..."
    Set-Location $ScriptDir

    if ($ForceRebuild) {
        Write-Info "Force rebuilding containers (--no-cache)..."
        docker compose -f $ComposeFile build --no-cache
    }
    else {
        docker compose -f $ComposeFile build
    }

    Write-Info "Containers built successfully"
    Write-Info "  - redis: Cache store (port 6381)"
    Write-Info "  - api: FastAPI forecasting service (port 8101)"
    Write-Info "  - forecast-worker: Batch forecast generator"
}

function Start-Services {
    Write-Info "Starting services..."
    Set-Location $ScriptDir

    # Start redis and api (not forecast-worker which runs on-demand)
    docker compose -f $ComposeFile up -d redis api
    Write-Info "Services starting..."
    Write-Info "  - redis: Starting cache store"
    Write-Info "  - api: Starting forecasting API"

    Write-Info "Waiting for services to be healthy..."
    Start-Sleep -Seconds 10

    # Health check
    for ($i = 1; $i -le 30; $i++) {
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:8101/health" -TimeoutSec 5
            if ($health.status -eq "healthy") {
                Write-Info "API is healthy"
                return
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }

    Write-Warn "API health check timed out. Check logs with: docker logs forecast-api"
}

function Stop-Services {
    Write-Info "Stopping services..."
    Set-Location $ScriptDir
    docker compose -f $ComposeFile down
    Write-Info "Services stopped"
}

function Start-ForecastWorker {
    Write-Info "Running forecast worker..."
    Set-Location $ScriptDir

    # Ensure redis is running
    docker compose -f $ComposeFile up -d redis
    Start-Sleep -Seconds 5

    Write-Info "Starting forecast worker (this may take 20-40 minutes for all products)..."
    docker compose -f $ComposeFile run --rm forecast-worker

    Write-Info "Forecast worker completed"
}

function Show-Status {
    Write-Host ""
    Write-Host "=== Service Status ===" -ForegroundColor Blue
    Set-Location $ScriptDir
    docker compose -f $ComposeFile ps

    Write-Host ""
    Write-Host "=== API Health ===" -ForegroundColor Blue
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8101/health" -TimeoutSec 5
        $health | ConvertTo-Json
    }
    catch {
        Write-Warn "API is not responding"
    }

    Write-Host ""
    Write-Host "=== Cached Forecasts ===" -ForegroundColor Blue
    try {
        $keys = docker exec forecast-redis redis-cli KEYS "forecast:product:*" 2>$null
        $count = if ($keys) { ($keys | Measure-Object -Line).Lines } else { 0 }
        Write-Info "Cached forecasts in Redis: $count"
    }
    catch {
        Write-Warn "Could not check cache stats"
    }
}

function Show-Endpoints {
    Write-Host ""
    Write-Host "=== API Endpoints ===" -ForegroundColor Blue
    Write-Host "  Health Check:     http://localhost:8101/health"
    Write-Host "  Forecast:         http://localhost:8101/forecast/{product_id}"
    Write-Host "  API Docs:         http://localhost:8101/docs"
    Write-Host "  Redis:            localhost:6381"
    Write-Host ""
    Write-Host "=== Services ===" -ForegroundColor Blue
    Write-Host "  redis:           Cache store for forecasts (7-day TTL)"
    Write-Host "  api:             FastAPI serving product forecasts"
    Write-Host "  forecast-worker: Batch worker (runs on-demand)"
    Write-Host ""
    Write-Host "=== Run Forecast Worker ===" -ForegroundColor Blue
    Write-Host "  .\deploy.ps1 -RunWorker    # Generate forecasts for all products"
    Write-Host ""
}

# Main execution
if ($Stop) {
    Stop-Services
}
elseif ($Status) {
    Show-Status
}
elseif ($RunWorker) {
    Start-ForecastWorker
}
else {
    Test-Docker
    Test-DockerCompose
    Set-DockerIgnore

    if ($Rebuild) {
        Build-Containers -ForceRebuild
    }
    else {
        Build-Containers
    }

    Start-Services
    Show-Status
    Show-Endpoints
}

Write-Host "Deployment complete!" -ForegroundColor Green
