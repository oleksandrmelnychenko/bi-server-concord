# Recommendation API Deployment Script for Windows
# =================================================
#
# Usage:
#   .\deploy.ps1              # Full deployment
#   .\deploy.ps1 -Rebuild     # Force rebuild containers
#   .\deploy.ps1 -Stop        # Stop all services
#   .\deploy.ps1 -Status      # Check service status
#

param(
    [switch]$Rebuild,
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName
$DataDir = "$ProjectRoot\data"
$EnvFile = "$ProjectRoot\.env"
$ComposeFile = "$ScriptDir\docker-compose.yml"

Write-Host "============================================" -ForegroundColor Blue
Write-Host "   Recommendation API Deployment (Windows)" -ForegroundColor Blue
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
        Write-Host "  Download from: https://docs.docker.com/desktop/install/windows-install/"
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

function Test-EnvFile {
    Write-Info "Checking environment configuration..."

    if (-not (Test-Path $EnvFile)) {
        Write-Error ".env file not found at $EnvFile"

        $envExample = "$ProjectRoot\.env.example"
        if (Test-Path $envExample) {
            Write-Info "Creating .env from template..."
            Copy-Item $envExample $EnvFile
            Write-Warn "Please edit $EnvFile with your database credentials"
            exit 1
        }
        else {
            Write-Error "No .env.example found. Please create .env manually."
            exit 1
        }
    }
    Write-Info "Environment file exists"
}

function Initialize-DataDirectory {
    Write-Info "Setting up data directory..."

    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    }
    Write-Info "Data directory ready: $DataDir"
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
    Write-Info "  - redis: Cache store (port 6380)"
    Write-Info "  - api: FastAPI recommendation service (port 8100)"
    Write-Info "  - worker: Scheduled jobs (daily ANN learning + weekly batch)"
}

function Start-Services {
    Write-Info "Starting services..."
    Set-Location $ScriptDir

    # Start all core services
    docker compose -f $ComposeFile up -d
    Write-Info "Services starting..."
    Write-Info "  - redis: Starting cache store"
    Write-Info "  - api: Starting recommendation API"
    Write-Info "  - worker: Starting scheduled worker"

    Write-Info "Waiting for services to be healthy..."
    Start-Sleep -Seconds 10

    # Health check
    for ($i = 1; $i -le 30; $i++) {
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:8100/health" -TimeoutSec 5
            if ($health.status -eq "healthy") {
                Write-Info "API is healthy"
                return
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }

    Write-Warn "API health check timed out. Check logs with: docker logs recommendation-api"
}

function Stop-Services {
    Write-Info "Stopping services..."
    Set-Location $ScriptDir
    docker compose -f $ComposeFile down
    Write-Info "Services stopped"
}

function Show-Status {
    Write-Host ""
    Write-Host "=== Service Status ===" -ForegroundColor Blue
    Set-Location $ScriptDir
    docker compose -f $ComposeFile ps

    Write-Host ""
    Write-Host "=== API Health ===" -ForegroundColor Blue
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8100/health" -TimeoutSec 5
        $health | ConvertTo-Json
    }
    catch {
        Write-Warn "API is not responding"
    }

    Write-Host ""
    Write-Host "=== ANN Index ===" -ForegroundColor Blue
    $annFile = "$DataDir\ann_neighbors.json"
    if (Test-Path $annFile) {
        $size = (Get-Item $annFile).Length / 1MB
        Write-Info "ANN file exists: $([math]::Round($size, 2)) MB"
    }
    else {
        Write-Warn "ANN file not found"
    }

    Write-Host ""
    Write-Host "=== Scheduled Jobs ===" -ForegroundColor Blue
    Write-Host "  Daily (2 AM):   ANN index regeneration"
    Write-Host "  Weekly (Sun):   Full recommendation batch"
}

function Show-Endpoints {
    Write-Host ""
    Write-Host "=== API Endpoints ===" -ForegroundColor Blue
    Write-Host "  Health Check:     http://localhost:8100/health"
    Write-Host "  Recommendations:  http://localhost:8100/recommendations/{customer_id}"
    Write-Host "  API Docs:         http://localhost:8100/docs"
    Write-Host "  Redis:            localhost:6380"
    Write-Host ""
    Write-Host "=== Worker Schedules ===" -ForegroundColor Blue
    Write-Host "  Daily (2:00 AM):    ANN index regeneration - learns from new purchase data"
    Write-Host "  Weekly (Sun 3 AM):  Batch recommendations - pre-computes for all customers"
    Write-Host ""
    Write-Host "=== Services ===" -ForegroundColor Blue
    Write-Host "  redis:  Cache store for recommendations"
    Write-Host "  api:    FastAPI serving recommendations"
    Write-Host "  worker: Background worker with scheduled jobs"
    Write-Host ""
}

# Main execution
if ($Stop) {
    Stop-Services
}
elseif ($Status) {
    Show-Status
}
else {
    Test-Docker
    Test-DockerCompose
    Test-EnvFile
    Initialize-DataDirectory
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
