#!/bin/bash
################################################################################
# Concord BI Server - Complete Installation Script
#
# This script:
# - Installs all dependencies
# - Sets up Redis
# - Configures database connection
# - Installs systemd services (Linux) or launchd (macOS)
# - Sets up automated scheduling for recommendations and forecasting
# - Runs initial data generation
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_info "Detected OS: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "Detected OS: macOS"
    else
        log_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
}

# Check if running as root (needed for systemd on Linux)
check_permissions() {
    if [[ "$OS" == "linux" ]] && [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root on Linux (use sudo)"
        exit 1
    fi
}

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_info "Project directory: $PROJECT_DIR"

# Step 1: Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Found Python $PYTHON_VERSION"

    # Check if venv exists, create if not
    if [ ! -d "$PROJECT_DIR/venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$PROJECT_DIR/venv"
    fi

    # Activate venv and install dependencies
    source "$PROJECT_DIR/venv/bin/activate"

    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip install --upgrade pip
        pip install -r "$PROJECT_DIR/requirements.txt"
    else
        log_warning "requirements.txt not found, installing core dependencies..."
        pip install --upgrade pip
        pip install fastapi uvicorn redis numpy scipy pyodbc python-dotenv
    fi

    log_success "Python dependencies installed"
}

# Step 2: Install and configure Redis
install_redis() {
    log_info "Checking Redis installation..."

    if command -v redis-server &> /dev/null; then
        log_success "Redis already installed"
    else
        log_info "Installing Redis..."

        if [[ "$OS" == "macos" ]]; then
            if command -v brew &> /dev/null; then
                brew install redis
            else
                log_error "Homebrew not found. Please install Homebrew first."
                exit 1
            fi
        elif [[ "$OS" == "linux" ]]; then
            apt-get update
            apt-get install -y redis-server
        fi

        log_success "Redis installed"
    fi

    # Start Redis
    if [[ "$OS" == "macos" ]]; then
        brew services start redis || log_warning "Redis may already be running"
    elif [[ "$OS" == "linux" ]]; then
        systemctl start redis-server || log_warning "Redis may already be running"
        systemctl enable redis-server
    fi

    # Test Redis connection
    if redis-cli ping &> /dev/null; then
        log_success "Redis is running"
    else
        log_error "Redis is not responding"
        exit 1
    fi
}

# Step 3: Configure environment variables
configure_environment() {
    log_info "Configuring environment variables..."

    ENV_FILE="$PROJECT_DIR/.env"

    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating .env file..."

        cat > "$ENV_FILE" << 'EOF'
# Database Configuration
DB_HOST=your_db_host
DB_PORT=1433
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

# Forecasting Configuration
FORECAST_CACHE_TTL=604800
FORECAST_WORKERS=10
FORECAST_MIN_CUSTOMERS=2
FORECAST_MIN_ORDERS=3

# API Configuration
API_PORT=8000
API_WORKERS=4
EOF

        log_warning "Please edit .env file with your database credentials"
        log_warning "Location: $ENV_FILE"
    else
        log_success ".env file already exists"
    fi
}

# Step 4: Set up systemd services (Linux)
setup_systemd_services() {
    log_info "Setting up systemd services..."

    # API Service
    cat > /etc/systemd/system/concord-api.service << EOF
[Unit]
Description=Concord BI API Server
After=network.target redis.service

[Service]
Type=simple
User=$SUDO_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Forecast Worker Service
    cat > /etc/systemd/system/forecast-worker.service << EOF
[Unit]
Description=Concord BI Forecast Worker
After=network.target redis.service

[Service]
Type=oneshot
User=$SUDO_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python3 scripts/forecasting/forecast_worker.py
StandardOutput=append:/var/log/forecast_worker.log
StandardError=append:/var/log/forecast_worker_error.log

[Install]
WantedBy=multi-user.target
EOF

    # Forecast Worker Timer (runs every Sunday at 3 AM)
    cat > /etc/systemd/system/forecast-worker.timer << EOF
[Unit]
Description=Run Forecast Worker Weekly
Requires=forecast-worker.service

[Timer]
OnCalendar=Sun *-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Recommendation Worker Service
    cat > /etc/systemd/system/recommendation-worker.service << EOF
[Unit]
Description=Concord BI Weekly Recommendation Worker
After=network.target redis.service

[Service]
Type=oneshot
User=$SUDO_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python3 scripts/weekly_recommendation_worker.py
StandardOutput=append:/var/log/recommendation_worker.log
StandardError=append:/var/log/recommendation_worker_error.log

[Install]
WantedBy=multi-user.target
EOF

    # Recommendation Worker Timer (runs every Sunday at 2 AM)
    cat > /etc/systemd/system/recommendation-worker.timer << EOF
[Unit]
Description=Run Recommendation Worker Weekly
Requires=recommendation-worker.service

[Timer]
OnCalendar=Sun *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Reload systemd
    systemctl daemon-reload

    # Enable services
    systemctl enable concord-api.service
    systemctl enable forecast-worker.timer
    systemctl enable recommendation-worker.timer

    log_success "Systemd services configured"
}

# Step 5: Set up launchd services (macOS)
setup_launchd_services() {
    log_info "Setting up launchd services..."

    LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$LAUNCH_AGENTS_DIR"

    # API Service
    cat > "$LAUNCH_AGENTS_DIR/com.concord.bi.api.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.concord.bi.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/venv/bin/uvicorn</string>
        <string>api.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
        <string>--workers</string>
        <string>4</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/api.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/api_error.log</string>
</dict>
</plist>
EOF

    # Forecast Worker (runs every Sunday at 3 AM)
    cat > "$LAUNCH_AGENTS_DIR/com.concord.bi.forecast.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.concord.bi.forecast</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/venv/bin/python3</string>
        <string>scripts/forecasting/forecast_worker.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>3</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/forecast_worker.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/forecast_worker_error.log</string>
</dict>
</plist>
EOF

    # Recommendation Worker (runs every Sunday at 2 AM)
    cat > "$LAUNCH_AGENTS_DIR/com.concord.bi.recommendations.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.concord.bi.recommendations</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/venv/bin/python3</string>
        <string>scripts/weekly_recommendation_worker.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/recommendation_worker.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/recommendation_worker_error.log</string>
</dict>
</plist>
EOF

    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs"

    # Load services
    launchctl load "$LAUNCH_AGENTS_DIR/com.concord.bi.api.plist"
    launchctl load "$LAUNCH_AGENTS_DIR/com.concord.bi.forecast.plist"
    launchctl load "$LAUNCH_AGENTS_DIR/com.concord.bi.recommendations.plist"

    log_success "Launchd services configured"
}

# Step 6: Run initial data generation
run_initial_generation() {
    log_info "Running initial data generation..."

    source "$PROJECT_DIR/venv/bin/activate"

    log_info "Generating weekly recommendations (this may take 10-15 minutes)..."
    python3 "$PROJECT_DIR/scripts/weekly_recommendation_worker.py" || log_warning "Recommendation generation failed (may need database credentials)"

    log_info "Generating forecasts (this may take 15-20 minutes)..."
    python3 "$PROJECT_DIR/scripts/forecasting/forecast_worker.py" || log_warning "Forecast generation failed (may need database credentials)"

    log_success "Initial data generation complete"
}

# Step 7: Start services
start_services() {
    log_info "Starting services..."

    if [[ "$OS" == "linux" ]]; then
        systemctl start concord-api.service
        systemctl start forecast-worker.timer
        systemctl start recommendation-worker.timer

        # Check status
        if systemctl is-active --quiet concord-api.service; then
            log_success "API service started"
        else
            log_error "API service failed to start"
        fi
    elif [[ "$OS" == "macos" ]]; then
        # Services already loaded via launchctl
        log_success "Services started via launchd"
    fi
}

# Step 8: Display summary
display_summary() {
    echo ""
    echo "================================================================================"
    log_success "Installation Complete!"
    echo "================================================================================"
    echo ""
    echo "📍 Project Directory: $PROJECT_DIR"
    echo "🐍 Python Virtual Environment: $PROJECT_DIR/venv"
    echo "🔧 Configuration File: $PROJECT_DIR/.env"
    echo ""
    echo "🚀 Services:"
    echo "   - API Server: http://localhost:8000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo "   - Health Check: http://localhost:8000/health"
    echo ""
    echo "⏰ Scheduled Tasks:"
    echo "   - Recommendations: Every Sunday at 2:00 AM"
    echo "   - Forecasts: Every Sunday at 3:00 AM"
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Edit .env file with your database credentials:"
    echo "      nano $PROJECT_DIR/.env"
    echo ""
    echo "   2. Test the API:"
    echo "      curl http://localhost:8000/health"
    echo ""
    echo "   3. Generate initial data (if not done automatically):"
    echo "      source $PROJECT_DIR/venv/bin/activate"
    echo "      python3 scripts/weekly_recommendation_worker.py"
    echo "      python3 scripts/forecasting/forecast_worker.py"
    echo ""
    if [[ "$OS" == "linux" ]]; then
        echo "   4. Manage services:"
        echo "      sudo systemctl status concord-api"
        echo "      sudo systemctl restart concord-api"
        echo "      sudo journalctl -u forecast-worker -f"
    elif [[ "$OS" == "macos" ]]; then
        echo "   4. Manage services:"
        echo "      launchctl list | grep concord"
        echo "      tail -f $PROJECT_DIR/logs/api.log"
    fi
    echo ""
    echo "================================================================================"
}

# Main installation flow
main() {
    echo "================================================================================"
    echo "                  Concord BI Server - Installation Script"
    echo "================================================================================"
    echo ""

    detect_os

    if [[ "$OS" == "linux" ]]; then
        check_permissions
    fi

    log_info "Starting installation..."
    echo ""

    install_dependencies
    install_redis
    configure_environment

    if [[ "$OS" == "linux" ]]; then
        setup_systemd_services
    elif [[ "$OS" == "macos" ]]; then
        setup_launchd_services
    fi

    # Ask if user wants to run initial generation now
    echo ""
    read -p "Run initial data generation now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_initial_generation
    else
        log_info "Skipping initial data generation. You can run it manually later."
    fi

    start_services

    display_summary
}

# Run main function
main
