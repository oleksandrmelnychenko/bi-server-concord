#!/bin/bash
# Setup Daily Model Refresh - Cron Job Installation
#
# This script installs a cron job that runs daily at 2 AM
# to refresh data and retrain ML models.

set -e  # Exit on error

echo "================================================================================"
echo "SETTING UP DAILY MODEL REFRESH"
echo "================================================================================"

# Get the absolute path to this directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"
echo "✓ Created logs directory"

# Make refresh script executable
chmod +x "$PROJECT_DIR/scripts/refresh_models_daily.py"
echo "✓ Made refresh script executable"

# Test run the refresh script (optional - comment out if you want to skip)
echo ""
read -p "Do you want to do a TEST RUN now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running test refresh..."
    python3 "$PROJECT_DIR/scripts/refresh_models_daily.py"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ TEST RUN SUCCESSFUL!"
    else
        echo ""
        echo "❌ TEST RUN FAILED - Please check the logs"
        exit 1
    fi
fi

# Create cron job entry
CRON_JOB="0 2 * * * cd $PROJECT_DIR && python3 scripts/refresh_models_daily.py >> logs/daily_refresh.log 2>&1"

echo ""
echo "Cron job to be installed:"
echo "----------------------------------------"
echo "$CRON_JOB"
echo "----------------------------------------"
echo ""
echo "This will run every day at 2:00 AM"
echo ""

read -p "Install this cron job? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Add to crontab (preserve existing cron jobs)
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

    echo ""
    echo "✅ Cron job installed successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l
    echo ""
else
    echo ""
    echo "Cron job NOT installed."
    echo ""
    echo "To install manually, run:"
    echo "  crontab -e"
    echo ""
    echo "Then add this line:"
    echo "  $CRON_JOB"
    echo ""
fi

echo ""
echo "================================================================================"
echo "SETUP COMPLETE"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. ✅ Models will refresh automatically every day at 2 AM"
echo ""
echo "2. Check logs:"
echo "   tail -f $PROJECT_DIR/logs/daily_refresh.log"
echo ""
echo "3. Manual refresh (if needed):"
echo "   python3 $PROJECT_DIR/scripts/refresh_models_daily.py"
echo ""
echo "4. Remove cron job (if needed):"
echo "   crontab -e  # Then delete the line"
echo ""
echo "================================================================================"
