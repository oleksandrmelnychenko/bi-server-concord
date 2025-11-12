#!/bin/bash
#
# Stop Workers Script
# Stops forecast worker and recommendation worker
#

echo "======================================================"
echo "Stopping Workers"
echo "======================================================"
echo ""

# Find running workers
FORECAST_PID=$(ps aux | grep "[f]orecast_worker.py" | awk '{print $2}')
RECOMMENDATION_PID=$(ps aux | grep "[w]eekly_recommendation_worker.py" | awk '{print $2}')

if [ -z "$FORECAST_PID" ] && [ -z "$RECOMMENDATION_PID" ]; then
    echo "No workers are currently running."
    echo ""
    exit 0
fi

# Stop forecast worker
if [ ! -z "$FORECAST_PID" ]; then
    echo "Stopping forecast worker (PID: $FORECAST_PID)..."
    kill $FORECAST_PID 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  Forecast worker stopped successfully"
    else
        echo "  Failed to stop forecast worker (may already be stopped)"
    fi
else
    echo "Forecast worker is not running"
fi

# Stop recommendation worker
if [ ! -z "$RECOMMENDATION_PID" ]; then
    echo "Stopping recommendation worker (PID: $RECOMMENDATION_PID)..."
    kill $RECOMMENDATION_PID 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  Recommendation worker stopped successfully"
    else
        echo "  Failed to stop recommendation worker (may already be stopped)"
    fi
else
    echo "Recommendation worker is not running"
fi

echo ""
echo "Waiting 2 seconds for workers to shutdown..."
sleep 2

echo ""
echo "======================================================"
echo "Final Status"
echo "======================================================"

# Verify workers are stopped
FORECAST_STILL_RUNNING=$(ps aux | grep "[f]orecast_worker.py" | awk '{print $2}')
RECOMMENDATION_STILL_RUNNING=$(ps aux | grep "[w]eekly_recommendation_worker.py" | awk '{print $2}')

if [ -z "$FORECAST_STILL_RUNNING" ]; then
    echo "  Forecast worker:       STOPPED"
else
    echo "  Forecast worker:       STILL RUNNING (PID: $FORECAST_STILL_RUNNING)"
    echo "    Force kill: kill -9 $FORECAST_STILL_RUNNING"
fi

if [ -z "$RECOMMENDATION_STILL_RUNNING" ]; then
    echo "  Recommendation worker: STOPPED"
else
    echo "  Recommendation worker: STILL RUNNING (PID: $RECOMMENDATION_STILL_RUNNING)"
    echo "    Force kill: kill -9 $RECOMMENDATION_STILL_RUNNING"
fi

echo ""
echo "======================================================"
