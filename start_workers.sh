#!/bin/bash
#
# Start Both Workers Script
# Starts forecast worker and recommendation worker in background
#

echo "======================================================"
echo "Starting Workers"
echo "======================================================"
echo ""

# Kill existing workers first
echo "Checking for existing workers..."
FORECAST_PID=$(ps aux | grep "[f]orecast_worker.py" | awk '{print $2}')
RECOMMENDATION_PID=$(ps aux | grep "[w]eekly_recommendation_worker.py" | awk '{print $2}')

if [ ! -z "$FORECAST_PID" ]; then
    echo "  Stopping existing forecast worker (PID: $FORECAST_PID)..."
    kill $FORECAST_PID 2>/dev/null
    sleep 2
fi

if [ ! -z "$RECOMMENDATION_PID" ]; then
    echo "  Stopping existing recommendation worker (PID: $RECOMMENDATION_PID)..."
    kill $RECOMMENDATION_PID 2>/dev/null
    sleep 2
fi

echo ""
echo "Starting new workers..."

# Start forecast worker
nohup python3 scripts/forecasting/forecast_worker.py > /tmp/forecast_worker.log 2>&1 &
FORECAST_PID=$!
echo "  Forecast worker started (PID: $FORECAST_PID)"
echo "    Log: /tmp/forecast_worker.log"
echo "    Monitor: tail -f /tmp/forecast_worker.log"

# Start recommendation worker
nohup python3 scripts/weekly_recommendation_worker.py > /tmp/recommendation_worker.log 2>&1 &
RECOMMENDATION_PID=$!
echo "  Recommendation worker started (PID: $RECOMMENDATION_PID)"
echo "    Log: /tmp/recommendation_worker.log"
echo "    Monitor: tail -f /tmp/recommendation_worker.log"

echo ""
echo "Waiting 3 seconds for workers to initialize..."
sleep 3

echo ""
echo "======================================================"
echo "Worker Status"
echo "======================================================"

# Check if processes are still running
if ps -p $FORECAST_PID > /dev/null 2>&1; then
    echo "  Forecast worker:       RUNNING (PID: $FORECAST_PID)"
else
    echo "  Forecast worker:       FAILED (check logs)"
fi

if ps -p $RECOMMENDATION_PID > /dev/null 2>&1; then
    echo "  Recommendation worker: RUNNING (PID: $RECOMMENDATION_PID)"
else
    echo "  Recommendation worker: FAILED (check logs)"
fi

echo ""
echo "Recent log output:"
echo ""
echo "Forecast worker:"
tail -5 /tmp/forecast_worker.log 2>/dev/null || echo "  No logs yet"
echo ""
echo "Recommendation worker:"
tail -5 /tmp/recommendation_worker.log 2>/dev/null || echo "  No logs yet"

echo ""
echo "======================================================"
echo "Workers started successfully!"
echo ""
echo "To check status:    ./check_workers.sh"
echo "To stop workers:    ./stop_workers.sh"
echo "======================================================"
