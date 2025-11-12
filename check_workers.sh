#!/bin/bash
echo "==================== WORKER STATUS ===================="
echo ""
echo "Forecast Worker:"
tail -3 /tmp/forecast_worker.log 2>/dev/null || echo "  No logs yet"
echo ""
echo "Recommendation Worker:"
tail -3 /tmp/recommendation_worker.log 2>/dev/null || echo "  No logs yet"
echo ""
echo "Running processes:"
ps aux | grep -E "(forecast_worker|weekly_recommendation_worker)" | grep -v grep | awk '{print "  PID " $2 ": " $11}'
echo ""
echo "======================================================="
