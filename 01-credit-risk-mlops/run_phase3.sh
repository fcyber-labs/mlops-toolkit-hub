#!/bin/bash
# Phase 3: Streaming + Real-World MLOps with Interactive Dashboard

# RUN: ./run_phase3.sh

set -e


echo "Phase 3: Streaming + Real-World MLOps"
echo "Interactive Dashboard with Drift Filters"


# Generate synthetic data for BOTH models
echo ""
echo "[1/4] Generating synthetic streaming data..."

echo "  → For Phase 1 (German Credit)..."
python src/streaming/data_generator_german.py

echo "  → For Phase 2 (LendingClub)..."
python src/streaming/data_generator.py

echo " Data generation complete!"

# Run drift detection for both models
echo ""
echo "[2/4] Running drift detection for both models..."
python src/streaming/drift_detector.py
echo " Drift detection complete!"

# Kill existing processes on ports 8000 and 8501
echo ""
echo "[3/4] Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true
echo " Cleanup complete!"

# Start API server in background
echo ""
echo "[4/4] Starting services..."
echo " Starting API server on port 8000..."
cd "$(pwd)"
python -c "
import sys
sys.path.insert(0, '.')
from src.api.app import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)
" > logs/api.log 2>&1 &
API_PID=$!
echo "   API PID: $API_PID"

# Wait a moment for API to start
sleep 3

# Start Streamlit dashboard in background
echo " Starting Streamlit dashboard on port 8501..."
streamlit run src/monitoring/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.serverAddress=localhost \
    > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "   Dashboard PID: $DASHBOARD_PID"

# Create logs directory if not exists
mkdir -p logs

echo ""

echo " ALL SERVICES STARTED SUCCESSFULLY!"

echo ""
echo " Access your services:"
echo "   API:        http://localhost:8000"
echo "   API Docs:   http://localhost:8000/docs"
echo "   Dashboard:  http://localhost:8501"
echo ""
echo " Log files:"
echo "   API:        logs/api.log"
echo "   Dashboard:  logs/dashboard.log"
echo ""
echo " To stop all services, run:"
echo "   kill $API_PID $DASHBOARD_PID"
echo "   or run: ./stop_services.sh"
echo ""


# Save PIDs to file for later cleanup
echo "$API_PID $DASHBOARD_PID" > .services.pid

# Open browser only once
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ ! -f .browser_opened ]; then
        echo ""
        echo " Opening browser..."
        sleep 2
        open "http://localhost:8501"
        touch .browser_opened
    fi
fi

# Keep script running and show logs
echo ""
echo " Streaming logs (Ctrl+C to stop services)..."

echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo " Stopping services..."
    kill $API_PID $DASHBOARD_PID 2>/dev/null
    rm -f .services.pid .browser_opened
    echo " Services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Tail logs
tail -f logs/api.log logs/dashboard.log