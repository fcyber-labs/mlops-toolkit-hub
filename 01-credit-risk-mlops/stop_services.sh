#!/bin/bash
# Stop all Phase 3 services

echo "Stopping Phase 3 Services"

if [ -f .services.pid ]; then
    PIDS=$(cat .services.pid)
    echo "Stopping PIDs: $PIDS"
    kill $PIDS 2>/dev/null
    rm -f .services.pid
    echo " Services stopped"
else
    echo " No PID file found. Trying to kill by port..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null && echo " Killed API on port 8000"
    lsof -ti:8501 | xargs kill -9 2>/dev/null && echo " Killed Dashboard on port 8501"
fi

echo "All services stopped!"