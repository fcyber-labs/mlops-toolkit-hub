#!/bin/bash
# run_all.sh - Complete Pipeline: Phase 1 → Phase 2 → Phase 3

echo "=========================================="
echo "COMPLETE MLOPS PIPELINE"
echo "Phase 1 → Phase 2 → Phase 3"
echo "=========================================="


# PHASE 1: German Credit (DVC)

echo ""
echo "🏦 PHASE 1: German Credit Pipeline"
echo "=========================================="

# Copy German Credit params
cp params_german.yaml params.yaml

# Run DVC pipeline
dvc repro

if [ $? -ne 0 ]; then
    echo "❌ Phase 1 failed!"
    exit 1
fi
echo "✅ Phase 1 complete!"


# PHASE 2: LendingClub Production

echo ""
echo "🏭 PHASE 2: LendingClub Pipeline"
echo "=========================================="

./run_full_lending_pipeline.sh

if [ $? -ne 0 ]; then
    echo "❌ Phase 2 failed!"
    exit 1
fi
echo "✅ Phase 2 complete!"


# PHASE 3: Streaming + Monitoring

echo ""
echo "📡 PHASE 3: Streaming + Monitoring"
echo "=========================================="

# Kill existing services before starting
./stop_services.sh 2>/dev/null

# Run Phase 3
./run_phase3.sh

if [ $? -ne 0 ]; then
    echo "❌ Phase 3 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ALL PHASES COMPLETE!"
echo "=========================================="
echo ""
echo "Services running:"
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Dashboard:  http://localhost:8501"
echo ""
echo "To stop all services: ./stop_services.sh"