#!/bin/bash

# RUN ./run_full_lending_pipeline.sh

# LightGBM Best Model (Phase 2)


set -e   # exit immediately on any error

echo "=========================================="
echo "LENDING CLUB PIPELINE — LIGHTGBM + OPTUNA"
echo "=========================================="

#  Environment 
export PYTHONPATH="${PYTHONPATH}:."

#  Config 
echo ""
echo "📋 Copying params_lending.yaml → params.yaml"
cp params_lending.yaml params.yaml
echo "✅ Params ready"

#  Step 1: Preprocess 
echo ""
echo "[1/4] Running Preprocessing..."
python src/preprocess.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi
echo "✅ Preprocessing complete"

#  Step 2: Train 
echo ""
echo "[2/4] Training LightGBM (Optuna HPO + business-optimal threshold)..."
python src/train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo "✅ Training complete"

#  Step 3: Evaluate 
echo ""
echo "[3/4] Evaluating Model (KS, Gini, Lift, Business P&L)..."
python src/evaluate.py
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi
echo "✅ Evaluation complete"

#  Step 4: Explain 
echo ""
echo "[4/4] Running SHAP Explainability..."
python src/explain.py
if [ $? -ne 0 ]; then
    echo "❌ Explainability failed!"
    exit 1
fi
echo "✅ Explainability complete"

#  Summary 
echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Artefacts:"
echo "  Model       : models/model_lending.pkl"
echo "  Booster     : models/model_lending_booster.txt"
echo "  Metadata    : models/model_metadata.json"
echo ""
echo "  Threshold   : reports/threshold.json"
echo "  Metrics     : reports/metrics_full.json"
echo "  Dashboard   : reports/results_dashboard.png"
echo "  Feat. Imp.  : reports/feature_importance.png"
echo ""
echo "  SHAP (combined)  : reports/shap_analysis.png"
echo "  SHAP (beeswarm)  : reports/shap_summary.png"
echo "  SHAP (bar)       : reports/shap_bar.png"
echo "  SHAP (CSV)       : reports/shap_feature_importance.csv"
echo ""
echo "MLflow UI: mlflow ui --backend-store-uri \$MLFLOW_TRACKING_URI"
echo ""