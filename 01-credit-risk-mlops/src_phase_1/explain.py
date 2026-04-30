import os

# For macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


import pandas as pd
import shap
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import json
import numpy as np
import joblib
import lightgbm as lgb
from dotenv import load_dotenv
import sys
sys.path.append('.')
from config.logging_config import logger

load_dotenv()

#  MLflow tracking 
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
experiment_name = "phase_1_credit_risk"
mlflow.set_experiment(experiment_name)

logger.info("*"*70)
logger.info("STARTING MODEL EXPLAINABILITY")
logger.info("*"*70)

os.makedirs("reports", exist_ok=True)

#   test data 
logger.info("Loading test data...")
X_test = pd.read_csv("data/processed/X_test.csv")
logger.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")

#  Align to selected features 
selected_features_path = "data/processed/selected_features.pkl"
if os.path.exists(selected_features_path):
    selected_features = joblib.load(selected_features_path)
    missing = [f for f in selected_features if f not in X_test.columns]
    if missing:
        logger.error(f"X_test missing {len(missing)} expected features: {missing[:5]}")
        sys.exit(1)
    X_test = X_test[selected_features]
    logger.info(f"Feature alignment applied: {X_test.shape[1]} features (from selected_features.pkl)")
else:
    logger.warning("selected_features.pkl not found — using X_test.csv columns as-is")

sample_size = min(100, X_test.shape[0])
X_test_sample = X_test.sample(n=sample_size, random_state=42)
logger.info(f"Using sample of {sample_size} samples for SHAP analysis")

#  pipeline config 

logger.info("Loading pipeline config and models...")

import yaml
pipeline_config_path = yaml.safe_load(open("params.yaml"))["train"]["model"]
if not os.path.exists(pipeline_config_path):
    logger.error(f"Pipeline config not found: {pipeline_config_path}")
    sys.exit(1)

pipeline_config = joblib.load(pipeline_config_path)
model_names     = pipeline_config["model_names"]
logger.info(f"Pipeline models: {model_names}")

#   the best LightGBM model for SHAP 

lgb_candidates = [n for n in model_names if "lgb" in n.lower()]
shap_model_name = lgb_candidates[0] if lgb_candidates else model_names[0]
shap_model_pkl  = f"models/{shap_model_name}.pkl"

if not os.path.exists(shap_model_pkl):
    logger.error(f"Model file not found: {shap_model_pkl}")
    sys.exit(1)

loaded = joblib.load(shap_model_pkl)
logger.info(f"Loaded {shap_model_name} from {shap_model_pkl} (type: {type(loaded).__name__})")

#  Extract raw LightGBM Booster for SHAP 

if hasattr(loaded, "booster_"):
    best_model = loaded.booster_
    logger.success(f"Extracted Booster from sklearn LGBMClassifier ({shap_model_name})")
elif isinstance(loaded, lgb.Booster):
    best_model = loaded
    logger.success(f"Raw LightGBM Booster loaded ({shap_model_name})")
elif hasattr(loaded, "predict"):

    best_model = loaded
    logger.warning(f"Non-LightGBM model loaded ({shap_model_name}): SHAP may be slower")
else:
    logger.error(f"Unrecognised model type for {shap_model_name}: {type(loaded)}")
    sys.exit(1)

#  Validate feature count before SHAP 
if isinstance(best_model, lgb.Booster):
    model_n_features = best_model.num_feature()
    if model_n_features != X_test.shape[1]:
        logger.error(
            f"Feature mismatch: model expects {model_n_features} features, "
            f"X_test has {X_test.shape[1]}. "
            f"Run 'dvc repro --force preprocess train' to realign artifacts."
        )
        sys.exit(1)
    logger.info(f"Feature count validated: {model_n_features} features ✓")

#  SHAP analysis 
logger.info("Computing SHAP values...")
logger.info("This may take a few minutes...")

try:

    explainer = shap.TreeExplainer(
        best_model,
        feature_perturbation="tree_path_dependent",
    )
    raw = explainer.shap_values(X_test_sample)


    if isinstance(raw, list):
        logger.info(f"Binary classifier detected – using class-1 SHAP values (list length {len(raw)})")
        shap_values = raw[1]
    else:
        shap_values = raw

    logger.success("SHAP values computed successfully")
except Exception as e:
    logger.error(f"SHAP computation failed: {e}")
    sys.exit(1)

#  Plots 
logger.info("Creating SHAP summary plot...")
try:
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", bbox_inches='tight', dpi=150)
    plt.close()
    logger.success("SHAP summary plot saved to reports/shap_summary.png")
except Exception as e:
    logger.warning(f"Could not create summary plot: {e}")

logger.info("Creating SHAP bar plot...")
try:
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_bar.png", bbox_inches='tight', dpi=150)
    plt.close()
    logger.success("SHAP bar plot saved to reports/shap_bar.png")
except Exception as e:
    logger.warning(f"Could not create bar plot: {e}")

logger.info("Creating SHAP waterfall plot (sample 0)...")
try:
    expected_value = explainer.expected_value
    # Binary classifier
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1])

    explanation = shap.Explanation(
        values=np.array(shap_values)[0],
        base_values=expected_value,
        data=X_test_sample.iloc[0].values,
        feature_names=list(X_test_sample.columns),
    )
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall_sample.png", bbox_inches='tight', dpi=150)
    plt.close()
    logger.success("SHAP waterfall plot saved to reports/shap_waterfall_sample.png")
except Exception as e:
    logger.warning(f"Could not create waterfall plot: {e}")

logger.info("Creating SHAP force plot (sample 0)...")
try:
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1])

    shap.initjs()
    force = shap.force_plot(
        expected_value,
        np.array(shap_values)[0],
        X_test_sample.iloc[0],
        matplotlib=True,
        show=False,
    )
    plt.savefig("reports/shap_force_plot.png", bbox_inches='tight', dpi=150)
    plt.close()
    logger.success("SHAP force plot saved to reports/shap_force_plot.png")
except Exception as e:
    logger.warning(f"Could not create force plot: {e}")

#  Feature importance 

arr = np.array(shap_values)
if arr.ndim == 2:
    mean_abs_shap = np.abs(arr).mean(axis=0)
elif arr.ndim == 3:     
    mean_abs_shap = np.abs(arr).mean(axis=(0, 1))
else:
    mean_abs_shap = np.abs(arr).flatten()

feature_importance_shap = pd.DataFrame({
    'feature': X_test_sample.columns,
    'shap_importance': mean_abs_shap
}).sort_values('shap_importance', ascending=False)

feature_importance_shap.to_csv("reports/shap_feature_importance.csv", index=False)
logger.success("SHAP feature importance saved to reports/shap_feature_importance.csv")

#  MLflow logging 

main_run_id = None
if os.path.exists("data/processed/main_run_id.txt"):
    with open("data/processed/main_run_id.txt", "r") as f:
        main_run_id = f.read().strip()

logger.info("Logging SHAP artifacts to MLflow...")
try:
    if main_run_id:

        with mlflow.start_run(run_id=main_run_id, nested=False):
            mlflow.log_artifact("reports/shap_summary.png")
            mlflow.log_artifact("reports/shap_bar.png")
            mlflow.log_artifact("reports/shap_waterfall_sample.png")
            mlflow.log_artifact("reports/shap_force_plot.png")
            mlflow.log_artifact("reports/shap_feature_importance.csv")
            logger.success(f"SHAP artifacts logged to main run: {main_run_id}")
    else:
        # Fallback
        with mlflow.start_run(run_name="explainability", nested=False):
            mlflow.log_artifact("reports/shap_summary.png")
            mlflow.log_artifact("reports/shap_bar.png")
            mlflow.log_artifact("reports/shap_waterfall_sample.png")
            mlflow.log_artifact("reports/shap_force_plot.png")
            mlflow.log_artifact("reports/shap_feature_importance.csv")
            logger.success("SHAP artifacts logged to MLflow (separate run)")
except Exception as e:
    logger.warning(f"Could not log to MLflow: {e}")

#  Summary 
logger.info("*"*70)
logger.info("EXPLAINABILITY SUMMARY")
logger.info("*"*70)
logger.info(f"Samples analyzed: {sample_size}")
logger.info(f"Features: {X_test_sample.shape[1]}")
logger.info("Top 5 most important features:")
for i, row in feature_importance_shap.head(5).iterrows():
    logger.info(f"  {i+1}. {row['feature']}: {row['shap_importance']:.4f}")

logger.success("Model explainability completed successfully!")