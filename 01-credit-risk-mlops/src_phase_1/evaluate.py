import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
sys.path.append(".")
from config.logging_config import logger

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"]      = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef, brier_score_loss,
    average_precision_score, roc_curve, precision_recall_curve,
)

logger.info("*" * 70)
logger.info("MODEL EVALUATION v3")
logger.info("*" * 70)

os.makedirs("reports", exist_ok=True)


import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# only for OpenMP on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"


import lightgbm as lgb


#  test data 
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0].values
logger.info(f"Test data: {X_test.shape[0]} samples × {X_test.shape[1]} features")

#  pipeline config
import yaml
model_path = yaml.safe_load(open("params.yaml"))["train"]["model"]

# Check if pipeline_config exists (ensemble) or fallback to single model
if os.path.exists(model_path):
    try:
        pipeline_config = joblib.load(model_path)
        # Check if it's a pipeline config (has model_names) or a single model
        if isinstance(pipeline_config, dict) and "model_names" in pipeline_config:
            model_names     = pipeline_config["model_names"]
            blend_weights   = np.array(pipeline_config["blend_weights"])
            final_threshold = pipeline_config["threshold"]
            logger.info(f"Loaded pipeline config: {model_names}")
            logger.info(f"Blend weights: {dict(zip(model_names, blend_weights.round(3)))}")
            logger.info(f"Decision threshold: {final_threshold:.3f}")
            
            # Load individual models
            models = {}
            for name in model_names:
                path = f"models/{name}.pkl"
                if os.path.exists(path):
                    models[name] = joblib.load(path)
                    logger.info(f"  Loaded {name}")
                else:
                    logger.error(f"  Model not found: {path}")
                    sys.exit(1)
            
            # Ensemble probabilities
            proba_matrix = np.column_stack([
                models[n].predict_proba(X_test)[:, 1] for n in model_names
            ])
            y_pred_proba = proba_matrix @ blend_weights
            y_pred       = (y_pred_proba >= final_threshold).astype(int)
        else:
            # Single model 
            model = pipeline_config
            final_threshold = 0.5
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= final_threshold).astype(int)
            logger.info("Using single model (not ensemble)")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
else:
    logger.error(f"Model not found: {model_path}")
    sys.exit(1)

#  metrics 
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred)

metrics = {
    "accuracy":          float(accuracy_score(y_test, y_pred)),
    "precision":         float(prec),
    "recall":            float(rec),
    "f1_score":          float(f1_score(y_test, y_pred)),
    "f2_score":          float((5 * prec * rec) / (4 * prec + rec + 1e-9)),
    "roc_auc":           float(roc_auc_score(y_test, y_pred_proba)),
    "avg_precision":     float(average_precision_score(y_test, y_pred_proba)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    "matthews_corr":     float(matthews_corrcoef(y_test, y_pred)),
    "brier_score":       float(brier_score_loss(y_test, y_pred_proba)),
    "threshold_used":    float(final_threshold),
}

logger.info("*" * 70)
logger.info("EVALUATION RESULTS")
logger.info("*" * 70)
for k, v in metrics.items():
    logger.info(f"  {k.upper():20s}: {v:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
logger.info(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

#  individual model comparison 
logger.info("\nPer-model recall @ threshold=0.35:")
if isinstance(pipeline_config, dict) and "model_names" in pipeline_config:
    for name in model_names:
        p = models[name].predict_proba(X_test)[:, 1]
        r = recall_score(y_test, (p >= 0.35).astype(int))
        a = roc_auc_score(y_test, p)
        logger.info(f"  {name:15s}: recall={r:.4f}  AUC={a:.4f}")

#  save reports 
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
logger.success("Metrics saved: reports/metrics.json")

cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=["Predicted Good", "Predicted Bad"],
    index=["Actual Good", "Actual Bad"],
)
with open("reports/confusion_matrix.txt", "w") as f:
    f.write("CONFUSION MATRIX\n" + "=" * 40 + "\n")
    f.write(str(cm_df) + "\n\n")
    f.write(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")
logger.success("Confusion matrix saved: reports/confusion_matrix.txt")

cr = classification_report(y_test, y_pred, target_names=["Good Credit", "Bad Credit"])
with open("reports/classification_report.txt", "w") as f:
    f.write("CLASSIFICATION REPORT\n" + "=" * 40 + "\n" + cr)
logger.success("Classification report saved: reports/classification_report.txt")
logger.info(f"\n{cr}")

#  ROC curve 
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f"AUC={metrics['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Soft Voting Ensemble v3", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
logger.success("ROC curve saved: reports/roc_curve.png")

#  PR curve 
pr_p, pr_r, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(pr_r, pr_p, lw=2, label=f"AP={metrics['avg_precision']:.3f}")
plt.axvline(x=metrics["recall"], color="red", ls=":", alpha=0.7,
            label=f"Op. point (t={final_threshold:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — v3", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()

#   MLflow 
main_run_id = None
if os.path.exists("data/processed/main_run_id.txt"):
    with open("data/processed/main_run_id.txt") as f:
        main_run_id = f.read().strip()

try:
    run_kwargs = dict(run_id=main_run_id) if main_run_id else dict(run_name="evaluation_v3")

    with mlflow.start_run(**run_kwargs, nested=False):
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items() if isinstance(v, float)})
        for art in ["reports/metrics.json", "reports/roc_curve.png",
                    "reports/pr_curve.png", "reports/confusion_matrix.txt",
                    "reports/classification_report.txt"]:
            if os.path.exists(art):
                mlflow.log_artifact(art)
        mlflow.log_param("model_source", "pipeline_config_v3")
    logger.success("Evaluation metrics logged to MLflow")
except Exception as e:
    logger.warning(f"MLflow logging skipped: {e}")

#  final summary 
logger.info("*" * 70)
logger.info("EVALUATION SUMMARY")
logger.info("*" * 70)
logger.info(f"Recall   : {metrics['recall']:.4f}  (bad credit detection rate)")
logger.info(f"Precision: {metrics['precision']:.4f}")
logger.info(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
logger.info(f"F2       : {metrics['f2_score']:.4f}")

if metrics["recall"] >= 0.85:
    logger.success("✓ Recall target ≥ 85% achieved!")
elif metrics["recall"] >= 0.70:
    logger.success("✓ Recall target ≥ 70% achieved")
else:
    logger.warning("⚠ Model below recall target (<70%)")

logger.success("Evaluation completed!")