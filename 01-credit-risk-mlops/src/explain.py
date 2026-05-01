"""

Lending Club Credit Risk — SHAP Explainability Pipeline

"""

import os
import sys
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from config.logging_config import logger

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# macOS / OpenMP safety
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

os.makedirs("reports", exist_ok=True)


#  MLflow setup
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))


# Main


def explain(model_path: str, sample_size: int = 2000) -> None:  # noqa: C901
    logger.info("=" * 70)
    logger.info("EXPLAINABILITY PIPELINE  —  SHAP Analysis")
    logger.info("=" * 70)

    #  Load test data
    logger.info("Loading test data …")
    X_test = pd.read_csv("data/processed/X_test.csv")
    logger.info(f"Test data : {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

    n = min(sample_size, X_test.shape[0])
    X_sample = X_test.sample(n=n, random_state=42)
    logger.info(f"SHAP sample : {n:,} rows")

    #  Load model
    logger.info(f"Loading model from {model_path} …")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        sys.exit(1)
    model = joblib.load(model_path)

    # Use the underlying booster for faster TreeExplainer
    booster = model.booster_ if hasattr(model, "booster_") else model
    logger.info("Using LightGBM booster for TreeExplainer")

    #  Load main run ID from training
    main_run_id = None
    run_id_path = "data/processed/main_run_id.txt"
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            main_run_id = f.read().strip()
        logger.info(f"Found main run ID: {main_run_id}")
    else:
        logger.warning(
            "No main run ID found. SHAP artifacts will be logged to separate run."
        )

    #  SHAP values
    logger.info("Computing SHAP values (this may take ~1 min for 2 k rows) …")
    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_sample)

        # Binary classification: shap_values is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            logger.info("Binary classifier — using class-1 SHAP values")

        logger.info("SHAP values computed successfully")
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        sys.exit(1)

    #  1. Beeswarm summary
    logger.info("Creating beeswarm summary plot …")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        plt.sca(axes[0])
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        axes[0].set_title("SHAP Summary (Beeswarm)", fontsize=13, fontweight="bold")

        plt.sca(axes[1])
        shap.summary_plot(
            shap_values, X_sample, plot_type="bar", show=False, max_display=20
        )
        axes[1].set_title(
            "SHAP Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig("reports/shap_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP combined plot saved : reports/shap_analysis.png")
    except Exception as e:
        logger.warning(f"Combined SHAP plot failed: {e}")

    #  2. Standalone summary
    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary saved      : reports/shap_summary.png")
    except Exception as e:
        logger.warning(f"Standalone summary plot failed: {e}")

    #  3. Bar chart
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample, plot_type="bar", show=False, max_display=20
        )
        plt.tight_layout()
        plt.savefig("reports/shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP bar chart saved    : reports/shap_bar.png")
    except Exception as e:
        logger.warning(f"Bar plot failed: {e}")

    #  4. Feature importance CSV
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    fi = (
        pd.DataFrame({"feature": X_sample.columns, "shap_importance": mean_abs_shap})
        .sort_values("shap_importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv("reports/shap_feature_importance.csv", index=False)
    logger.info("SHAP importance CSV saved : reports/shap_feature_importance.csv")

    logger.info("Top 10 features by SHAP importance:")
    for i, row in fi.head(10).iterrows():
        logger.info(f"  {i + 1:2d}. {row['feature']:<30} {row['shap_importance']:.4f}")

    #  5. Dependence plots for top-3 features
    for feat in fi.head(3)["feature"].tolist():
        if feat not in X_sample.columns:
            continue
        try:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(feat, shap_values, X_sample, show=False)
            plt.title(f"SHAP Dependence — {feat}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            safe_name = feat.replace("/", "_")
            plt.savefig(
                f"reports/shap_dependence_{safe_name}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
            logger.info(
                f"Dependence plot saved   : reports/shap_dependence_{safe_name}.png"
            )
        except Exception as e:
            logger.warning(f"Dependence plot for {feat} failed: {e}")

    #  MLflow logging
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    try:
        mlflow.set_experiment("phase_2_lending_club")

        # Use main run if available, otherwise create separate run
        if main_run_id:
            with mlflow.start_run(run_id=main_run_id, nested=True):
                for artefact in [
                    "reports/shap_analysis.png",
                    "reports/shap_summary.png",
                    "reports/shap_bar.png",
                    "reports/shap_feature_importance.csv",
                    "reports/shap_dependence_grade_x_int_rate.png",
                    "reports/shap_dependence_sub_grade_enc.png",
                    "reports/shap_dependence_composite_risk.png",
                ]:
                    if os.path.exists(artefact):
                        mlflow.log_artifact(artefact)

                mlflow.log_metric(
                    "top_feature_shap", float(fi.iloc[0]["shap_importance"])
                )
                mlflow.log_metric(
                    "second_feature_shap", float(fi.iloc[1]["shap_importance"])
                )
                mlflow.log_metric(
                    "third_feature_shap", float(fi.iloc[2]["shap_importance"])
                )
                mlflow.log_param("top_feature", fi.iloc[0]["feature"])
                mlflow.log_param("shap_sample_size", n)
                mlflow.log_param("n_features", X_sample.shape[1])
                logger.info(f"SHAP artifacts logged to main run: {main_run_id}")
        else:
            with mlflow.start_run(run_name="explainability", nested=False):
                for artefact in [
                    "reports/shap_analysis.png",
                    "reports/shap_summary.png",
                    "reports/shap_bar.png",
                    "reports/shap_feature_importance.csv",
                    "reports/shap_dependence_grade_x_int_rate.png",
                    "reports/shap_dependence_sub_grade_enc.png",
                    "reports/shap_dependence_composite_risk.png",
                ]:
                    if os.path.exists(artefact):
                        mlflow.log_artifact(artefact)

                mlflow.log_metric(
                    "top_feature_shap", float(fi.iloc[0]["shap_importance"])
                )
                mlflow.log_metric(
                    "second_feature_shap", float(fi.iloc[1]["shap_importance"])
                )
                mlflow.log_metric(
                    "third_feature_shap", float(fi.iloc[2]["shap_importance"])
                )
                mlflow.log_param("top_feature", fi.iloc[0]["feature"])
                mlflow.log_param("shap_sample_size", n)
                mlflow.log_param("n_features", X_sample.shape[1])
                logger.info("SHAP artifacts logged to separate run")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    #  Summary
    logger.info("=" * 70)
    logger.info("EXPLAINABILITY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Samples analysed : {n:,}")
    logger.info(f"Features         : {X_sample.shape[1]}")
    logger.info(
        f"Top feature      : {fi.iloc[0]['feature']}  ({fi.iloc[0]['shap_importance']:.4f})"
    )
    logger.info("Explainability complete ✓")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability pipeline")
    parser.add_argument("--model", default="models/model_lending.pkl")
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Number of test samples to use for SHAP (default 2000)",
    )
    args = parser.parse_args()

    explain(args.model, args.n)
