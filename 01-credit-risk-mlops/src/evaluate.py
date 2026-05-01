import os
import sys
import json
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.logging_config import logger

warnings.filterwarnings("ignore")


#  MLflow setup
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))


# Config

COST_FN = 13_500
COST_FP = 1_800
os.makedirs("reports", exist_ok=True)


# Core metrics


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: int = COST_FN,
    cost_fp: int = COST_FP,
) -> dict:
    """Full metrics suite"""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    #  KS Statistic
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_proba)
    ks_idx = int(np.argmax(tpr_arr - fpr_arr))
    ks_statistic = float(np.max(tpr_arr - fpr_arr))
    ks_threshold = float(thr_arr[ks_idx])

    #  Gini
    gini = 2 * auc - 1

    #  Lift curve
    sorted_idx = np.argsort(-y_proba)
    n_total = len(y_true)
    n_pos = int(y_true.sum())
    decile_size = n_total // 10
    base_rate = n_pos / n_total

    lift_at_10 = float(y_true[sorted_idx[:decile_size]].mean() / base_rate)
    lift_at_20 = float(y_true[sorted_idx[: 2 * decile_size]].mean() / base_rate)

    # Lift per decile
    lift_by_decile = [
        float(y_true[sorted_idx[: d * decile_size]].mean() / base_rate)
        for d in range(1, 11)
    ]

    #  Business cost
    business_cost = int(fp * cost_fp + fn * cost_fn)
    random_cost = int(n_pos * 0.5) * cost_fn + int((n_total - n_pos) * 0.5) * cost_fp
    cost_savings = float((1 - business_cost / (random_cost + 1e-9)) * 100)
    profit_score = float(tp * cost_fn / (n_pos * cost_fn + 1e-9) * 100)

    #  Calibration
    brier = float(brier_score_loss(y_true, y_proba))
    logloss = float(log_loss(y_true, y_proba))

    return {
        # Standard
        "recall": float(rec),
        "precision": float(prec),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": float(auc),
        "avg_precision": float(ap),
        # Scoring
        "ks_statistic": ks_statistic,
        "ks_threshold": ks_threshold,
        "gini": float(gini),
        "lift_at_10pct": lift_at_10,
        "lift_at_20pct": lift_at_20,
        "lift_by_decile": lift_by_decile,
        # Calibration
        "brier": brier,
        "log_loss": logloss,
        # Business
        "business_cost": business_cost,
        "cost_savings_pct": cost_savings,
        "profit_score": profit_score,
        # Confusion matrix
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        # Metadata
        "threshold": float(threshold),
        "n_total": n_total,
        "n_pos": n_pos,
        # Internals for charting
        "_fpr_arr": fpr_arr.tolist(),
        "_tpr_arr": tpr_arr.tolist(),
        "_thr_arr": thr_arr.tolist(),
        "_ks_idx": ks_idx,
    }


def print_metrics(m: dict) -> None:
    """Pretty-print the metrics table - Production Version"""
    print("=" * 65)
    print("FULL METRICS SUMMARY")
    print("=" * 65)

    # Core metrics
    print(f"\n  {'Core Metrics':^60}")
    print(f"  {'-' * 60}")
    print(f"  Recall (Sensitivity)     : {m['recall']:.4f}")
    print(f"  Precision (PPV)          : {m['precision']:.4f}")
    print(f"  ROC AUC                  : {m['roc_auc']:.4f}")
    print(f"  KS Statistic             : {m['ks_statistic']:.4f}")
    print(f"  Gini Coefficient         : {m['gini']:.4f}")
    print(f"  Lift @ Top 10%           : {m['lift_at_10pct']:.2f}x")
    print(f"  Lift @ Top 20%           : {m['lift_at_20pct']:.2f}x")

    # Score metrics
    print(f"\n  {'Score Metrics':^60}")
    print(f"  {'-' * 60}")
    print(f"  F1 Score                 : {m['f1']:.4f}")
    print(f"  F2 Score                 : {m['f2']:.4f}")
    print(f"  Avg Precision (PR-AUC)   : {m['avg_precision']:.4f}")
    print(f"  Brier Score              : {m['brier']:.4f}")
    print(f"  Log Loss                 : {m['log_loss']:.4f}")

    # Business metrics
    print(f"\n  {'Business Impact':^60}")
    print(f"  {'-' * 60}")
    print(f"  Business Cost            : ${m['business_cost']:,.0f}")
    print(f"  Cost Savings vs Random   : {m['cost_savings_pct']:.1f}%")
    print(f"  Default Loss Prevented   : {m['profit_score']:.1f}%")

    # Confusion matrix
    print(f"\n  {'Confusion Matrix':^60}")
    print(f"  {'-' * 60}")
    print(f"  True Negatives  (TN)     : {m['tn']:,}")
    print(f"  False Positives (FP)     : {m['fp']:,}")
    print(f"  False Negatives (FN)     : {m['fn']:,}")
    print(f"  True Positives  (TP)     : {m['tp']:,}")

    # Decision threshold
    print(f"\n  {'Decision Threshold':^60}")
    print(f"  {'-' * 60}")
    print(f"  Threshold                : {m['threshold']:.4f}")

    print("=" * 65)


# 9-panel results dashboard


def plot_dashboard(
    m: dict,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    feature_names: list[str],
    feature_importances: np.ndarray,
    sweep: dict,
    out_path: str = "reports/results_dashboard.png",
) -> None:
    fpr_arr = np.array(m["_fpr_arr"])
    tpr_arr = np.array(m["_tpr_arr"])
    thr_arr = np.array(m["_thr_arr"])
    ks_idx = m["_ks_idx"]
    threshold = m["threshold"]

    thresholds = np.array(sweep["thresholds"])
    costs = np.array(sweep["costs"])
    recalls_sw = np.array(sweep["recalls"])
    precs_sw = np.array(sweep["precisions"])
    f2s_sw = np.array(sweep["f2s"])

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    #  1. ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        fpr_arr,
        tpr_arr,
        lw=2.5,
        color="#2563EB",
        label=f"LightGBM AUC={m['roc_auc']:.3f}",
    )
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ks_fpr, ks_tpr = fpr_arr[ks_idx], tpr_arr[ks_idx]
    ax1.annotate(
        "",
        xy=(ks_fpr, ks_tpr),
        xytext=(ks_fpr, ks_fpr),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )
    ax1.text(
        ks_fpr + 0.02,
        (ks_tpr + ks_fpr) / 2,
        f"KS={m['ks_statistic']:.3f}",
        color="red",
        fontsize=10,
    )
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    #  2. Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(
        pr_rec, pr_prec, lw=2.5, color="#16A34A", label=f"AP={m['avg_precision']:.3f}"
    )
    ax2.axvline(
        m["recall"],
        color="red",
        ls=":",
        alpha=0.7,
        label=f"Op.Recall={m['recall']:.3f}",
    )
    ax2.axhline(
        m["precision"],
        color="blue",
        ls=":",
        alpha=0.7,
        label=f"Op.Prec={m['precision']:.3f}",
    )
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    #  3. KS Separation Plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(thr_arr, tpr_arr, color="#2563EB", lw=2, label="TPR (Sensitivity)")
    ax3.plot(thr_arr, fpr_arr, color="#DC2626", lw=2, label="FPR (1-Specificity)")
    ax3.axvline(
        m["ks_threshold"],
        color="green",
        ls="--",
        lw=1.5,
        label=f"KS={m['ks_statistic']:.3f} @ {m['ks_threshold']:.3f}",
    )
    ax3.set(xlabel="Threshold", ylabel="Rate", title="KS Separation Plot")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    #  4. Lift Curve by Decile
    ax4 = fig.add_subplot(gs[1, 0])
    deciles = list(range(1, 11))
    ax4.bar(deciles, m["lift_by_decile"], color="#7C3AED", alpha=0.8, edgecolor="white")
    ax4.axhline(1.0, color="gray", ls="--", lw=1.5, label="Random baseline")
    ax4.set(
        xlabel="Decile",
        ylabel="Lift",
        title="Lift Curve by Decile",
        xticks=deciles,
        xticklabels=[f"{d * 10}%" for d in deciles],
    )
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    #  5. Confusion Matrix
    ax5 = fig.add_subplot(gs[1, 1])
    cm_arr = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt=",d",
        cmap="Blues",
        ax=ax5,
        xticklabels=["Pred Good", "Pred Default"],
        yticklabels=["Actual Good", "Actual Default"],
        annot_kws={"size": 13},
    )
    ax5.set_title(f"Confusion Matrix\n(threshold={threshold:.3f})")

    #  6. Score Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(
        y_proba[y_true == 0],
        bins=60,
        alpha=0.6,
        color="#16A34A",
        label="Good (0)",
        density=True,
    )
    ax6.hist(
        y_proba[y_true == 1],
        bins=60,
        alpha=0.6,
        color="#DC2626",
        label="Default (1)",
        density=True,
    )
    ax6.axvline(
        threshold, color="black", ls="--", lw=2, label=f"Threshold={threshold:.3f}"
    )
    ax6.set(
        xlabel="Predicted Probability",
        ylabel="Density",
        title="Score Distribution by Class",
    )
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

    #  7. Business Cost Curve
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(thresholds, costs / 1e6, color="#EA580C", lw=2.5)
    ax7.axvline(
        threshold, color="black", ls="--", lw=1.5, label=f"Optimal t={threshold:.3f}"
    )
    ax7.set(xlabel="Threshold", ylabel="Cost ($M)", title="Business Cost vs Threshold")
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.3)

    #  8. Recall / Precision / F2 vs Threshold
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(thresholds, recalls_sw, color="#2563EB", lw=2, label="Recall")
    ax8.plot(thresholds, precs_sw, color="#DC2626", lw=2, label="Precision")
    ax8.plot(thresholds, f2s_sw, color="#7C3AED", lw=2, label="F2", ls="--")
    ax8.axvline(threshold, color="black", ls="--", lw=1.5)
    ax8.axhspan(0.65, 0.75, alpha=0.10, color="blue", label="Recall target")
    ax8.axhspan(0.70, 0.85, alpha=0.10, color="red", label="Precision target")
    ax8.set(
        xlabel="Threshold",
        ylabel="Score",
        title="Recall / Precision / F2 vs Threshold",
        ylim=[0, 1],
    )
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)

    #  9. Feature Importance (top 15)
    ax9 = fig.add_subplot(gs[2, 2])
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": feature_importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    ax9.barh(
        fi["feature"][::-1], fi["importance"][::-1], color="#0284C7", edgecolor="white"
    )
    ax9.set(xlabel="Importance (Gain)", title="Top 15 Features (LightGBM)")
    ax9.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Lending Club Credit Risk — LightGBM Results",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Dashboard saved : {out_path}")


# Main


def evaluate(model_path: str, threshold_override: float | None = None) -> dict:  # noqa: C901
    logger.info("=" * 70)
    logger.info("EVALUATION PIPELINE  —  Full Metrics Suite")
    logger.info("=" * 70)

    #  Load data
    logger.info("Loading test data …")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0].values
    logger.info(f"Test set : {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

    #  Load model
    logger.info(f"Loading model from {model_path} …")
    if not os.path.exists(model_path):
        logger.error(f"Model not found : {model_path}")
        sys.exit(1)
    model = joblib.load(model_path)

    #  Load threshold - from train.py
    threshold_path = "reports/threshold.json"
    if threshold_override is not None:
        threshold = threshold_override
        sweep = None
        logger.info(f"Using override threshold : {threshold:.3f}")
    elif os.path.exists(threshold_path):
        with open(threshold_path) as f:
            td = json.load(f)
        threshold = td["threshold"]
        sweep = td["sweep"]
        logger.info(f"Threshold loaded from {threshold_path} : {threshold:.3f}")
    else:
        threshold = 0.50
        sweep = None
        logger.warning(f"threshold.json not found — defaulting to {threshold}")

    #  Load main run ID from training
    main_run_id = None
    run_id_path = "data/processed/main_run_id.txt"
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            main_run_id = f.read().strip()
        logger.info(f"Found main run ID: {main_run_id}")
    else:
        logger.warning("No main run ID found. Metrics will be logged to separate run.")

    #  Predict
    y_proba = model.predict_proba(X_test)[:, 1]

    #  Full metrics
    m = compute_metrics(y_test, y_proba, threshold)
    print_metrics(m)

    #  Save metrics JSON
    export = {k: v for k, v in m.items() if not k.startswith("_")}
    with open("reports/metrics_full.json", "w") as f:
        json.dump(export, f, indent=2)
    logger.info("Full metrics saved : reports/metrics_full.json")

    #  Dashboard
    if sweep is None:
        # Reconstruct a minimal sweep dict if not available
        thr_range = np.arange(0.10, 0.90, 0.005)
        sweep = {
            "thresholds": thr_range.tolist(),
            "costs": [COST_FP * 0 + COST_FN * 0] * len(thr_range),  # dummy !!!
            "recalls": [0.0] * len(thr_range),
            "precisions": [0.0] * len(thr_range),
            "f2s": [0.0] * len(thr_range),
        }

    fi_arr = model.feature_importances_
    fi_names = X_test.columns.tolist()
    plot_dashboard(m, y_test, y_proba, fi_names, fi_arr, sweep)

    #  MLflow logging
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    try:
        mlflow.set_experiment("phase_2_lending_club")

        # Use main run if available, otherwise create separate run
        if main_run_id:
            with mlflow.start_run(run_id=main_run_id, nested=True):
                # Log all metrics (including ints)
                mlflow.log_metrics(
                    {
                        "recall": m["recall"],
                        "precision": m["precision"],
                        "f1": m["f1"],
                        "f2": m["f2"],
                        "roc_auc": m["roc_auc"],
                        "avg_precision": m["avg_precision"],
                        "ks_statistic": m["ks_statistic"],
                        "gini": m["gini"],
                        "lift_at_10pct": m["lift_at_10pct"],
                        "lift_at_20pct": m["lift_at_20pct"],
                        "brier": m["brier"],
                        "log_loss": m["log_loss"],
                        "business_cost": m["business_cost"],
                        "cost_savings_pct": m["cost_savings_pct"],
                        "profit_score": m["profit_score"],
                        "tn": m["tn"],
                        "fp": m["fp"],
                        "fn": m["fn"],
                        "tp": m["tp"],
                    }
                )

                # Log parameters
                mlflow.log_param("model_type", "lightgbm")
                mlflow.log_param("threshold", threshold)
                mlflow.log_param("test_samples", m["n_total"])
                mlflow.log_param("n_features", X_test.shape[1])

                # Log artifacts
                for artefact in [
                    "reports/metrics_full.json",
                    "reports/results_dashboard.png",
                ]:
                    if os.path.exists(artefact):
                        mlflow.log_artifact(artefact)

                # Log ROC curve (create and save)
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f"AUC={m['roc_auc']:.3f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig("/tmp/eval_roc.png", dpi=150)
                plt.close()
                mlflow.log_artifact("/tmp/eval_roc.png")

            logger.info(f"Evaluation metrics logged to main run: {main_run_id}")
        else:
            with mlflow.start_run(run_name="evaluation", nested=False):
                # Log all metrics (including ints)
                mlflow.log_metrics(
                    {
                        "recall": m["recall"],
                        "precision": m["precision"],
                        "f1": m["f1"],
                        "f2": m["f2"],
                        "roc_auc": m["roc_auc"],
                        "avg_precision": m["avg_precision"],
                        "ks_statistic": m["ks_statistic"],
                        "gini": m["gini"],
                        "lift_at_10pct": m["lift_at_10pct"],
                        "lift_at_20pct": m["lift_at_20pct"],
                        "brier": m["brier"],
                        "log_loss": m["log_loss"],
                        "business_cost": m["business_cost"],
                        "cost_savings_pct": m["cost_savings_pct"],
                        "profit_score": m["profit_score"],
                        "tn": m["tn"],
                        "fp": m["fp"],
                        "fn": m["fn"],
                        "tp": m["tp"],
                    }
                )

                # Log parameters
                mlflow.log_param("model_type", "lightgbm")
                mlflow.log_param("threshold", threshold)
                mlflow.log_param("test_samples", m["n_total"])
                mlflow.log_param("n_features", X_test.shape[1])

                # Log artifacts
                for artefact in [
                    "reports/metrics_full.json",
                    "reports/results_dashboard.png",
                ]:
                    if os.path.exists(artefact):
                        mlflow.log_artifact(artefact)

                # Log ROC curve (create and save)
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f"AUC={m['roc_auc']:.3f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig("/tmp/eval_roc.png", dpi=150)
                plt.close()
                mlflow.log_artifact("/tmp/eval_roc.png")

            logger.info("Evaluation metrics logged to separate run")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    logger.info("Evaluation complete ✓")
    return export


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--model", default="models/model_lending.pkl")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override the saved threshold"
    )
    args = parser.parse_args()

    evaluate(args.model, args.threshold)
