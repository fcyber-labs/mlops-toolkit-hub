"""

Lending Club Credit Risk — LightGBM Training Pipeline

"""

import os
import sys
import json
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
)
from mlflow.models import infer_signature
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.logging_config import logger

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


load_dotenv()


#  MLflow setup
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

#  macOS / OpenMP safety
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# Config

RANDOM_STATE = 42
N_TRIALS = 1  # CPU case --- Optuna trials USE on GPU min. 20
OPTUNA_FRAC = 0.40
COST_FN = 13_500
COST_FP = 1_800


# Optuna objective


def _lgb_objective(trial, X_opt, y_opt, X_val, y_val, pos_weight: float) -> float:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": 2000,
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "scale_pos_weight": pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_opt,
        y_opt,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
    )
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    trial.report(auc, step=model.best_iteration_)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return auc


# Business-optimal threshold


def find_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: int = COST_FN,
    cost_fp: int = COST_FP,
) -> tuple[float, dict]:
    """
    Sweep thresholds on the validation set.
    Among those with Precision ≥ 0.65 AND Recall ≥ 0.60, pick lowest cost.
    Falls back to unconstrained minimum cost if no threshold meets both constraints.

    """
    thresholds = np.arange(0.10, 0.90, 0.005)
    costs, recalls, precisions, f2s = [], [], [], []

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
        costs.append(fp * cost_fp + fn * cost_fn)
        recalls.append(rec)
        precisions.append(prec)
        f2s.append(f2)

    costs = np.array(costs)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    f2s = np.array(f2s)

    mask = (precisions >= 0.65) & (recalls >= 0.60)
    if mask.any():
        best_t = float(thresholds[mask][np.argmin(costs[mask])])
    else:
        logger.warning(
            "No threshold met Prec≥0.65 & Recall≥0.60 — using unconstrained minimum cost"
        )
        best_t = float(thresholds[np.argmin(costs)])

    sweep = {
        "thresholds": thresholds.tolist(),
        "costs": costs.tolist(),
        "recalls": recalls.tolist(),
        "precisions": precisions.tolist(),
        "f2s": f2s.tolist(),
    }
    return best_t, sweep


# Main training function


def train(model_path: str) -> dict:
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE  —  LightGBM + Optuna")
    logger.info("=" * 70)

    #  Load data
    logger.info("Loading processed data …")
    X_train = pd.read_csv("data/processed/X_train_full.csv")
    y_train = pd.read_csv("data/processed/y_train_full.csv").iloc[:, 0]
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").iloc[:, 0]
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

    logger.info(f"Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")
    logger.info(f"Train default rate : {y_train.mean():.2%}")

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(f"scale_pos_weight   : {pos_weight:.2f}")

    #  Subsample for Optuna
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train), size=int(len(X_train) * OPTUNA_FRAC), replace=False
    )
    X_opt = X_train.iloc[idx]
    y_opt = y_train.iloc[idx]
    logger.info(
        f"Optuna subsample   : {len(X_opt):,} rows  ({OPTUNA_FRAC:.0%} of train)"
    )

    #  Optuna hyperparameter search
    logger.info(f"Running Optuna ({N_TRIALS} trials) …")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=100),
    )
    study.optimize(
        lambda t: _lgb_objective(t, X_opt, y_opt, X_val, y_val, pos_weight),
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
    )

    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    logger.info(f"Best val AUC   : {study.best_value:.4f}")
    logger.info(f"Completed / Pruned : {len(study.trials)} / {pruned}")
    logger.info(f"Best params    : {study.best_params}")

    #  Final model on full train + val
    best_params = {
        **study.best_params,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "scale_pos_weight": pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    logger.info("Training final model on train + val …")
    final_model = LGBMClassifier(**best_params)
    final_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    logger.info(f"Best iteration : {final_model.best_iteration_}")

    #  Business-optimal threshold
    logger.info("Finding business-optimal threshold on validation set …")
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    threshold, sweep = find_threshold(y_val.values, y_val_proba)
    logger.info(f"Optimal threshold : {threshold:.3f}")

    #  Test-set evaluation
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test.values, y_test_pred).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "avg_precision": float(average_precision_score(y_test, y_test_proba)),
        "matthews_corr": float(matthews_corrcoef(y_test, y_test_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "best_cv_auc": float(study.best_value),
        "business_cost": int(fp * COST_FP + fn * COST_FN),
    }

    logger.info("─" * 50)
    logger.info("TEST SET RESULTS")
    logger.info("─" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k.upper():<22}: {v}")

    #  Save model
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(final_model, model_path)
    booster_path = model_path.replace(".pkl", "_booster.txt")
    final_model.booster_.save_model(booster_path)
    logger.info(f"Model saved    : {model_path}")
    logger.info(f"Booster saved  : {booster_path}")

    #  Save threshold + sweep for evaluate.py
    os.makedirs("reports", exist_ok=True)
    threshold_data = {
        "threshold": threshold,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "sweep": sweep,
    }
    with open("reports/threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)

    #  Save training metadata
    metadata = {
        "model_type": "LightGBM",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params": best_params,
        "best_cv_auc": study.best_value,
        "optuna_trials": N_TRIALS,
        "optuna_subsample": OPTUNA_FRAC,
        "test_metrics": metrics,
        "n_features": X_train.shape[1],
        "n_train_samples": X_train.shape[0] + X_val.shape[0],
        "n_test_samples": X_test.shape[0],
        "threshold": threshold,
        "cost_matrix": {"FN": COST_FN, "FP": COST_FP},
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    #  Feature importance plot
    fi = (
        pd.DataFrame(
            {"feature": X_train.columns, "importance": final_model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#0284C7")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Top 20 Features — LightGBM")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()

    #  MLflow logging
    _log_to_mlflow(final_model, best_params, metrics, study, X_test, y_test_proba)

    logger.info("Training complete ✓")
    return metrics


def _log_to_mlflow(model, params, metrics, study, X_test, y_test_proba):
    """Log params, metrics, and artefacts to MLflow."""

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    mlflow.set_experiment("phase_2_lending_club")

    try:
        with mlflow.start_run(run_name="lgb_optuna") as run:
            mlflow.set_tag("model_type", "lightgbm")
            mlflow.set_tag("optimization", f"optuna_{N_TRIALS}trials")

            # Log all parameters
            mlflow.log_params(params)

            # Log all metrics
            mlflow.log_metrics(
                {
                    "roc_auc": float(metrics["roc_auc"]),
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "avg_precision": float(metrics["avg_precision"]),
                    "matthews_corr": float(metrics["matthews_corr"]),
                    "best_cv_auc": float(study.best_value),
                    "tn": float(metrics["tn"]),
                    "fp": float(metrics["fp"]),
                    "fn": float(metrics["fn"]),
                    "tp": float(metrics["tp"]),
                    "business_cost": float(metrics["business_cost"]),
                    "threshold": float(metrics["threshold"]),
                }
            )

            # Log artifacts
            for artefact in [
                "reports/feature_importance.png",
                "reports/threshold.json",
                "models/model_metadata.json",
            ]:
                if os.path.exists(artefact):
                    mlflow.log_artifact(artefact)

            # ROC curve
            from sklearn.metrics import roc_curve as _roc_curve

            y_test_true = pd.read_csv("data/processed/y_test.csv").iloc[:, 0].values
            fpr, tpr, _ = _roc_curve(y_test_true, y_test_proba)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, lw=2, label=f"AUC={metrics['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
            ax.set_title("ROC Curve — LightGBM")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("/tmp/roc_curve.png", dpi=150)
            plt.close()
            mlflow.log_artifact("/tmp/roc_curve.png")

            # Log model
            sig = infer_signature(X_test, y_test_proba)
            mlflow.sklearn.log_model(model, "lgb_model", signature=sig)

            logger.info(f"MLflow run : {run.info.run_id}")

            # Save run_id for evaluate/explain
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/main_run_id.txt", "w") as f:
                f.write(run.info.run_id)
            logger.info(f"Main run ID saved: {run.info.run_id}")

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM training pipeline")
    parser.add_argument("--model", default=None, help="Output model path (.pkl)")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    model_path = args.model or params["train"]["model"]
    train(model_path)
