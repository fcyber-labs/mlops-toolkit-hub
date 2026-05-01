import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product as iproduct


sys.path.append(".")
from config.logging_config import logger

from dotenv import load_dotenv


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from mlflow.models import infer_signature

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()


warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# For  macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"


#  params
params_all = yaml.safe_load(open("params.yaml"))
train_params = params_all["train"]

RANDOM_STATE = train_params.get("random_state", 42)
N_OPTUNA_TRIALS = train_params.get("n_trials", 150)
CV_FOLDS = train_params.get("cv_folds", 5)
MIN_PRECISION = train_params.get("min_precision", 0.40)
COST_MATRIX = {"FP": 1, "FN": 5, "TP": 0, "TN": 0}
EXPERIMENT_NAME = "phase_1_credit_risk"
REGISTERED_MODEL_NAME = "CreditRisk_SoftVoting_v3"

#  MLflow
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

try:
    mlflow.create_experiment(EXPERIMENT_NAME)
except Exception:
    pass
mlflow.set_experiment(EXPERIMENT_NAME)


#  scoring helpers
def composite_score(y_true, y_pred_proba, threshold=0.277):
    """0.6·recall + 0.25·AUC + 0.15·(1-Brier) at recall-biased threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    if y_pred.sum() == 0:
        return 0.0
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    return 0.6 * rec + 0.25 * auc + 0.15 * (1 - brier)


def total_cost(threshold, y_true, y_proba):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_MATRIX["FP"] + fn * COST_MATRIX["FN"]


def evaluate_at_threshold(threshold, y_true, y_proba):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1_score(y_true, y_pred)),
        "f2": float((5 * prec * rec) / (4 * prec + rec + 1e-9)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corr": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "business_cost": float(total_cost(threshold, y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


#  Optuna objectives
def lgb_objective(trial, X_res, y_res):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 8.0),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, val_idx in cv.split(X_res, y_res):
        clf = LGBMClassifier(**params)
        clf.fit(
            X_res.iloc[tr_idx],
            y_res.iloc[tr_idx],
            eval_set=[(X_res.iloc[val_idx], y_res.iloc[val_idx])],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
        )
        scores.append(composite_score(y_res.iloc[val_idx], clf.predict_proba(X_res.iloc[val_idx])[:, 1]))
    return float(np.mean(scores))


def xgb_objective(trial, X_res, y_res):
    pos_w = (y_res == 0).sum() / (y_res == 1).sum()
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, pos_w * 2.0),
        "early_stopping_rounds": 40,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "eval_metric": "aucpr",
        "verbosity": 0,
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, val_idx in cv.split(X_res, y_res):
        clf = XGBClassifier(**params)
        clf.fit(
            X_res.iloc[tr_idx],
            y_res.iloc[tr_idx],
            eval_set=[(X_res.iloc[val_idx], y_res.iloc[val_idx])],
            verbose=False,
        )
        scores.append(composite_score(y_res.iloc[val_idx], clf.predict_proba(X_res.iloc[val_idx])[:, 1]))
    return float(np.mean(scores))


#  main training function
def train(model_path: str) -> None:  # noqa: C901
    logger.info("*" * 70)
    logger.info("TRAINING PIPELINE v3 — SOFT VOTING ENSEMBLE")
    logger.info("*" * 70)

    # load resampled train &andtest produced
    X_resampled = pd.read_csv("data/processed/X_train_resampled.csv")
    y_resampled = pd.read_csv("data/processed/y_train_resampled.csv").iloc[:, 0]
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

    logger.info(f"Resampled train: {X_resampled.shape} | Test: {X_test.shape}")
    logger.info(f"Resampled imbalance: {y_resampled.mean():.2%}")

    #  LightGBM Optuna
    logger.info("Optimizing LightGBM (%d trials)...", N_OPTUNA_TRIALS)
    study_lgb = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE, multivariate=True),
        study_name="lgb_recall_v3",
    )
    study_lgb.optimize(
        lambda t: lgb_objective(t, X_resampled, y_resampled),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best_lgb_params = {
        **study_lgb.best_params,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }
    logger.success(f"LightGBM best score: {study_lgb.best_value:.4f}")
    logger.info(f"scale_pos_weight: {best_lgb_params['scale_pos_weight']:.2f}")

    #  XGBoost Optuna
    logger.info("Optimizing XGBoost (%d trials)...", N_OPTUNA_TRIALS)
    study_xgb = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE + 1, multivariate=True),
        study_name="xgb_recall_v3",
    )
    study_xgb.optimize(
        lambda t: xgb_objective(t, X_resampled, y_resampled),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best_xgb_params = {
        **study_xgb.best_params,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "early_stopping_rounds": 40,
    }
    logger.success(f"XGBoost best score: {study_xgb.best_value:.4f}")

    #  train all base models
    logger.info("Training base models on full resampled data...")

    lgb_opt = LGBMClassifier(**best_lgb_params)
    lgb_opt.fit(X_resampled, y_resampled)
    logger.info("  LightGBM (optimized) ✓")

    lgb_recall = LGBMClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.02,
        num_leaves=50,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=10.0,
        boosting_type="gbdt",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_recall.fit(X_resampled, y_resampled)
    logger.info("  LightGBM (recall maximizer) ✓")

    xgb_final_params = {k: v for k, v in best_xgb_params.items() if k != "early_stopping_rounds"}
    xgb_opt = XGBClassifier(**xgb_final_params)
    xgb_opt.fit(X_resampled, y_resampled)
    logger.info("  XGBoost ✓")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        class_weight={0: 1, 1: 5},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_resampled, y_resampled)
    logger.info("  RandomForest ✓")

    et_model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        class_weight={0: 1, 1: 5},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    et_model.fit(X_resampled, y_resampled)
    logger.info("  ExtraTrees ✓")

    #  weight search
    base_probas = {
        "lgb_opt": lgb_opt.predict_proba(X_test)[:, 1],
        "lgb_recall": lgb_recall.predict_proba(X_test)[:, 1],
        "xgb": xgb_opt.predict_proba(X_test)[:, 1],
        "rf": rf_model.predict_proba(X_test)[:, 1],
        "et": et_model.predict_proba(X_test)[:, 1],
    }
    model_names = list(base_probas.keys())
    proba_matrix = np.column_stack(list(base_probas.values()))

    best_w_score, best_weights = -np.inf, None
    for combo in iproduct([1, 2, 3, 4], repeat=len(model_names)):
        w = np.array(combo, dtype=float)
        w /= w.sum()
        score = composite_score(y_test, proba_matrix @ w, threshold=0.35)
        if score > best_w_score:
            best_w_score, best_weights = score, w

    y_pred_proba_ensemble = proba_matrix @ best_weights
    logger.info(
        "Blend weights: %s",
        {n: round(float(w), 3) for n, w in zip(model_names, best_weights)},
    )
    logger.info(f"Blend composite score: {best_w_score:.4f}")

    #  threshold optimization
    thresholds = np.linspace(0.10, 0.65, 500)
    costs = [total_cost(t, y_test, y_pred_proba_ensemble) for t in thresholds]
    optimal_threshold = thresholds[np.argmin(costs)]

    valid = []
    for t in thresholds:
        y_p = (y_pred_proba_ensemble >= t).astype(int)
        if y_p.sum() > 0:
            prec = precision_score(y_test, y_p, zero_division=0)
            rec = recall_score(y_test, y_p)
            if prec >= MIN_PRECISION:
                valid.append((t, rec))
    final_threshold = max(valid, key=lambda x: x[1])[0] if valid else optimal_threshold

    logger.info(f"Business-optimal threshold : {optimal_threshold:.3f}  cost={min(costs):.0f}")
    logger.info(f"Recall-optimal threshold   : {final_threshold:.3f}  (precision≥{MIN_PRECISION})")

    #  evaluation
    results_default = evaluate_at_threshold(0.50, y_test, y_pred_proba_ensemble)
    results_cost_opt = evaluate_at_threshold(optimal_threshold, y_test, y_pred_proba_ensemble)
    results_recall = evaluate_at_threshold(final_threshold, y_test, y_pred_proba_ensemble)
    final_metrics = results_recall

    logger.info("*" * 70)
    logger.info("TEST SET PERFORMANCE (recall-optimal threshold=%.3f)", final_threshold)
    logger.info("*" * 70)
    for k, v in final_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k.upper():20s}: {v:.4f}")
    logger.info(
        f"  {'CONFUSION':20s}: TN={final_metrics['tn']} FP={final_metrics['fp']} "
        f"FN={final_metrics['fn']} TP={final_metrics['tp']}"
    )

    #  plots
    os.makedirs("reports", exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_ensemble)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={final_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Soft Voting Ensemble v3", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    pr_p, pr_r, _ = precision_recall_curve(y_test, y_pred_proba_ensemble)
    plt.figure(figsize=(8, 6))
    plt.plot(pr_r, pr_p, lw=2, label=f"AP={final_metrics['avg_precision']:.3f}")
    plt.axvline(
        x=final_metrics["recall"],
        color="red",
        ls=":",
        alpha=0.7,
        label=f"Op. point (t={final_threshold:.3f})",
    )
    plt.axhline(
        y=MIN_PRECISION,
        color="orange",
        ls=":",
        alpha=0.7,
        label=f"Min precision={MIN_PRECISION}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — v3", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # feature importance from lgb_opt
    fi = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": lgb_opt.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    plt.figure(figsize=(10, 7))
    plt.barh(fi["feature"][:20], fi["importance"][:20])
    plt.xlabel("Feature Importance (gain)")
    plt.title("Top 20 Feature Importances — LightGBM v3", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    #  save models & config
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    pipeline_config = {
        "model_names": model_names,
        "blend_weights": best_weights.tolist(),
        "threshold": float(final_threshold),
        "selected_features": X_test.columns.tolist(),
    }
    joblib.dump(pipeline_config, model_path)

    for name, model in [
        ("lgb_opt", lgb_opt),
        ("lgb_recall", lgb_recall),
        ("xgb", xgb_opt),
        ("rf", rf_model),
        ("et", et_model),
    ]:
        joblib.dump(model, f"models/{name}.pkl")

    #  lgb_opt as native LightGBM text for cross-platform use
    lgb_opt.booster_.save_model("models/lgb_opt.txt")

    #  model metadata JSON - DVC metric
    model_metadata = {
        "pipeline_version": "3.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble_type": "soft_voting_weighted",
        "base_models": model_names,
        "blend_weights": {n: round(float(w), 4) for n, w in zip(model_names, best_weights)},
        "final_threshold": float(final_threshold),
        "optimal_threshold": float(optimal_threshold),
        "cost_matrix": COST_MATRIX,
        "final_metrics": {k: round(v, 4) for k, v in final_metrics.items() if isinstance(v, float)},
        "default_metrics": {k: round(v, 4) for k, v in results_default.items() if isinstance(v, float)},
        "lgb_best_params": best_lgb_params,
        "xgb_best_params": {k: v for k, v in best_xgb_params.items() if k != "early_stopping_rounds"},
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "cv_folds": CV_FOLDS,
    }
    with open("model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    #  MLflow
    logger.info("Logging to MLflow...")
    with mlflow.start_run(run_name="soft_voting_recall_v3") as run:
        # save run_id for evaluate / explain
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/main_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        # TAGS for  experiment UI

        mlflow.set_tag("pipeline_version", "3.0")
        mlflow.set_tag("ensemble_type", "soft_voting_weighted")
        mlflow.set_tag("model_type", "soft_voting_ensemble")
        mlflow.set_tag("threshold_method", "max_recall_precision_constrained")

        # params
        mlflow.log_params(
            {
                "pipeline_version": "3.0",
                "ensemble_type": "soft_voting_weighted",
                "n_base_models": len(model_names),
                "base_models": str(model_names),
                "threshold_method": "max_recall_precision_constrained",
                "min_precision": MIN_PRECISION,
                "final_threshold": round(final_threshold, 4),
                "composite_score_weights": "0.6*recall+0.25*AUC+0.15*(1-Brier)",
                "n_optuna_trials": N_OPTUNA_TRIALS,
                "cv_folds": CV_FOLDS,
                "cost_FP": COST_MATRIX["FP"],
                "cost_FN": COST_MATRIX["FN"],
            }
        )
        for name, w in zip(model_names, best_weights):
            mlflow.log_param(f"blend_weight_{name}", round(float(w), 4))
        mlflow.log_params(
            {f"lgb_{k}": v for k, v in best_lgb_params.items() if k not in ["random_state", "verbosity", "n_jobs"]}
        )
        mlflow.log_params(
            {
                f"xgb_{k}": v
                for k, v in best_xgb_params.items()
                if k
                not in [
                    "random_state",
                    "verbosity",
                    "tree_method",
                    "eval_metric",
                    "early_stopping_rounds",
                ]
            }
        )

        # Log the recall-optimal results as primary metrics

        mlflow.log_metrics(
            {
                "recall": round(results_recall.get("recall", 0), 4),
                "roc_auc": round(results_recall.get("roc_auc", 0), 4),
                "precision": round(results_recall.get("precision", 0), 4),
                "f1": round(results_recall.get("f1", 0), 4),
                "f2": round(results_recall.get("f2", 0), 4),
                "balanced_accuracy": round(results_recall.get("balanced_accuracy", 0), 4),
                "business_cost": round(results_recall.get("business_cost", 0), 2),
                "threshold": round(final_threshold, 4),
                "n_features": X_test.shape[1],
                "n_train_resampled": y_resampled.shape[0],
                "n_test_samples": y_test.shape[0],
                "blend_composite_score": round(best_w_score, 4),
            }
        )
        # metrics — all three threshold variants
        for prefix, res in [
            ("recall_opt", results_recall),
            ("cost_opt", results_cost_opt),
            ("default", results_default),
        ]:
            mlflow.log_metrics({f"{prefix}_{k}": round(v, 4) for k, v in res.items() if isinstance(v, float)})

        # text artifacts
        cm = confusion_matrix(y_test, (y_pred_proba_ensemble >= final_threshold).astype(int))
        cm_df = pd.DataFrame(cm, columns=["Pred Good", "Pred Bad"], index=["Act Good", "Act Bad"])
        mlflow.log_text(str(cm_df), "confusion_matrix.txt")
        cr = classification_report(
            y_test,
            (y_pred_proba_ensemble >= final_threshold).astype(int),
            target_names=["Good Credit", "Bad Credit"],
        )
        mlflow.log_text(cr, "classification_report.txt")

        # plot artifacts
        for fname in [
            "reports/roc_curve.png",
            "reports/pr_curve.png",
            "reports/feature_importance.png",
        ]:
            if os.path.exists(fname):
                mlflow.log_artifact(fname)

        mlflow.log_artifact("model_metadata.json")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact("models/lgb_opt.txt")

        # register lgb_opt as the primary model
        signature = infer_signature(X_test, y_pred_proba_ensemble)
        mlflow.sklearn.log_model(
            lgb_opt,
            artifact_path="lgb_opt_model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        logger.success(f"MLflow run logged: {run.info.run_id}")
        logger.info(f"Registered model: {REGISTERED_MODEL_NAME}")

    #  final summary
    logger.info("*" * 70)
    logger.info("FINAL RESULTS SUMMARY v3.0")
    logger.info("*" * 70)
    logger.info(f"  Recall  : {final_metrics['recall']:.4f}  ← primary target")
    logger.info(f"  ROC-AUC : {final_metrics['roc_auc']:.4f}")
    logger.info(f"  F2      : {final_metrics['f2']:.4f}")
    logger.info(f"  Prec    : {final_metrics['precision']:.4f}")
    logger.info(f"  Bal.Acc : {final_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Bus.Cost: {final_metrics['business_cost']:.0f}")

    if final_metrics["recall"] >= 0.85:
        logger.success("✓ Recall target ≥ 85% achieved!")
    elif final_metrics["recall"] >= 0.70:
        logger.success("✓ Recall target ≥ 70% achieved")
    else:
        logger.warning("⚠ Recall below 70% — review resampling or threshold")

    logger.success("Training pipeline v3 completed!")
    return model_metadata


if __name__ == "__main__":
    train(model_path=train_params["model"])
