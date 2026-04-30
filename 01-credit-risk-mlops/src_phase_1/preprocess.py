import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# For macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"


import lightgbm as lgb

import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks

warnings.filterwarnings("ignore")
sys.path.append(".")
from config.logging_config import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — using RF feature_importances_ fallback")

params = yaml.safe_load(open("params.yaml"))["preprocess"]
RANDOM_STATE = params.get("random_state", 42)


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove special characters from column names for LightGBM compatibility"""
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    df.columns = df.columns.str.replace('__', '_', regex=False)
    df.columns = df.columns.str.strip('_')
    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering from the v3 notebook."""
    df = df.copy()

    df["Saving accounts"]  = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")

    #  monotone transforms 
    df["age_log"]    = np.log1p(df["Age"])
    df["age_sqrt"]   = np.sqrt(df["Age"])
    df["age_squared"] = df["Age"] ** 2
    df["age_risk_bucket"] = pd.cut(
        df["Age"], bins=[0, 25, 35, 50, 65, 100],
        labels=[0, 1, 2, 1, 0], ordered=False
    ).astype(int)

    df["credit_amount_log"]    = np.log1p(df["Credit amount"])
    df["credit_amount_sqrt"]   = np.sqrt(df["Credit amount"])
    df["credit_amount_boxcox"], _ = stats.boxcox(df["Credit amount"] + 1)

    df["duration_log"]   = np.log1p(df["Duration"])
    df["duration_sqrt"]  = np.sqrt(df["Duration"])
    df["duration_years"] = df["Duration"] / 12

    #  ratios / interactions 
    df["credit_per_age"]      = df["Credit amount"] / (df["Age"] + 1)
    df["credit_per_duration"] = df["Credit amount"] / (df["Duration"] + 1)
    df["monthly_obligation"]  = df["Credit amount"] / (df["Duration"] + 1)
    df["age_duration_ratio"]  = df["Age"] / (df["Duration"] + 1)
    df["loan_stress"]         = df["Credit amount"] / (df["Duration"] ** 0.5 + 1)

    #  domain risk scores 
    df["age_risk_score"] = np.where(
        df["Age"] < 25, 3,
        np.where(df["Age"] < 30, 2,
                 np.where(df["Age"] < 40, 1, 0.5))
    )
    df["job_risk_score"]      = df["Job"].map({0: 0.8, 1: 0.6, 2: 0.3, 3: 0.1})
    df["housing_risk_score"]  = df["Housing"].map({"own": 0.1, "rent": 0.6, "free": 0.9})
    df["savings_risk_score"]  = df["Saving accounts"].map(
        {"little": 0.8, "moderate": 0.5, "quite rich": 0.2, "rich": 0.1, "unknown": 0.6}
    )
    df["checking_risk_score"] = df["Checking account"].map(
        {"little": 0.8, "moderate": 0.5, "rich": 0.2, "unknown": 0.6}
    )
    df["composite_risk"] = (
        df["age_risk_score"]     * 0.2
        + df["savings_risk_score"]  * 0.3
        + df["checking_risk_score"] * 0.3
        + df["housing_risk_score"]  * 0.2
    )

    #  smoothed target encoding 
    global_mean = df["Risk"].mean()
    for col in ["Purpose", "Sex"]:
        col_stats = df.groupby(col)["Risk"].agg(["mean", "count"])
        col_stats["smooth"] = (
            (col_stats["mean"] * col_stats["count"] + global_mean * 10)
            / (col_stats["count"] + 10)
        )
        df[f"{col}_target_enc"] = df[col].map(col_stats["smooth"])

    #  group statistical features 
    for grp_col in ["Purpose", "Housing"]:
        for val_col in ["Age", "Credit amount", "Duration"]:
            grp_mean = df.groupby(grp_col)[val_col].transform("mean")
            grp_std  = df.groupby(grp_col)[val_col].transform("std").fillna(0)
            df[f"{val_col}_mean_by_{grp_col}"]   = grp_mean
            df[f"{val_col}_std_by_{grp_col}"]    = grp_std
            df[f"{val_col}_zscore_in_{grp_col}"] = (df[val_col] - grp_mean) / (grp_std + 1e-6)

    #  outlier flags 
    for col in ["Age", "Credit amount", "Duration"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[f"{col}_outlier"] = (
            (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        ).astype(int)

    #  clustering features 
    _sc     = StandardScaler()
    _scaled = _sc.fit_transform(df[["Age", "Credit amount", "Duration"]])
    _km     = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    df["credit_cluster"]  = _km.fit_predict(_scaled)
    df["dist_to_cluster"] = np.min(_km.transform(_scaled), axis=1)

    #  polynomial interactions 
    df["age_x_credit"]    = df["Age"] * df["credit_amount_log"]
    df["age_x_duration"]  = df["Age"] * df["duration_log"]
    df["stress_x_risk"]   = df["loan_stress"] * df["composite_risk"]
    df["savings_x_check"] = df["savings_risk_score"] * df["checking_risk_score"]

    #  rank features 
    for col in ["Credit amount", "Duration", "Age"]:
        df[f"{col}_rank"] = df[col].rank(pct=True)

    return df


def _adversarial_validation(df_raw: pd.DataFrame) -> float:
    """Check train/test distribution similarity. Returns AUC (closer to 0.5 = better)."""
    from sklearn.model_selection import cross_val_score
    from lightgbm import LGBMClassifier

    X_adv = df_raw.drop("Risk", axis=1).copy()
    X_adv["Saving accounts"]  = X_adv["Saving accounts"].fillna("unknown")
    X_adv["Checking account"] = X_adv["Checking account"].fillna("unknown")
    X_adv_enc = pd.get_dummies(X_adv)
    
    # Clean feature names 
    X_adv_enc = clean_feature_names(X_adv_enc)

    X_tr, X_te = train_test_split(X_adv_enc, test_size=0.20, random_state=RANDOM_STATE)
    X_all = pd.concat([X_tr, X_te])
    y_all = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])

    clf = LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbosity=-1)
    auc = cross_val_score(clf, X_all, y_all, cv=5, scoring="roc_auc").mean()
    return float(auc)


def preprocess(input_path: str, output_path: str):
    logger.info("*" * 70)
    logger.info("PREPROCESSING PIPELINE v3")
    logger.info("*" * 70)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    #  load 
    df_raw = pd.read_csv(input_path)
    logger.info(f"Raw data: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    df_raw["Risk"] = df_raw["Risk"].map({"good": 0, "bad": 1})
    logger.info(f"Target distribution: {df_raw['Risk'].value_counts().to_dict()}")

    #  adversarial validation 
    adv_auc = _adversarial_validation(df_raw)
    logger.info(f"Adversarial validation AUC: {adv_auc:.4f}")
    if adv_auc > 0.6:
        logger.warning("Train/test distributions differ — consider domain-aware split")
    else:
        logger.info("Train/test distributions OK — random split acceptable")

    #  feature engineering 
    df = _feature_engineering(df_raw)
    logger.info(f"Feature engineering: {df_raw.shape[1]} → {df.shape[1]} columns")

    #  one-hot encoding 
    categorical_columns = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    df_encoded = df_encoded.drop(columns=categorical_columns, errors="ignore")

    # clean column names
    df_encoded = clean_feature_names(df_encoded)
    logger.info(f"After encoding: {df_encoded.shape}")

    #  X / y 
    X_raw = df_encoded.drop("Risk", axis=1)
    y     = df_encoded["Risk"]

    #  power transform 
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    skewed_cols  = [c for c in numeric_cols if abs(X_raw[c].skew()) > 1.0]
    pt = PowerTransformer(method="yeo-johnson")
    X_transformed = X_raw.copy()
    if skewed_cols:
        X_transformed[skewed_cols] = pt.fit_transform(X_raw[skewed_cols])
        logger.info(f"Yeo-Johnson applied to {len(skewed_cols)} skewed features")

    #  train / test split 
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_transformed, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train_full.shape[0]} | Test: {X_test.shape[0]}")
    logger.info(f"Train imbalance: {y_train_full.mean():.2%}")

    #  SHAP / RF feature selection 
    _rf_quick = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    )
    _rf_quick.fit(X_train_full, y_train_full)

    if SHAP_AVAILABLE:
        import shap as _shap
        _explainer = _shap.TreeExplainer(_rf_quick)
        _shap_vals = _explainer.shap_values(X_train_full)
        if isinstance(_shap_vals, list):
            _shap_vals = _shap_vals[1]
        elif _shap_vals.ndim == 3:
            _shap_vals = _shap_vals[:, :, 1]
        _feat_imp = pd.Series(np.abs(_shap_vals).mean(axis=0), index=X_train_full.columns)
        logger.info("Feature importance: SHAP")
    else:
        _feat_imp = pd.Series(_rf_quick.feature_importances_, index=X_train_full.columns)
        logger.info("Feature importance: RF gain (SHAP unavailable)")

    selected_features = _feat_imp.sort_values(ascending=False).head(60).index.tolist()
    X_train = X_train_full[selected_features]
    X_test  = X_test[selected_features]
    logger.info(f"Feature selection: {len(X_train_full.columns)} → {len(selected_features)}")
    logger.info(f"Top 10: {selected_features[:10]}")

    #  BorderlineSMOTE + TomekLinks 
    bsmote = BorderlineSMOTE(
        sampling_strategy=0.65,
        random_state=RANDOM_STATE,
        k_neighbors=5,
        kind="borderline-1",
    )
    X_resampled, y_resampled = bsmote.fit_resample(X_train, y_train_full)
    tomek = TomekLinks(n_jobs=-1)
    X_resampled, y_resampled = tomek.fit_resample(X_resampled, y_resampled)
    logger.info(f"After resampling: {X_resampled.shape[0]} samples | "
                f"imbalance: {y_resampled.mean():.2%}")

    #  save artifacts 

    df_encoded.to_csv(output_path, index=False)
    logger.info(f"Full encoded dataset saved: {output_path}")

    # resampled train
    X_resampled.to_csv("data/processed/X_train_resampled.csv", index=False)
    pd.Series(y_resampled, name="Risk").to_csv(
        "data/processed/y_train_resampled.csv", index=False
    )

    # test set 
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test_df = pd.DataFrame(y_test.values, columns=["Risk_bad"])
    y_test_df.to_csv("data/processed/y_test.csv", index=False)

    # transformers and metadata
    joblib.dump(pt, "data/processed/power_transformer.pkl")
    joblib.dump(selected_features, "data/processed/selected_features.pkl")

    preprocess_meta = {
        "adversarial_auc": adv_auc,
        "n_raw_features":  int(df_raw.shape[1]),
        "n_engineered":    int(df.shape[1]),
        "n_encoded":       int(df_encoded.shape[1]),
        "n_selected":      len(selected_features),
        "selected_features": selected_features,
        "n_train_resampled": int(X_resampled.shape[0]),
        "n_test":          int(X_test.shape[0]),
        "train_imbalance": float(y_resampled.mean()),
        "test_imbalance":  float(y_test.mean()),
        "skewed_cols":     skewed_cols,
    }
    with open("data/processed/preprocess_meta.json", "w") as f:
        json.dump(preprocess_meta, f, indent=2)

    logger.success("Preprocessing completed successfully!")
    return X_train, X_test, y_train_full, y_test, selected_features


if __name__ == "__main__":
    preprocess(params["input"], params["output"])