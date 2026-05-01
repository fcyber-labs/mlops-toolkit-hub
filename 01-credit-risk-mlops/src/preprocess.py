"""

Lending Club Credit Risk — Preprocessing Pipeline


"""

import sys
import os
import gc
import json
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.logging_config import logger

warnings.filterwarnings("ignore")


#  Constants
RANDOM_STATE = 42
CHUNK_SIZE = 500_000

# Origination-time columns only — no post-origination leakage
FEATURE_COLS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "purpose",
    "initial_list_status",
    "application_type",
    "earliest_cr_line",
    "addr_state",
]


# Feature engineering


def engineer_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:  # noqa: C901
    """
    Apply all v6 feature engineering in-place.
    Returns (transformed_df, label_encoder_dict).
    """
    df = df.copy()

    #  1. Basic type / string cleaning
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    if "int_rate" in df.columns:
        df["int_rate"] = pd.to_numeric(df["int_rate"].astype(str).str.replace("%", ""), errors="coerce")

    if "revol_util" in df.columns:
        df["revol_util"] = pd.to_numeric(df["revol_util"].astype(str).str.replace("%", ""), errors="coerce")

    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0)

    #  2. FICO midpoint
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        df.drop(["fico_range_low", "fico_range_high"], axis=1, inplace=True)

    #  3. Credit history length (months since earliest credit line)
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
        ref_date = pd.Timestamp("2018-12-01")
        df["credit_history_months"] = ((ref_date - df["earliest_cr_line"]).dt.days / 30.44).fillna(0).clip(0)
        df.drop("earliest_cr_line", axis=1, inplace=True)

    #  4. Grade / sub_grade ordinal encoding
    # sub_grade A1→G5 maps to 1→35
    if "sub_grade" in df.columns:
        order = [g + str(n) for g in "ABCDEFG" for n in range(1, 6)]
        sub_grade_map = {sg: i + 1 for i, sg in enumerate(order)}
        df["sub_grade_enc"] = df["sub_grade"].map(sub_grade_map).fillna(18)
        df.drop("sub_grade", axis=1, inplace=True)

    if "grade" in df.columns:
        grade_map = {g: i + 1 for i, g in enumerate("ABCDEFG")}
        df["grade_enc"] = df["grade"].map(grade_map).fillna(4)
        df.drop("grade", axis=1, inplace=True)

    #  5. Numeric imputation (median)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            df[col] = df[col].fillna(df[col].median())

    #  6. mort_acc smart fill
    if "mort_acc" in df.columns and "total_acc" in df.columns:
        fill_map = df.groupby("total_acc")["mort_acc"].median()
        df["mort_acc"] = df.apply(
            lambda r: fill_map.get(r["total_acc"], 0) if pd.isna(r["mort_acc"]) else r["mort_acc"],
            axis=1,
        )

    #  7. Categorical label encoding
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
    le_dict: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    #  8. Core financial ratios
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_income_ratio"] = df["loan_amnt"] / (df["annual_inc"] + 1)

    if "installment" in df.columns and "annual_inc" in df.columns:
        df["installment_income_ratio"] = df["installment"] / (df["annual_inc"] / 12 + 1)

    if "installment" in df.columns and "loan_amnt" in df.columns:
        df["payment_rate"] = df["installment"] / (df["loan_amnt"] + 1)

    #  9. Credit utilisation
    if "revol_bal" in df.columns and "open_acc" in df.columns:
        df["revol_util_per_acc"] = df["revol_bal"] / (df["open_acc"] + 1)

    if "revol_bal" in df.columns and "annual_inc" in df.columns:
        df["revol_bal_income_ratio"] = df["revol_bal"] / (df["annual_inc"] + 1)

    #  10. Risk interactions
    if "int_rate" in df.columns and "dti" in df.columns:
        df["int_rate_x_dti"] = df["int_rate"] * df["dti"]

    if "int_rate" in df.columns and "term" in df.columns:
        df["int_rate_x_term"] = df["int_rate"] * df["term"]

    if "fico_score" in df.columns and "dti" in df.columns:
        df["fico_x_dti"] = df["fico_score"] * df["dti"]

    if "grade_enc" in df.columns and "int_rate" in df.columns:
        df["grade_x_int_rate"] = df["grade_enc"] * df["int_rate"]

    if "sub_grade_enc" in df.columns and "dti" in df.columns:
        df["sub_grade_x_dti"] = df["sub_grade_enc"] * df["dti"]

    #  11. Credit bureau derogatory signal
    df["derogatory_score"] = df.get("delinq_2yrs", pd.Series(0, index=df.index))
    if "pub_rec" in df.columns:
        df["derogatory_score"] += df["pub_rec"] * 2
    if "inq_last_6mths" in df.columns:
        df["derogatory_score"] += df["inq_last_6mths"] * 0.5

    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["open_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

    #  12. Log transforms (reduce skew)
    for raw, log in [
        ("loan_amnt", "log_loan_amnt"),
        ("annual_inc", "log_annual_inc"),
        ("installment", "log_installment"),
    ]:
        if raw in df.columns:
            df[log] = np.log1p(df[raw])

    if "revol_bal" in df.columns:
        df["log_revol_bal"] = np.log1p(df["revol_bal"] + 1)

    #  13. FICO risk tiers
    if "fico_score" in df.columns:
        df["fico_tier"] = (
            pd.cut(
                df["fico_score"],
                bins=[300, 580, 620, 660, 700, 740, 780, 850],
                labels=[0, 1, 2, 3, 4, 5, 6],
                ordered=True,
            )
            .astype(float)
            .fillna(2)
        )

    #  14. Composite risk score
    required_risk = [
        "sub_grade_enc",
        "dti",
        "loan_income_ratio",
        "derogatory_score",
        "fico_score",
    ]
    if all(c in df.columns for c in required_risk):
        df["composite_risk"] = (
            df["sub_grade_enc"] / 35 * 0.30
            + df["dti"].clip(0, 50) / 50 * 0.20
            + df["loan_income_ratio"].clip(0, 5) / 5 * 0.15
            + df["derogatory_score"].clip(0, 10) / 10 * 0.20
            + (1 - df["fico_score"].clip(580, 820) / 820) * 0.15
        )

    #  15. Outlier clipping
    clip_rules = {
        "dti": (0, 50),
        "annual_inc": (0, 300_000),
        "revol_bal": (0, 300_000),
        "revol_util": (0, 120),
        "open_acc": (0, 40),
        "total_acc": (0, 80),
        "loan_amnt": (0, 40_000),
        "int_rate": (0, 32),
    }
    for col, (lo, hi) in clip_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    return df, le_dict


def _clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip special chars for LightGBM compatibility."""
    df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True).str.replace("__", "_", regex=False).str.strip("_")
    return df


# Main pipeline


def preprocess(input_path: str, output_path: str) -> None:
    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE  —  v6 Feature Engineering")
    logger.info("=" * 70)
    logger.info(f"Input  : {input_path}")
    logger.info(f"Output : {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    #  Load raw CSV in chunks
    logger.info(f"Loading data in {CHUNK_SIZE:,}-row chunks …")
    chunks, total_rows = [], 0

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk = chunk[chunk["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
        chunk["target"] = (chunk["loan_status"] == "Charged Off").astype(int)
        available = [c for c in FEATURE_COLS if c in chunk.columns]
        chunk = chunk[available + ["target"]]
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info(f"  Chunk {i + 1}: {len(chunk):,} rows  (cumulative {total_rows:,})")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    logger.info(f"Dataset loaded : {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"Default rate   : {df['target'].mean():.2%}")

    target_col = "target"

    #  Feature engineering
    logger.info("Applying feature engineering …")
    df, le_dict = engineer_features(df, target_col)
    logger.info(f"After engineering : {df.shape[1]} columns")

    # Separate features / target
    X = df.drop(target_col, axis=1).select_dtypes(include=[np.number]).copy()
    y = df[target_col]

    # Ensure no residual NaNs
    if X.isnull().any().any():
        X = X.fillna(X.median())
        logger.info("Residual NaNs filled with column median")

    X = _clean_col_names(X)

    #  Optional: Yeo-Johnson power transform on highly skewed features
    skewed = [c for c in X.columns if X[c].skew() > 1.0]
    if skewed:
        pt = PowerTransformer(method="yeo-johnson")
        X[skewed] = pt.fit_transform(X[skewed])
        joblib.dump(pt, "data/processed/power_transformer.pkl")
        logger.info(f"Yeo-Johnson applied to {len(skewed)} skewed features")

    #  Mutual-information feature selection , only top 60
    k = min(60, X.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_sel = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(X_sel, columns=selected_features)
    logger.info(f"Feature selection : {k} of {X.shape[1] + (X.shape[1] - k)} kept")

    #  70 / 15 / 15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    logger.info(f"Split  train={X_train.shape[0]:,}  val={X_val.shape[0]:,}  test={X_test.shape[0]:,}")
    logger.info(f"Default rates  train={y_train.mean():.2%}  val={y_val.mean():.2%}  test={y_test.mean():.2%}")
    logger.info(f"scale_pos_weight : {pos_weight:.2f}")

    #  Save processed splits
    X_train.to_csv("data/processed/X_train_full.csv", index=False)
    y_train.to_csv("data/processed/y_train_full.csv", index=False)
    X_val.to_csv("data/processed/X_val.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_test.values, columns=["risk"]).to_csv("data/processed/y_test.csv", index=False)

    #  Save artefacts
    joblib.dump(selector, "data/processed/feature_selector.pkl")
    joblib.dump(selected_features, "data/processed/selected_features.pkl")
    joblib.dump(le_dict, "data/processed/label_encoders.pkl")

    meta = {
        "n_raw_rows": int(df.shape[0]),
        "n_features_final": len(selected_features),
        "selected_features": selected_features,
        "scale_pos_weight": pos_weight,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "default_rate_train": float(y_train.mean()),
        "default_rate_val": float(y_val.mean()),
        "default_rate_test": float(y_test.mean()),
    }
    with open("data/processed/feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Full dataset for DVC compatibility
    df_out = X.copy()
    df_out[target_col] = y.values
    df_out.to_csv(output_path, index=False)

    logger.info("All artefacts saved to data/processed/")
    logger.info(f"Preprocessed dataset : {output_path}")
    logger.info("Preprocessing complete ✓")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lending Club preprocessing pipeline")
    parser.add_argument("--input", default=None, help="Path to raw CSV")
    parser.add_argument("--output", default=None, help="Path for processed output CSV")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    input_path = args.input or params["preprocess"]["input"]
    output_path = args.output or params["preprocess"]["output"]

    preprocess(input_path, output_path)
