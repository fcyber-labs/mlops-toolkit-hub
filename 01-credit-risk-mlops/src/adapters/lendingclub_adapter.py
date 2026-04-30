import pandas as pd
import numpy as np
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")


def adapt_lendingclub(
    accepted_path="data/raw/lendingclub/accepted_2007_to_2018Q4.csv", output_path="data/raw/pipeline_input.csv"
):
    """
    Convert LendingClub accepted loans to pipeline format
    """
    print("=" * 60)
    print("LendingClub Adapter")
    print("=" * 60)

    # Load data

    print("\nLoading accepted loans...")
    df = pd.read_csv(accepted_path, low_memory=False)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Create target variable

    print("\nCreating target variable...")

    # Map loan_status to binary risk: 1 = Bad ; 0 = Good

    bad_statuses = ["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"]
    good_statuses = ["Fully Paid", "Current"]

    # Filter to known statuses
    df = df[df["loan_status"].isin(bad_statuses + good_statuses)]
    print(f"After filtering: {len(df):,} rows")

    df["risk"] = df["loan_status"].apply(lambda x: 1 if x in bad_statuses else 0)

    print("Target distribution:")
    print(f"  Good (0): {(df['risk'] == 0).sum():,} ({(df['risk'] == 0).mean():.1%})")
    print(f"  Bad (1):  {(df['risk'] == 1).sum():,} ({(df['risk'] == 1).mean():.1%})")

    # Feature engineering

    print("\nFeature engineering...")

    # Credit amount
    df["credit_amount"] = np.log1p(df["loan_amnt"])

    # Duration
    df["duration"] = df["term"].str.extract(r"(\d+)").astype(float)

    # Age from earliest credit line
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    df["age"] = 2024 - df["earliest_cr_line"].dt.year
    df["age"] = df["age"].clip(18, 80).fillna(40)

    # Income
    df["income"] = np.log1p(df["annual_inc"].fillna(50000))

    # Employment length to numeric
    def emp_to_num(x):
        if pd.isna(x):
            return 5
        if "< 1" in str(x):
            return 0.5
        if "10+" in str(x):
            return 15
        nums = "".join(filter(str.isdigit, str(x)))
        return float(nums) if nums else 5

    df["emp_length"] = df["emp_length"].apply(emp_to_num)

    # Housing
    housing_map = {"RENT": "rent", "MORTGAGE": "mortgage", "OWN": "own", "ANY": "other", "NONE": "other"}
    df["housing"] = df["home_ownership"].map(housing_map).fillna("other")

    # Purpose
    purpose_freq = df["purpose"].value_counts(normalize=True).to_dict()
    df["purpose"] = df["purpose"].map(purpose_freq).fillna(0)

    # debt-to-income
    df["dti"] = df["dti"].fillna(df["dti"].median())

    # Interest rate
    df["int_rate"] = df["int_rate"].fillna(df["int_rate"].median())

    # Select final columns

    final_columns = [
        "risk",
        "age",
        "credit_amount",
        "duration",
        "purpose",
        "income",
        "emp_length",
        "housing",
        "dti",
        "int_rate",
    ]

    df_final = df[final_columns].copy()
    df_final = df_final.dropna()

    print(f"\nFinal dataset: {df_final.shape}")
    print(f"Default rate: {df_final['risk'].mean():.2%}")
    print(f"Columns: {list(df_final.columns)}")

    # Save

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"   Shape: {df_final.shape}")

    return df_final


if __name__ == "__main__":
    adapt_lendingclub()
