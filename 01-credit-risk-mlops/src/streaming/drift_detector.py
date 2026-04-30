"""
Drift Detection - Supports BOTH Phase 1 and Phase 2 with their own streaming data
"""

import os
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import json
from pathlib import Path
from datetime import datetime
import mlflow
from dotenv import load_dotenv

#  MLflow setup
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))


class DriftDetector:
    def __init__(self, reference_data=None, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_report = {}

    def set_reference(self, reference_data):
        self.reference_data = reference_data
        print(f"Reference data set with {len(reference_data)} samples")

    def detect_numerical_drift(self, col_name, current_values):
        """Use Kolmogorov-Smirnov test for numerical features"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        ref_values = pd.to_numeric(self.reference_data[col_name], errors="coerce").dropna()
        current_values = pd.to_numeric(current_values, errors="coerce").dropna()

        if len(ref_values) < 2 or len(current_values) < 2:
            return {"feature": col_name, "type": "numerical", "drift_detected": False, "p_value": 1.0}

        try:
            statistic, p_value = ks_2samp(ref_values, current_values)
        except Exception:
            p_value = 1.0
            statistic = 0

        return {
            "feature": col_name,
            "type": "numerical",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < self.threshold,
        }

    def detect_categorical_drift(self, col_name, current_values):
        """Use Chi-square test for categorical features"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        ref_values = self.reference_data[col_name].astype(str).fillna("missing")
        current_values = current_values.astype(str).fillna("missing")

        ref_counts = ref_values.value_counts()
        current_counts = current_values.value_counts()

        all_categories = set(ref_counts.index) | set(current_counts.index)

        ref_dist = [ref_counts.get(cat, 0) for cat in all_categories]
        current_dist = [current_counts.get(cat, 0) for cat in all_categories]

        try:
            chi2, p_value, dof, expected = chi2_contingency([ref_dist, current_dist])
        except Exception:
            p_value = 1.0
            chi2 = 0

        return {
            "feature": col_name,
            "type": "categorical",
            "statistic": float(chi2),
            "p_value": float(p_value),
            "drift_detected": p_value < self.threshold,
        }

    def detect_all_drift(self, current_data, model_name="lending"):
        """Detect drift for all features"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        results = []
        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))

        # Filter out non-feature columns
        exclude_cols = ["day", "timestamp", "risk", "target", "Unnamed: 0"]
        common_cols = [col for col in common_cols if col not in exclude_cols]

        print(f"Common columns for drift detection: {common_cols[:5]}... (total: {len(common_cols)})")

        for col in common_cols:
            if self.reference_data[col].dtype in ["float64", "int64"]:
                result = self.detect_numerical_drift(col, current_data[col])
            else:
                result = self.detect_categorical_drift(col, current_data[col])
            results.append(result)

        drift_count = sum(1 for r in results if r["drift_detected"])
        total_features = len(results)
        overall_drift_score = drift_count / total_features if total_features > 0 else 0

        self.drift_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_name": model_name,
            "total_features": total_features,
            "features_with_drift": drift_count,
            "overall_drift_score": overall_drift_score,
            "drift_alert": overall_drift_score > 0.3,
            "results": results,
        }

        return self.drift_report

    def print_report(self):
        if not self.drift_report:
            print("No drift report available.")
            return

        print("\n" + "=" * 60)
        print(f"DRIFT DETECTION REPORT - {self.drift_report.get('model_name', 'Unknown').upper()}")
        print("=" * 60)
        print(f"Total features analyzed: {self.drift_report['total_features']}")
        print(f"Features with drift: {self.drift_report['features_with_drift']}")
        print(f"Overall drift score: {self.drift_report['overall_drift_score']:.2%}")
        print(f"DRIFT ALERT: {'⚠️ YES' if self.drift_report['drift_alert'] else '✅ NO'}")

    def save_report(self, output_path="reports/drift_report.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.drift_report, f, indent=2, default=str)
        print(f"✅ Drift report saved to {output_path}")


def detect_drift_for_both_models():
    """Run drift detection for both Phase 1 and Phase 2 with their own streaming data"""
    from data_generator_german import GermanCreditDataGenerator
    from data_generator import LoanDataGenerator

    print("=" * 60)
    print("DRIFT DETECTION FOR BOTH MODELS")
    print("=" * 60)

    # Phase 1: German Credit

    print("\n" + "-" * 40)
    print("Phase 1: German Credit")
    print("-" * 40)

    # Check if German streaming data exists, generate if not
    german_stream_path = Path("data/streaming/german/all_loans.csv")
    if not german_stream_path.exists():
        print("⚠️ No German streaming data found. Generating sample data...")
        generator_german = GermanCreditDataGenerator()
        generator_german.generate_stream(days=5, loans_per_day=10)

    current_data_german = pd.read_csv(german_stream_path)
    print(f"Streaming data: {len(current_data_german)} rows")
    print(f"Streaming columns: {list(current_data_german.columns)[:5]}...")

    # Load Phase 1 reference data
    ref_path_german = Path("data/processed/features_german.csv")
    if ref_path_german.exists():
        ref_data_german = pd.read_csv(ref_path_german)
        print(f"Reference data: {len(ref_data_german)} rows")
        print(f"Reference columns: {list(ref_data_german.columns)[:5]}...")

        # Find common columns
        common_cols = list(set(ref_data_german.columns) & set(current_data_german.columns))
        print(f"Common columns: {len(common_cols)}")

        if len(common_cols) > 0:
            detector_german = DriftDetector(threshold=0.05)
            detector_german.set_reference(ref_data_german)
            report_german = detector_german.detect_all_drift(
                current_data_german[common_cols], model_name="german_credit"
            )
            detector_german.print_report()
            detector_german.save_report("reports/drift_report_german.json")

            # Log to MLflow
            mlflow.set_experiment("phase_3_drifting")
            with mlflow.start_run(run_name=f"drift_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("phase", "phase1")
                mlflow.log_param("model_name", "german_credit")
                mlflow.log_metric("total_features", report_german["total_features"])
                mlflow.log_metric("features_with_drift", report_german["features_with_drift"])
                mlflow.log_metric("overall_drift_score", report_german["overall_drift_score"])
                mlflow.log_metric("drift_alert", 1 if report_german["drift_alert"] else 0)
                mlflow.log_artifact("reports/drift_report_german.json")
                mlflow.log_artifact(german_stream_path)
        else:
            print("⚠️ No common columns found for drift detection")
            print("   Phase 1 and streaming data have different schemas")
    else:
        print("⚠️ No reference data found for Phase 1. Run Phase 1 first.")

    # Phase 2: LendingClub

    print("\n" + "-" * 40)
    print("Phase 2: LendingClub")
    print("-" * 40)

    # Check if LendingClub streaming data exists
    lending_stream_path = Path("data/streaming/all_loans.csv")
    if not lending_stream_path.exists():
        print("⚠️ No LendingClub streaming data found. Generating sample data...")
        generator_lending = LoanDataGenerator()
        generator_lending.generate_stream(days=5, loans_per_day=10)

    current_data_lending = pd.read_csv(lending_stream_path)
    print(f"Streaming data: {len(current_data_lending)} rows")
    print(f"Streaming columns: {list(current_data_lending.columns)[:5]}...")

    # Load Phase 2 reference data
    ref_path_lending = Path("data/processed/features_lending.csv")
    if ref_path_lending.exists():
        ref_data_lending = pd.read_csv(ref_path_lending)
        print(f"Reference data: {len(ref_data_lending)} rows")
        print(f"Reference columns: {list(ref_data_lending.columns)[:5]}...")

        # Find common columns
        common_cols = list(set(ref_data_lending.columns) & set(current_data_lending.columns))
        print(f"Common columns: {len(common_cols)}")

        if len(common_cols) > 0:
            detector_lending = DriftDetector(threshold=0.05)
            detector_lending.set_reference(ref_data_lending)
            report_lending = detector_lending.detect_all_drift(
                current_data_lending[common_cols], model_name="lendingclub"
            )
            detector_lending.print_report()
            detector_lending.save_report("reports/drift_report.json")

            # Log to MLflow
            mlflow.set_experiment("phase_3_drifting")
            with mlflow.start_run(run_name=f"drift_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("phase", "phase2")
                mlflow.log_param("model_name", "lendingclub")
                mlflow.log_metric("total_features", report_lending["total_features"])
                mlflow.log_metric("features_with_drift", report_lending["features_with_drift"])
                mlflow.log_metric("overall_drift_score", report_lending["overall_drift_score"])
                mlflow.log_metric("drift_alert", 1 if report_lending["drift_alert"] else 0)
                mlflow.log_artifact("reports/drift_report.json")
                mlflow.log_artifact(lending_stream_path)

                # Log individual feature drift metrics
                for result in report_lending["results"]:
                    if result["drift_detected"]:
                        mlflow.log_metric(f"drift_{result['feature']}", 1)
                        mlflow.log_metric(f"p_value_{result['feature']}", result["p_value"])
        else:
            print("⚠️ No common columns found for drift detection")
    else:
        print("⚠️ No reference data found for Phase 2. Run Phase 2 first.")

    print("\n✅ Drift detection complete for both models!")


if __name__ == "__main__":
    detect_drift_for_both_models()
