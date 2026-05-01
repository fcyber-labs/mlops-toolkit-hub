"""
Auto-Retraining - Automatically retrain model when drift is detected
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import subprocess
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from drift_detector import DriftDetector

#  MLflow setup
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))


class AutoRetrainer:
    """
    Monitor data drift and trigger retraining when needed
    """

    def __init__(
        self,
        model_path="models/model_lending.pkl",
        config_path="params_lending.yaml",
        drift_threshold=0.3,
        min_samples=1000,
    ):

        self.model_path = Path(model_path)
        self.config_path = config_path
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.detector = DriftDetector(threshold=0.05)

    def load_reference_data(self, reference_path="data/processed/features_lending.csv"):
        """Load reference training data"""
        if Path(reference_path).exists():
            self.reference_data = pd.read_csv(reference_path)
            self.detector.set_reference(self.reference_data)
            print(f"✅ Reference data loaded: {self.reference_data.shape}")
        else:
            print(f"⚠️ Reference data not found at {reference_path}")

    def check_and_retrain(self, new_data_path, force=False):
        """
        Check drift on new data and retrain if needed
        """
        print(f"\n{'=' * 60}")
        print(f"Auto-Retraining Check at {datetime.now()}")
        print(f"{'=' * 60}")

        # Load new data
        new_data = pd.read_csv(new_data_path)
        print(f"New data shape: {new_data.shape}")

        if len(new_data) < self.min_samples:
            print(f"⚠️ Insufficient samples: {len(new_data)} < {self.min_samples}")
            return False

        # Detect drift
        report = self.detector.detect_all_drift(new_data)
        self.detector.print_report()
        self.detector.save_report()

        # Check if retraining is needed
        should_retrain = force or report["drift_alert"]

        if should_retrain:
            print("\n🚨 DRIFT DETECTED! Starting auto-retraining...")
            self.retrain_model()
            return True
        else:
            print("\n✅ No significant drift detected. Retraining not needed.")
            return False

    def retrain_model(self):
        """Trigger model retraining"""
        print("\n🏋️ Starting model retraining...")

        # Log retraining event
        mlflow.set_experiment("phase_3_drifting")
        with mlflow.start_run(run_name=f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("retrain_reason", "data_drift")
            mlflow.log_param("timestamp", datetime.now().isoformat())
            mlflow.log_param("drift_threshold", self.drift_threshold)
            mlflow.log_metric("drift_score", self.detector.drift_report.get("overall_drift_score", 0))
            mlflow.log_metric(
                "features_with_drift",
                self.detector.drift_report.get("features_with_drift", 0),
            )

            # Log drift report as artifact
            if hasattr(self.detector, "drift_report") and self.detector.drift_report:
                with open("/tmp/drift_report_retrain.json", "w") as f:
                    json.dump(self.detector.drift_report, f, indent=2, default=str)
                mlflow.log_artifact("/tmp/drift_report_retrain.json")

        # Run training pipeline
        result = subprocess.run(["python", "src/train.py"], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Auto-retraining completed successfully!")
            self._log_retraining_success()

            # Log retraining success to MLflow
            mlflow.set_experiment("phase_3_drifting")
            with mlflow.start_run(run_name=f"retrain_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("status", "success")
                mlflow.log_param("model_path", str(self.model_path))
                mlflow.log_metric("retraining_duration_seconds", 0)  # Could calculate actual duration
        else:
            print("❌ Auto-retraining failed!")
            print(result.stderr)
            self._log_retraining_failure(result.stderr)

            # Log retraining failure to MLflow
            mlflow.set_experiment("phase_3_drifting")
            with mlflow.start_run(run_name=f"retrain_failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", result.stderr[:500])  # Log first 500 chars
                mlflow.log_artifact("/tmp/retrain_error.log")

        return result.returncode == 0

    def _log_retraining_success(self):
        """Log successful retraining"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "drift_score": self.detector.drift_report.get("overall_drift_score", 0),
            "model_path": str(self.model_path),
        }

        log_path = Path("reports/retraining_log.json")
        if log_path.exists():
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            log = []

        log.append(report)

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    def _log_retraining_failure(self, error):
        """Log failed retraining"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": error[:500],
            "drift_score": self.detector.drift_report.get("overall_drift_score", 0),
        }

        log_path = Path("reports/retraining_log.json")
        if log_path.exists():
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            log = []

        log.append(report)

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)


if __name__ == "__main__":
    # Test auto-retraining
    retrainer = AutoRetrainer()
    retrainer.load_reference_data()

    # Check and retrain on new streaming data
    retrainer.check_and_retrain("data/streaming/all_loans.csv", force=False)
