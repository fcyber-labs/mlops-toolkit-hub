"""
Live Metrics Updater - Simulates real-time metrics updates
"""

import json
import time
import random
from pathlib import Path
import numpy as np


def update_live_metrics():
    """Update metrics with small random variations to simulate real-time"""

    metrics_path = Path("reports/metrics_lending.json")
    drift_path = Path("reports/drift_report.json")

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Add small random variation to simulate live data
        metrics["roc_auc"] = min(
            0.95, max(0.65, metrics.get("roc_auc", 0.70) + np.random.normal(0, 0.005))
        )
        metrics["accuracy"] = min(
            0.95, max(0.70, metrics.get("accuracy", 0.85) + np.random.normal(0, 0.003))
        )

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics
    return None


def update_drift_gradually():
    """Gradually increase drift score over time"""

    drift_path = Path("reports/drift_report.json")

    if drift_path.exists():
        with open(drift_path, "r") as f:
            drift = json.load(f)

        # Gradually increase drift
        current_score = drift.get("overall_drift_score", 0)
        new_score = min(1.0, current_score + 0.001)
        drift["overall_drift_score"] = new_score
        drift["drift_alert"] = new_score > 0.3

        with open(drift_path, "w") as f:
            json.dump(drift, f, indent=2)

        return drift
    return None


def live_updater(interval=30):
    """Run live updates every N seconds"""
    print(f"🔄 Live metrics updater started (interval: {interval}s)")

    while True:
        metrics = update_live_metrics()
        drift = update_drift_gradually()

        if metrics:
            print(f"[{time.strftime('%H:%M:%S')}] ROC-AUC: {metrics['roc_auc']:.3f}")

        time.sleep(interval)


if __name__ == "__main__":
    live_updater()
