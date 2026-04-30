"""
Model Registry - Version and manage models
"""

import joblib
import json
from pathlib import Path
from datetime import datetime


class ModelRegistry:
    """
    Simple model registry for versioning and model management
    """

    def __init__(self, registry_path="models/registry.json"):
        self.registry_path = Path(registry_path)
        self.models = self._load_registry()

    def _load_registry(self):
        """Load existing registry"""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": [], "latest": None}

    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_path, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model(self, model_path, metrics, version=None):
        """Register a new model version"""

        if version is None:
            version = len(self.models["models"]) + 1

        model_info = {
            "version": version,
            "path": str(model_path),
            "metrics": metrics,
            "registered_at": datetime.now().isoformat(),
            "status": "active",
        }

        self.models["models"].append(model_info)
        self.models["latest"] = version

        self._save_registry()
        print(f"Model version {version} registered")

        return version

    def get_latest_model(self):
        """Get the latest model"""
        if self.models["latest"] is None:
            return None

        for model in self.models["models"]:
            if model["version"] == self.models["latest"]:
                return joblib.load(model["path"])

        return None

    def get_model_by_version(self, version):
        """Get a specific model version"""
        for model in self.models["models"]:
            if model["version"] == version:
                return joblib.load(model["path"])
        return None

    def list_models(self):
        """List all registered models"""
        print("\n" + "=" * 60)
        print("MODEL REGISTRY")
        print("=" * 60)
        for model in self.models["models"]:
            print(f"Version {model['version']}: {model['registered_at']}")
            print(f"  Path: {model['path']}")
            print(f"  ROC-AUC: {model['metrics'].get('roc_auc', 'N/A')}")
            print(f"  Accuracy: {model['metrics'].get('accuracy', 'N/A')}")
        print("=" * 60)
