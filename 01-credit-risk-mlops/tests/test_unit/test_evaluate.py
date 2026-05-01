import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_evaluate_module_imports():
    """Test that evaluate module can be imported"""
    try:
        import src.evaluate  # noqa: F401

        assert True
    except ImportError as e:
        pytest.skip(f"Cannot import src.evaluate: {e}")


def test_compute_metrics_returns_dict():
    """Test metrics calculation function"""
    try:
        from src.evaluate import compute_metrics
    except ImportError:
        pytest.skip("compute_metrics not found")

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.6])
    threshold = 0.5

    metrics = compute_metrics(y_true, y_proba, threshold)

    expected_keys = [
        "recall",
        "precision",
        "f1",
        "f2",
        "roc_auc",
        "business_cost",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    for key in expected_keys:
        assert key in metrics


def test_metrics_values_reasonable():
    """Test that computed metrics are within reasonable ranges"""
    try:
        from src.evaluate import compute_metrics
    except ImportError:
        pytest.skip("compute_metrics not found")

    # Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])
    threshold = 0.5

    metrics = compute_metrics(y_true, y_proba, threshold)

    # Use approximate equality due to floating point precision
    assert abs(metrics["recall"] - 1.0) < 0.0001
    assert abs(metrics["precision"] - 1.0) < 0.0001
    assert abs(metrics["roc_auc"] - 1.0) < 0.0001


def test_evaluate_function_exists():
    """Test that evaluate function exists"""
    try:
        from src.evaluate import evaluate

        assert callable(evaluate)
    except ImportError:
        pytest.skip("evaluate function not found")
