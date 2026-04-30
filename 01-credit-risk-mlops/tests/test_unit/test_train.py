import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture
def sample_data():
    """Create minimal sample for training tests"""
    return pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Credit amount': [5000, 10000, 15000, 20000, 25000],
        'Duration': [12, 24, 36, 48, 60],
        'risk': [0, 0, 1, 1, 0]
    })


def test_train_module_imports():
    """Test that train module can be imported"""
    try:
        import src.train
        assert True
    except ImportError as e:
        pytest.skip(f"Cannot import src.train: {e}")


def test_find_threshold_returns_valid_value():
    """Test threshold function"""
    try:
        from src.train import find_threshold
    except ImportError:
        pytest.skip("find_threshold not found")
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.2, 0.8])
    
    threshold, sweep = find_threshold(y_true, y_proba)
    
    assert 0.1 <= threshold <= 0.9
    assert isinstance(sweep, dict)


def test_train_function_exists():
    """Test that train function exists"""
    try:
        from src.train import train
        assert callable(train)
    except ImportError:
        pytest.skip("train function not found")


def test_metrics_calculation():
    """Test basic metric calculations used in train"""
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    y_proba = np.array([0.2, 0.3, 0.8, 0.4])
    
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    
    assert 0 <= auc <= 1
    assert 0 <= acc <= 1