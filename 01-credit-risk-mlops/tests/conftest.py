import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



@pytest.fixture
def sample_data():
    """Create small sample dataset for testing"""
    return pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Credit amount': [5000, 10000, 15000, 20000, 25000],
        'Duration': [12, 24, 36, 48, 60],
        'int_rate': [10.5, 12.0, 15.5, 18.0, 20.0],
        'dti': [10, 15, 20, 25, 30],
        'purpose': ['car', 'education', 'car', 'business', 'education'],
        'housing': ['own', 'rent', 'own', 'rent', 'own'],
        'income': [30000, 40000, 50000, 60000, 70000],
        'emp_length': [1, 2, 3, 4, 5],
        'Risk': ['good', 'good', 'bad', 'bad', 'good']
    })

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_model():
    """Simple mock model for testing"""
    class MockModel:
        def predict(self, X):
            return np.array([0, 1, 0])
        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        def feature_importances_(self):
            return np.array([0.5, 0.3, 0.2])
    return MockModel()

