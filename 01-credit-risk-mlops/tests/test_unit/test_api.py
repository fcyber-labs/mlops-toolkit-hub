from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.app import app

client = TestClient(app)


def test_health_endpoint_returns_healthy():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint_returns_info():
    """Test root endpoint returns API information"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()


def test_predict_endpoint_accepts_valid_input():
    """Test prediction endpoint with valid input"""
    payload = {
        "age": 35,
        "credit_amount": 15000,
        "duration": 36,
        "purpose": 2,
        "income": 50000,
        "emp_length": 5,
        "housing": 1,
        "dti": 20,
        "int_rate": 15,
    }
    response = client.post("/predict", json=payload)

    # If model not loaded, returns 503
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert "prediction" in response.json()
        assert "probability_default" in response.json()


def test_predict_batch_endpoint_accepts_list():
    """Test batch prediction endpoint"""
    payload = {
        "loans": [
            {
                "age": 30,
                "credit_amount": 10000,
                "duration": 24,
                "purpose": 1,
                "income": 40000,
                "emp_length": 3,
                "housing": 2,
                "dti": 15,
                "int_rate": 12,
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code in [200, 503]
