"""
FastAPI Application - Serve model predictions
"""
import sys
import os
from pathlib import Path
import time
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import mlflow
from datetime import datetime

#  MLflow setup 
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

# Import model registry
from src.api.model_registry import ModelRegistry

# Initialize FastAPI
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict loan default risk using Machine Learning",
    version="1.0.0"
)

# Initialize model registry
registry = ModelRegistry()

# Load model (latest version)
model = None
model_version = None

def load_model():
    """Load the latest model from registry"""
    global model, model_version
    
    model = registry.get_latest_model()
    if model is None:
        # Fallback to default model path
        model_path = Path("models/model_lending.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            model_version = "latest"
            print(f"Loaded model from {model_path}")
        else:
            print("No model found!")
    
    return model

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()
    print(" API started!")

# Request/Response models
class LoanApplication(BaseModel):
    age: float
    credit_amount: float
    duration: float
    purpose: float
    income: float
    emp_length: float
    housing: float
    dti: float = 20.0
    int_rate: float = 15.0

class PredictionResponse(BaseModel):
    prediction: int
    probability_default: float
    probability_good: float
    risk_level: str
    timestamp: str
    model_version: str
    latency_ms: float

class BatchPredictionRequest(BaseModel):
    loans: List[LoanApplication]


# API Endpoints


@app.get("/")
async def root():
    return {
        "message": "Credit Risk Prediction API",
        "status": "healthy",
        "model_loaded": model is not None,
        "endpoints": [
            "/predict",
            "/predict/batch",
            "/health",
            "/metrics",
            "/models"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(loan: LoanApplication):
    """Predict risk for a single loan application"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Convert to DataFrame
    input_data = pd.DataFrame([loan.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    latency_ms = (time.time() - start_time) * 1000
    
    risk_level = "High" if prediction == 1 else "Low"
    
    # Log to MLflow (async, don't block response)
    mlflow.set_experiment("phase_3_drifting")
    try:
        with mlflow.start_run(run_name=f"api_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("model_version", str(model_version))
            mlflow.log_metric("prediction", int(prediction))
            mlflow.log_metric("probability_default", float(probability[1]))
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_param("timestamp", datetime.now().isoformat())
    except Exception as e:
        print(f"MLflow logging warning: {e}")
    
    return PredictionResponse(
        prediction=int(prediction),
        probability_default=float(probability[1]),
        probability_good=float(probability[0]),
        risk_level=risk_level,
        timestamp=datetime.now().isoformat(),
        model_version=str(model_version),
        latency_ms=round(latency_ms, 2)
    )

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict risk for multiple loan applications"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Convert to DataFrame
    input_data = pd.DataFrame([loan.dict() for loan in request.loans])
    
    # Make predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    latency_ms = (time.time() - start_time) * 1000
    
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "loan_id": i,
            "prediction": int(pred),
            "probability_default": float(prob[1]),
            "probability_good": float(prob[0]),
            "risk_level": "High" if pred == 1 else "Low"
        })
    
    # Log batch prediction to MLflow
    mlflow.set_experiment("phase_3_drifting")
    try:
        with mlflow.start_run(run_name=f"api_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("model_version", str(model_version))
            mlflow.log_param("batch_size", len(request.loans))
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("avg_prediction", float(np.mean(predictions)))
    except Exception as e:
        print(f"MLflow logging warning: {e}")
    
    return {
        "n_predictions": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "model_version": str(model_version),
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    
    metrics_path = Path("reports/metrics_lending.json")
    if metrics_path.exists():
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    else:
        return {"error": "Metrics not found"}

@app.get("/models")
async def list_models():
    """List all registered models"""
    registry.list_models()
    return registry.models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)