# api/main.py
import os
import time
import logging
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-risk-api")

FEATURE_NAMES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
    "is_male"
]

CLASS_NAMES = ["good_credit", "bad_credit_risk"]

# Prometheus Metrics (Similar to sample project structure to match Grafana)
PREDICTION_COUNTER = Counter(
    "ml_predictions_total", "Total number of credit predictions", ["predicted_class", "model_version"]
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_duration_seconds", "Prediction latency in seconds"
)
CONFIDENCE_HISTOGRAM = Histogram(
    "ml_prediction_confidence", "Prediction confidence distribution", buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)
API_ERROR_COUNTER = Counter(
    "ml_api_errors_total", "Total API & Guardrail errors", ["error_type"]
)
MODEL_LOADED_GAUGE = Gauge(
    "ml_model_loaded", "Whether the production model is loaded (1=yes, 0=no)"
)

app = FastAPI(
    title="Credit Risk Assessment MLOps API",
    description="Production-ready ML Service with Guardrails, Prometheus Monitoring, and Responsible AI Explainability",
    version="1.0.0"
)

# Global variable holding the model
model_state = {
    "model": None,
    "version": "unloaded",
    "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "model_name": os.getenv("MODEL_NAME", "credit_risk_classifier"),
    "model_stage": os.getenv("MODEL_STAGE", "Production")
}


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="[duration, credit_amount, installment_commitment, residence_since, age, existing_credits, num_dependents, is_male]"
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    risk_probability: float
    confidence: float
    action_recommendation: str
    model_version: str
    latency_ms: float


def apply_credit_guardrails(features: List[float]) -> Optional[str]:
    """Safety Guardrails (Lecture S02): Check hard business rules before calling ML."""
    duration, credit_amount, installment, residence, age, existing_credits, dependents, is_male = features
    if age < 18 or age > 100:
        return "Guardrail Violation: Customer must be between 18 and 100 years old."
    if credit_amount <= 0 or credit_amount > 500000:
        return "Guardrail Violation: Invalid loan amount (must be > 0 and <= 500,000)."
    if duration < 1 or duration > 120:
        return "Guardrail Violation: Loan duration must be between 1 and 120 months."
    return None


def load_production_model():
    """Load model from MLflow Model Registry."""
    try:
        mlflow.set_tracking_uri(model_state["mlflow_uri"])
        model_uri = f"models:/{model_state['model_name']}/{model_state['model_stage']}"
        model_state["model"] = mlflow.pyfunc.load_model(model_uri)
        model_state["version"] = model_state["model_stage"]
        MODEL_LOADED_GAUGE.set(1)
        logger.info(f"Successfully loaded model {model_uri}")
    except Exception as e:
        MODEL_LOADED_GAUGE.set(0)
        logger.warning(f"Failed to load model from MLflow ({e}). You can run /reload-model after training.")


@app.on_event("startup")
async def startup_event():
    load_production_model()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_state["model"] is not None,
        "model_version": model_state["version"]
    }


@app.post("/reload-model")
async def reload_model():
    load_production_model()
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Failed to load model from MLflow.")
    return {"status": "reloaded", "model_version": model_state["version"]}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()

    # 1. Check Guardrails
    guardrail_error = apply_credit_guardrails(request.features)
    if guardrail_error:
        API_ERROR_COUNTER.labels(error_type="guardrail_rejected").inc()
        raise HTTPException(status_code=422, detail=guardrail_error)

    if model_state["model"] is None:
        API_ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not ready.")

    try:
        input_df = pd.DataFrame([request.features], columns=FEATURE_NAMES)
        # Access pipeline underneath pyfunc to get probability
        raw_model = model_state["model"]._model_impl.sklearn_model
        probs = raw_model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(probs))
        risk_prob = float(probs[1])
        confidence = float(np.max(probs))

        # User experience design according to Forcefulness Spectrum (S02)
        if risk_prob >= 0.75:
            action = "REJECT (Automatically rejected due to very high default risk)"
        elif risk_prob >= 0.40:
            action = "HUMAN_REVIEW (Escalate to credit officer for further review)"
        else:
            action = "APPROVE (Eligible for automatic loan approval)"

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        CONFIDENCE_HISTOGRAM.observe(confidence)
        PREDICTION_COUNTER.labels(
            predicted_class=CLASS_NAMES[pred_class],
            model_version=str(model_state["version"])
        ).inc()

        return PredictionResponse(
            prediction=pred_class,
            class_name=CLASS_NAMES[pred_class],
            risk_probability=round(risk_prob, 4),
            confidence=round(confidence, 4),
            action_recommendation=action,
            model_version=str(model_state["version"]),
            latency_ms=round(latency * 1000, 2)
        )
    except Exception as e:
        API_ERROR_COUNTER.labels(error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_prediction(request: PredictionRequest) -> Dict:
    """Responsible AI Endpoint: Explain the contribution of each feature (Feature Attribution)."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    input_df = pd.DataFrame([request.features], columns=FEATURE_NAMES)
    raw_pipeline = model_state["model"]._model_impl.sklearn_model
    scaler = raw_pipeline.named_steps["scaler"]
    classifier = raw_pipeline.named_steps["classifier"]

    scaled_x = scaler.transform(input_df)[0]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)

    # Local contribution score
    contributions = {
        FEATURE_NAMES[i]: round(float(scaled_x[i] * importances[i]), 4)
        for i in range(len(FEATURE_NAMES))
    }
    sorted_contrib = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))

    return {
        "feature_contributions": sorted_contrib,
        "top_risk_driver": next(iter(sorted_contrib)),
        "explanation_method": "Local Feature Attribution (SHAP/Importance-weighted)"
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)