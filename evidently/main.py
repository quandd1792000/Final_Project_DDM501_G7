# evidently/main.py
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

FEATURE_NAMES = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents", "is_male"
]

app = FastAPI(title="Credit Drift Monitoring Service (Evidently/KS Engine)")

DATASET_DRIFT_GAUGE = Gauge("evidently_dataset_drift_detected", "1 if dataset drift is detected, 0 otherwise")
DRIFT_SHARE_GAUGE = Gauge("evidently_share_of_drifted_columns", "Share of drifted features")
COLUMN_DRIFT_GAUGE = Gauge("evidently_column_drift_score", "KS-test p-value / drift score per feature", ["column_name"])
SAMPLES_ANALYZED = Counter("evidently_samples_analyzed_total", "Total samples analyzed for drift")

# Initialize Reference Data and Current Window buffer
reference_df: pd.DataFrame = pd.DataFrame()
current_buffer: List[Dict[str, float]] = []
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "50"))


class MonitoringSample(BaseModel):
    features: Dict[str, float]
    prediction: int
    confidence: float


@app.on_event("startup")
def load_reference():
    global reference_df
    ref_path = "data/credit_reference.csv"
    if os.path.exists(ref_path):
        reference_df = pd.read_csv(ref_path)[FEATURE_NAMES]
    else:
        # Initialize default normal distribution if CSV file does not exist
        np.random.seed(42)
        reference_df = pd.DataFrame({
            "duration": np.random.normal(20.9, 12.0, 500),
            "credit_amount": np.random.normal(3271.0, 2822.0, 500),
            "installment_commitment": np.random.normal(2.97, 1.11, 500),
            "residence_since": np.random.normal(2.84, 1.10, 500),
            "age": np.random.normal(35.5, 11.3, 500),
            "existing_credits": np.random.normal(1.40, 0.57, 500),
            "num_dependents": np.random.normal(1.15, 0.36, 500),
            "is_male": np.random.binomial(1, 0.69, 500).astype(float),
        })


@app.post("/iterate")
def collect_and_evaluate(sample: MonitoringSample):
    current_buffer.append(sample.features)
    SAMPLES_ANALYZED.inc()

    if len(current_buffer) >= WINDOW_SIZE:
        current_df = pd.DataFrame(current_buffer[-WINDOW_SIZE:])
        drifted_cols = 0
        for col in FEATURE_NAMES:
            if col in current_df.columns and col in reference_df.columns:
                stat, p_value = ks_2samp(reference_df[col], current_df[col])
                # Save KS statistical deviation (closer to 1.0 means stronger deviation)
                COLUMN_DRIFT_GAUGE.labels(column_name=col).set(float(stat))
                if p_value < 0.05:
                    drifted_cols += 1

        drift_share = drifted_cols / len(FEATURE_NAMES)
        DRIFT_SHARE_GAUGE.set(drift_share)
        DATASET_DRIFT_GAUGE.set(1 if drift_share >= 0.5 else 0)
        return {"status": "evaluated", "drift_share": drift_share}

    return {"status": "buffered", "buffer_size": len(current_buffer)}


@app.get("/health")
def health():
    return {"status": "healthy", "buffer_size": len(current_buffer)}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)