# tests/test_system.py
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.main import app, apply_credit_guardrails, FEATURE_NAMES
from scripts.training import evaluate_fairness

client = TestClient(app)


# 1. UNIT TESTS: Kiểm tra logic Guardrails
class TestUnitGuardrails:
    def test_valid_applicant_passes(self):
        valid_features = [24.0, 3500.0, 3.0, 2.0, 35.0, 1.0, 1.0, 1.0]
        assert apply_credit_guardrails(valid_features) is None

    def test_underage_applicant_rejected(self):
        underage = [24.0, 3500.0, 3.0, 2.0, 16.0, 1.0, 1.0, 1.0]
        error = apply_credit_guardrails(underage)
        assert error is not None and "18" in error

    def test_negative_loan_amount_rejected(self):
        negative_loan = [24.0, -500.0, 3.0, 2.0, 30.0, 1.0, 1.0, 1.0]
        assert apply_credit_guardrails(negative_loan) is not None


# 2. DATA QUALITY TESTS: Kiểm tra tính toàn vẹn của Schema và miền giá trị
class TestDataQuality:
    def test_feature_schema_completeness(self):
        assert len(FEATURE_NAMES) == 8
        assert "age" in FEATURE_NAMES and "credit_amount" in FEATURE_NAMES

    def test_synthetic_data_has_no_nulls(self):
        from simulations.run_simulation import generate_sample
        f_list, f_dict = generate_sample(drift_magnitude=0.0)
        assert len(f_list) == 8
        assert not any(np.isnan(f_list))
        assert 18.0 <= f_dict["age"] <= 100.0


# 3. MODEL VALIDATION & FAIRNESS TESTS: Kiểm tra chất lượng mô hình & tính công bằng
class TestModelValidation:
    def test_model_pipeline_sanity_and_fairness(self):
        np.random.seed(42)
        X_dummy = pd.DataFrame({
            "duration": np.random.randint(6, 60, 100).astype(float),
            "credit_amount": np.random.uniform(500, 20000, 100),
            "installment_commitment": np.random.randint(1, 4, 100).astype(float),
            "residence_since": np.random.randint(1, 4, 100).astype(float),
            "age": np.random.randint(20, 65, 100).astype(float),
            "existing_credits": np.random.randint(1, 3, 100).astype(float),
            "num_dependents": np.random.randint(1, 2, 100).astype(float),
            "is_male": np.random.choice([0.0, 1.0], 100),
        })
        y_dummy = (X_dummy["credit_amount"] > 10000).astype(int)

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X_dummy, y_dummy)
        preds = pipe.predict(X_dummy)

        # Kiểm tra độ chính xác tối thiểu (Sanity check - Rule #9 of ML)
        assert (preds == y_dummy).mean() > 0.70

        # Kiểm tra hàm đánh giá Fairness hoạt động đúng
        fairness = evaluate_fairness(X_dummy, preds)
        assert "fairness_disparate_impact_age" in fairness


# 4. INTEGRATION TESTS: Kiểm tra các API Endpoints
class TestAPIIntegration:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_metrics_endpoint_exposes_prometheus(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "ml_predictions_total" in response.text

    def test_guardrail_blocks_invalid_request_via_api(self):
        bad_payload = {"features": [12.0, -100.0, 2.0, 2.0, 15.0, 1.0, 1.0, 1.0]}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422