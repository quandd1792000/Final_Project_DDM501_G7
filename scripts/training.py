# scripts/training.py
import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# Configure MLflow & MinIO connection
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "credit-risk-production-pipeline"
MODEL_NAME = os.getenv("MODEL_NAME", "credit_risk_classifier")

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


def load_and_preprocess_credit_data(save_path="data/credit_reference.csv"):
    """Load real German Credit data from OpenML and normalize to 8 numeric columns."""
    print("1. Loading real German Credit data (credit-g) from OpenML...")
    dataset = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df_raw = dataset.frame

    df = pd.DataFrame()
    df["duration"] = df_raw["duration"].astype(float)
    df["credit_amount"] = df_raw["credit_amount"].astype(float)
    df["installment_commitment"] = df_raw["installment_commitment"].astype(float)
    df["residence_since"] = df_raw["residence_since"].astype(float)
    df["age"] = df_raw["age"].astype(float)
    df["existing_credits"] = df_raw["existing_credits"].astype(float)
    df["num_dependents"] = df_raw["num_dependents"].astype(float)
    # Encode sensitive attribute: Male = 1.0, Female = 0.0
    df["is_male"] = df_raw["personal_status"].astype(str).str.contains("male single|male mar|male div").astype(float)

    # Label: 'bad' (default) = 1, 'good' = 0
    df["target"] = (df_raw["class"] == "bad").astype(int)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"   Saved normalized data at {save_path} ({len(df)} rows).")
    return df


def evaluate_fairness(X_test_df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Evaluate fairness (Responsible AI) by age and gender."""
    # High-risk evaluation rate (y_pred == 1) in young group (<30 years old) vs older group (>=30 years old)
    young_mask = X_test_df["age"] < 30
    rate_young = y_pred[young_mask].mean() if young_mask.sum() > 0 else 0.0
    rate_older = y_pred[~young_mask].mean() if (~young_mask).sum() > 0 else 0.0

    # Disparate Impact Ratio
    disparate_impact_age = (rate_older / rate_young) if rate_young > 0 else 1.0

    male_mask = X_test_df["is_male"] == 1.0
    rate_male = y_pred[male_mask].mean() if male_mask.sum() > 0 else 0.0
    rate_female = y_pred[~male_mask].mean() if (~male_mask).sum() > 0 else 0.0
    disparate_impact_gender = (rate_female / rate_male) if rate_male > 0 else 1.0

    return {
        "fairness_disparate_impact_age": round(float(disparate_impact_age), 4),
        "fairness_disparate_impact_gender": round(float(disparate_impact_gender), 4),
        "default_rate_young": round(float(rate_young), 4),
        "default_rate_older": round(float(rate_older), 4),
    }


def train_and_track():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    df = load_and_preprocess_credit_data()
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # List of experiment configurations (Multiple Experiments)
    candidate_models = [
        ("LogisticRegression_Baseline", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("RandomForest_100trees", RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)),
        ("RandomForest_200trees", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42)),
        ("GradientBoosting_Default", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)),
    ]

    best_roc_auc = -1.0
    best_run_id = None
    best_model_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n2. Starting training and tracking experiments on MLflow...")
    for run_name, estimator in candidate_models:
        with mlflow.start_run(run_name=run_name) as run:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", estimator)
            ])

            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "cv_roc_auc_mean": float(cv_scores.mean()),
                "cv_roc_auc_std": float(cv_scores.std()),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "test_recall_default": float(recall_score(y_test, y_pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
            }

            # Add Responsible AI (Fairness) metrics
            fairness_metrics = evaluate_fairness(X_test, y_pred)
            metrics.update(fairness_metrics)

            # Log parameters & metrics to MLflow
            mlflow.log_param("model_family", estimator.__class__.__name__)
            mlflow.log_params(estimator.get_params())
            mlflow.log_metrics(metrics)

            # Log model with Signature
            signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:3]
            )

            print(f"   [{run_name}] CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} | Test ROC-AUC: {metrics['test_roc_auc']:.4f} | Recall: {metrics['test_recall_default']:.4f}")

            if metrics["test_roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["test_roc_auc"]
                best_run_id = run.info.run_id
                best_model_name = run_name

    # Register winning model to Model Registry & transition to Production
    print(f"\n3. Winning model: {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"   Transitioned {MODEL_NAME} (version {mv.version}) to 'Production' stage!")


if __name__ == "__main__":
    train_and_track()