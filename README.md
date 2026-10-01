# Credit Risk Classifier - MLOps Pipeline (Final Project DDM501 G7)

[![MLOps CI/CD Pipeline](https://github.com/quandd1792000/Final_Project_DDM501_G7/actions/workflows/ci.yml/badge.svg)](https://github.com/quandd1792000/Final_Project_DDM501_G7/actions/workflows/ci.yml)

An end-to-end Machine Learning Operations (MLOps) architecture designed to train, deploy, and monitor a Credit Risk Classification model. This repository contains the complete infrastructure code, APIs, and model monitoring dashboards.

## 🚀 Features

* **Model Serving (FastAPI)**: Robust REST API to serve predictions in real-time, fortified with input guardrails.
* **Experiment Tracking (MLflow)**: Full integration with MLflow for tracking parameters, metrics, and managing the model lifecycle (Backend: PostgreSQL, Artifacts: MinIO S3).
* **Data Drift Monitoring (Evidently AI)**: Automated detection of data drift and data quality degradation.
* **System Observability (Prometheus & Grafana)**: Interactive dashboards for monitoring real-time API performance, model metrics, and system health.
* **CI/CD Automation (GitHub Actions)**: Automated code linting (`flake8`) and a comprehensive testing suite (`pytest`) ensuring model sanity and data fairness.

## 📁 Project Structure

```bash
.
├── .github/workflows/   # CI/CD pipelines (GitHub Actions)
├── api/                 # FastAPI service for the prediction endpoints
├── config/              # Prometheus, Grafana, and AlertManager configurations
├── evidently/           # Data Drift tracking and ML monitoring logic
├── scripts/             # Model training and deployment scripts
├── simulations/         # Synthetic data generation for testing
├── tests/               # Unit, data quality, integration, and fairness tests
├── docker-compose.yml   # Multi-container Docker setup
└── requirements.txt     # Python dependencies
```

## 🛠️ Prerequisites

Ensure you have the following installed on your machine:
* [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
* Python 3.11+ (if running locally without Docker)
* Git

## 🚦 Getting Started

### 1. Start the Infrastructure (Docker Compose)

The entire MLOps environment is fully containerized. Start all the necessary services with a single command:

```bash
docker-compose up -d
```

This command spins up the following services:
* **Prediction API**: `http://localhost:8000` (Swagger UI: `http://localhost:8000/docs`)
* **MLflow Tracking UI**: `http://localhost:5000`
* **MinIO Object Storage**: `http://localhost:9001` (User: `minioadmin` / Pass: `minioadmin123`)
* **Evidently AI Dashboard**: `http://localhost:8001`
* **Grafana Dashboards**: `http://localhost:3000` (User: `admin` / Pass: `admin`)
* **Prometheus**: `http://localhost:9090`
* **PostgreSQL**: `localhost:5432`

### 2. Run Tests Locally

Before deploying, ensure all tests are passing. We use `pytest` for various checks including unit tests, data quality tests, model sanity validations, and integration tests.

```bash
# Install requirements
pip install -r requirements.txt

# Run the test suite
PYTHONPATH=. pytest tests/ -v
```

## 📊 Endpoints & Monitoring

- **Health Check**: `GET http://localhost:8000/health`
- **Predict**: `POST http://localhost:8000/predict`
  - Accepts a JSON payload containing applicant features.
- **Metrics**: `GET http://localhost:8000/metrics`
  - Scraped by Prometheus to monitor application performance and request volume.

## 🤝 CI/CD Pipeline

This project uses **GitHub Actions** for Continuous Integration. Every push or pull request to the `main` or `develop` branch triggers a workflow that:
1. Provisions a Python 3.11 environment.
2. Lints the source code to enforce clean code standards (`flake8`).
3. Executes the full `pytest` suite ensuring Code Coverage remains high.

---
*Created for the DDM501 Module - Group 7.*
