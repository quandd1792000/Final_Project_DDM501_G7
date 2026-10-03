# Credit Risk Classifier - MLOps Pipeline (Final Project DDM501 G7)

[![MLOps CI/CD Pipeline](https://github.com/quandd1792000/Final_Project_DDM501_G7/actions/workflows/ci.yml/badge.svg)](https://github.com/quandd1792000/Final_Project_DDM501_G7/actions/workflows/ci.yml)

An end-to-end Machine Learning Operations (MLOps) architecture designed to train, deploy, and monitor a Credit Risk Classification model. This repository contains the complete infrastructure code, APIs, and model monitoring dashboards.

## 🚀 Features

* **Model Serving (FastAPI)**: Robust REST API to serve predictions in real-time, fortified with input guardrails.
* **Experiment Tracking & Orchestration (MLflow + Airflow)**: Full integration with MLflow for tracking parameters and metrics, and Airflow for automated model training and retraining pipelines (Backend: PostgreSQL, Artifacts: MinIO S3).
* **Data Drift Monitoring (Evidently AI)**: Automated detection of data drift and data quality degradation running asynchronously in the background.
* **System Observability (Prometheus & Grafana)**: Interactive dashboards for monitoring real-time API performance, model metrics, and system health.
* **Automated Alerting (Alertmanager + Discord)**: Real-time alerting for critical system events (e.g., High API Error Rate, Model Not Loaded) delivered directly to a Discord channel.
* **CI/CD Automation (GitHub Actions + Watchtower)**: Automated code linting (`flake8`), testing (`pytest`), Docker image building/pushing to GHCR, and continuous deployment using Watchtower.

## 📁 Project Structure

```bash
.
├── .github/workflows/   # CI/CD pipelines (GitHub Actions)
├── airflow/             # Airflow orchestration configuration and Dockerfile
├── api/                 # FastAPI service for the prediction endpoints
├── config/              # Prometheus, Grafana, Alertmanager, and Alert Rules configurations
├── dags/                # Airflow DAGs for model training
├── evidently/           # Data Drift tracking and ML monitoring logic
├── scripts/             # Model training and deployment scripts
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
* **Airflow Orchestrator**: `http://localhost:8080` (Trigger automated training here)
* **MLflow Tracking UI**: `http://localhost:5000`
* **MinIO Object Storage**: `http://localhost:9001` (User: `minioadmin` / Pass: `minioadmin123`)
* **Evidently AI Dashboard**: `http://localhost:8001`
* **Grafana Dashboards**: `http://localhost:3000` (User: `admin` / Pass: `admin`)
* **Prometheus**: `http://localhost:9090`
* **Alertmanager**: `http://localhost:9093`
* **Watchtower**: Automatically polls GHCR for new images and updates containers dynamically.

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
  - Accepts a JSON payload containing applicant features. Forwards prediction data to Evidently.
- **Metrics**: `GET http://localhost:8000/metrics`
  - Scraped by Prometheus to monitor application performance and request volume.

## 🤝 CI/CD Pipeline

This project uses **GitHub Actions** and **Watchtower** for Continuous Integration and Continuous Deployment (CI/CD):
1. **CI**: Every push to the `main` branch triggers a workflow that:
   - Provisions a Python 3.11 environment.
   - Lints the source code to enforce clean code standards (`flake8`).
   - Executes the full `pytest` suite ensuring Code Coverage.
   - Builds Docker images for the API and Evidently services and pushes them to GitHub Container Registry (GHCR).
2. **CD**: Watchtower continuously polls GHCR (every 60 seconds). Once a new image is pushed, Watchtower automatically downloads the image, stops the old containers, and restarts the new ones with zero downtime.

---
*Created for the DDM501 Module - Group 7.*
