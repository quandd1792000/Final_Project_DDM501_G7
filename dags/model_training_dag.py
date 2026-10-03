from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Ensure the scripts module can be imported
sys.path.append("/opt/airflow")

from scripts.training import train_and_track  # noqa: E402

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'credit_risk_model_training',
    default_args=default_args,
    description='A DAG to train and evaluate the Credit Risk model',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'training'],
) as dag:

    def run_training():
        # Overwrite MinIO URL to be reachable from within Docker network
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
        os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
        train_and_track()

    train_task = PythonOperator(
        task_id='train_and_track_model',
        python_callable=run_training,
    )

    train_task
