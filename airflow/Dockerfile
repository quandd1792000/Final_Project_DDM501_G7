FROM apache/airflow:2.8.1-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN grep -v -i "SQLAlchemy" /requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt "SQLAlchemy==1.4.51"
