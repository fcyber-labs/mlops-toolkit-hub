FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi streamlit

WORKDIR /app

RUN git init && \
    git config user.email "docker@example.com" && \
    git config user.name "Docker Build"

COPY src/         src/
COPY src_phase_1/ src_phase_1/
COPY config/      config/
COPY *.sh         ./
COPY *.yaml       ./
COPY .dvc/        .dvc/
COPY dvc.yaml     dvc.yaml
COPY dvc.lock     dvc.lock


RUN mkdir -p \
    data/raw \
    data/processed \
    data/streaming/german \
    models reports logs

RUN chmod +x *.sh
RUN git add . || true


RUN dvc remote add origin s3://dvc \
    --config /app/.dvc/config 2>/dev/null || true && \
    dvc remote modify origin endpointurl \
    https://dagshub.com/fcyber/german-credit-mlops.dvc \
    --config /app/.dvc/config 2>/dev/null || true && \
    dvc remote default origin \
    --config /app/.dvc/config 2>/dev/null || true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    DVC_NO_ANALYTICS=1

EXPOSE 8000 8501
CMD ["./run_all.sh"]