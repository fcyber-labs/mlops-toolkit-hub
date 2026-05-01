
# STAGE 1: BASE - Common dependencies

FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi streamlit

WORKDIR /app

# Git setup for DVC
RUN git init && \
    git config user.email "docker@example.com" && \
    git config user.name "Docker Build"

# Copy common files
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

# Configure DVC remote
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


# STAGE 2: API - For serving predictions

FROM base AS api

WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


# STAGE 3: TRAINER - For running pipelines

FROM base AS trainer

WORKDIR /app
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["./run_all.sh"]


# DEFAULT: Full image (if no target specified)

FROM base AS full

WORKDIR /app
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["./run_all.sh"]