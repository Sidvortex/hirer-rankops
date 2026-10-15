FROM python:3.13-slim

# libgomp is LightGBM's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml pyproject.toml ./
COPY rankops ./rankops
COPY flows ./flows
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1 \
    RANKOPS_ROOT=/app \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "rankops.serve:app", "--host", "0.0.0.0", "--port", "8000"]
