#!/bin/bash
set -e

# Combined container: MLflow tracking server (background) + model server (main process)
mkdir -p /mlflow/artifacts

mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --artifacts-destination /mlflow/artifacts \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 &

# Wait for the tracking server before starting inference logging
until python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/health')" 2>/dev/null; do
    sleep 1
done

exec mlflow models serve \
    -m /app/models/model \
    --host 0.0.0.0 \
    --port 5001 \
    --no-conda
