#!/bin/bash
set -e

# Start a local MLflow tracking server for training. Mirrors the layout CI
# uses (see .github/workflows/train.yml): sqlite backend plus artifacts under
# mlflow-state/, all served by the same process, so run history accumulates
# across local training runs and the directory stays gitignored.
#
# Usage: ./scripts/start_mlflow_local.sh   (keep this terminal open)
# Stop with: pkill -f "mlflow server"

PORT=5000
STATE_DIR="./mlflow-state"

if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "Tracking server already running at http://127.0.0.1:$PORT"
    exit 0
fi

mkdir -p "$STATE_DIR/artifacts"

echo "Starting MLflow tracking server..."
uv run mlflow server \
  --backend-store-uri "sqlite:///$PWD/$STATE_DIR/mlflow.db" \
  --artifacts-destination "file:///$PWD/$STATE_DIR/artifacts" \
  --serve-artifacts \
  --host 127.0.0.1 \
  --port "$PORT" &

# Wait for the server to come up
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: tracking server did not become healthy on port $PORT."
    echo "On macOS, port 5000 is used by AirPlay Receiver - disable it or"
    echo "free the port, then retry."
    exit 1
fi

echo "=========================================="
echo "MLflow tracking server running"
echo "  URL:   http://127.0.0.1:$PORT"
echo "  State: $STATE_DIR (gitignored, persists between runs)"
echo "  Stop:  pkill -f 'mlflow server'"
echo "=========================================="
echo "Next: train with  uv run python -m src.pipeline"
