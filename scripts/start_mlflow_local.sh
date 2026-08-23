#!/bin/bash
set -e

# Local MLflow Server Setup Script
# Adapted from Guide 2: Individual Group MLflow Servers on DigitalOcean

MLFLOW_DIR="$HOME/mlflow"
PORT=5000

# Create directory structure
mkdir -p "$MLFLOW_DIR/artifacts"

# Create virtual environment and install MLflow
python3 -m venv "$MLFLOW_DIR/venv"
"$MLFLOW_DIR/venv/bin/pip" install mlflow

# Start MLflow tracking server
echo "Starting MLflow server..."
"$MLFLOW_DIR/venv/bin/mlflow" server \
  --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db" \
  --default-artifact-root "$MLFLOW_DIR/artifacts" \
  --host 127.0.0.1 \
  --port $PORT \
  > "$MLFLOW_DIR/mlflow.log" 2>&1 &

echo "=========================================="
echo "MLflow server is running!"
echo "  URL: http://127.0.0.1:$PORT"
echo "  Artifact root: $MLFLOW_DIR/artifacts"
echo "  Backend store: sqlite:///$MLFLOW_DIR/mlflow.db"
echo "  Logs: $MLFLOW_DIR/mlflow.log"
echo "  PID: $!"
echo "=========================================="
echo ""
echo "To stop the server, run:"
echo "  pkill -f 'mlflow server'"
echo ""
echo "To use in your code:"
echo "  mlflow.set_tracking_uri('http://127.0.0.1:$PORT')"
