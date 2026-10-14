#!/bin/bash
set -e

# Serve the exported pyfunc model (models/model) with `mlflow models serve`.
# The model is served straight from the local directory: no tracking server
# is contacted while serving, and replacing the files requires a restart.
#
# If models/model is missing, it is exported from the latest training run on
# the local tracking server (which must then be running - see
# scripts/start_mlflow_local.sh - and a training run must exist).
#
# Usage: ./scripts/start_model_server.sh
# Stop with: pkill -f "mlflow models serve"

PORT=5001
LOG_FILE="./logs/model_server.log"
MODEL_DIR="models/model"
TRACKING_URI="http://127.0.0.1:5000"

mkdir -p ./logs

if [ ! -f "$MODEL_DIR/MLmodel" ]; then
    echo "$MODEL_DIR not found; exporting from the latest training run..."
    if ! curl -sf "$TRACKING_URI/health" > /dev/null 2>&1; then
        echo "ERROR: no exported model at $MODEL_DIR and no tracking server at $TRACKING_URI."
        echo "Start one with ./scripts/start_mlflow_local.sh and train with:"
        echo "  uv run python -m src.pipeline"
        exit 1
    fi
    MLFLOW_TRACKING_URI="$TRACKING_URI" uv run python - <<'EOF'
import mlflow

runs = mlflow.search_runs(
    experiment_names=["vial_image_classification"],
    max_results=1,
    order_by=["start_time DESC"],
)
if runs.empty:
    raise SystemExit("No training run found on the tracking server. Train first with: uv run python -m src.pipeline")
run_id = runs.iloc[0].run_id
mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model_pyfunc",
    dst_path="models",
)
print(f"Exported pyfunc model from run {run_id}")
EOF
    rm -rf "$MODEL_DIR"
    mv models/model_pyfunc "$MODEL_DIR"
fi

# Free the port from a previous model server, if any
if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "Port $PORT is already in use. Stopping the existing model server..."
    lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo "Starting model server on port $PORT (logs: $LOG_FILE)..."
nohup uv run mlflow models serve \
    -m "$MODEL_DIR" \
    -p "$PORT" \
    --host 0.0.0.0 \
    --workers 1 \
    --no-conda \
    > "$LOG_FILE" 2>&1 &

echo "Model server starting in background."
echo "  Model: $MODEL_DIR (local directory; restart after replacing it)"
echo "  Inference endpoint: http://localhost:$PORT/invocations"
echo "  Log file: $LOG_FILE"
echo "To stop: pkill -f 'mlflow models serve'"
