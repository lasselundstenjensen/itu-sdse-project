#!/bin/bash

# Start MLflow Model Server for Production model
# Usage: ./start_model_server.sh
# NOTE: The MLflow tracking server must be running separately (use scripts/start_mlflow_local.sh)

MODEL_NAME="cnn_model"
PORT=5001
LOG_FILE="./logs/model_server.log"

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Set MLflow tracking URI - assumes tracking server is running on port 5002
export MLFLOW_TRACKING_URI=http://127.0.0.1:5002

echo "Starting MLflow model server for ${MODEL_NAME} on port ${PORT}..."
echo "Logs will be written to: ${LOG_FILE}"

# Kill any existing model server processes on port 5001
if lsof -i :${PORT} > /dev/null 2>&1; then
    echo "Port ${PORT} is already in use. Killing existing model server processes..."
    pkill -f "mlflow models serve" || true
    lsof -ti :${PORT} | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start MLflow model server in background
# --no-conda: Use the current Python environment instead of creating a new conda environment
# -m models:/${MODEL_NAME}@Production: Serve the model with alias 'Production'
# -p ${PORT}: Serve on port 5001
# > ${LOG_FILE} 2>&1: Redirect stdout and stderr to log file
nohup mlflow models serve \
    -m models:/${MODEL_NAME}@Production \
    -p ${PORT} \
    --no-conda \
    > ${LOG_FILE} 2>&1 &

echo "Model server started in background!"
echo "  Model: models:/${MODEL_NAME}@Production"
echo "  Port: ${PORT}"
echo "  Log file: ${LOG_FILE}"
echo "  Inference endpoint: http://localhost:${PORT}/invocations"
echo ""
echo "To stop the server, run: pkill -f 'mlflow models serve'"
