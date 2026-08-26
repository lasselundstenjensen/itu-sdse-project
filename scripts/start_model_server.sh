#!/bin/bash

# Start MLflow Model Server for Production model
# Usage: ./start_model_server.sh

MODEL_NAME="cnn_model"
PORT=5001
LOG_FILE="./logs/model_server.log"

# Create logs directory if it doesn't exist
mkdir -p ./logs

echo "Starting MLflow model server for ${MODEL_NAME} on port ${PORT}..."
echo "Logs will be written to: ${LOG_FILE}"

# Start MLflow model server in background
# --no-conda: Use the current Python environment instead of creating a new conda environment
# -m models:/${MODEL_NAME}/Production: Serve the model with alias 'Production'
# -p ${PORT}: Serve on port 5001
# > ${LOG_FILE} 2>&1: Redirect stdout and stderr to log file
nohup mlflow models serve \
    -m models:/${MODEL_NAME}/Production \
    -p ${PORT} \
    --no-conda \
    > ${LOG_FILE} 2>&1 &

echo "Model server started in background!"
echo "  Model: models:/${MODEL_NAME}/Production"
echo "  Port: ${PORT}"
echo "  Log file: ${LOG_FILE}"
echo "  Inference endpoint: http://localhost:${PORT}/invoke"
echo ""
echo "To stop the server, run: pkill -f 'mlflow models serve'"
