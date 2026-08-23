#!/bin/bash
set -e

# DigitalOcean MLflow Server Setup Script
# Adapted from Guide 2: Individual Group MLflow Servers on DigitalOcean

MLFLOW_DIR="/home/mlflow"
PORT=5000

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me)

echo "Detected public IP: $PUBLIC_IP"
echo ""

# System update and install dependencies
echo "Updating system and installing dependencies..."
apt update && apt upgrade -y
apt install -y python3-pip python3-venv

# Create MLflow user and directory structure
echo "Creating MLflow user and directories..."
adduser --system --group mlflow
mkdir -p "$MLFLOW_DIR/artifacts"
chown -R mlflow:mlflow "$MLFLOW_DIR"

# Install MLflow in virtual environment
echo "Installing MLflow..."
sudo -u mlflow bash -c "python3 -m venv $MLFLOW_DIR/venv"
sudo -u mlflow bash -c "$MLFLOW_DIR/venv/bin/pip install mlflow"

# Start MLflow tracking server
echo "Starting MLflow server..."
sudo -u mlflow bash -c "nohup $MLFLOW_DIR/venv/bin/mlflow server \
  --backend-store-uri sqlite:///$MLFLOW_DIR/mlflow.db \
  --default-artifact-root $MLFLOW_DIR/artifacts \
  --host 0.0.0.0 \
  --port $PORT > $MLFLOW_DIR/mlflow.log 2>&1 &"

# Configure firewall
echo "Configuring firewall..."
ufw allow $PORT/tcp
ufw allow 22/tcp
ufw --force enable

echo ""
echo "=========================================="
echo "MLflow server is running!"
echo "  Public URL: http://$PUBLIC_IP:$PORT"
echo "  Local URL: http://localhost:$PORT"
echo "  Artifact root: $MLFLOW_DIR/artifacts"
echo "  Backend store: sqlite:///$MLFLOW_DIR/mlflow.db"
echo "  Logs: $MLFLOW_DIR/mlflow.log"
echo "=========================================="
echo ""
echo "To use in your code (on this server):"
echo "  mlflow.set_tracking_uri('http://localhost:$PORT')"
echo ""
echo "To use in your code (from external machine):"
echo "  mlflow.set_tracking_uri('http://$PUBLIC_IP:$PORT')"
echo ""
echo "To stop the server, run:"
echo "  pkill -f 'mlflow server'"
echo ""
echo "Note: The server runs as the 'mlflow' user under nohup,"
echo "so it will continue running after you disconnect SSH."
