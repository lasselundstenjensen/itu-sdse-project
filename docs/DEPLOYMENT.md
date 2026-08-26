# Model Deployment & Serving Guide

This document explains the two-step workflow for training, deploying, and serving the glass vial image classification model using MLflow.

## Overview

The project uses a **two-step workflow** to separate concerns:

1. **Step 1: Training & Registration** - Train the model and register it to MLflow Model Registry
2. **Step 2: Deployment & Serving** - Deploy the best model and start the model server

This separation allows for:
- Clean CI/CD pipeline integration
- Independent scaling of training and serving
- Flexible deployment strategies (e.g., only deploy after manual approval)

## Key Features

- **MLflow Model Registry**: Models are registered and versioned in MLflow
- **Stage Transitions**: Models move from None → Staging → Production
- **Inference Logging**: All predictions logged to JSON file and MLflow metrics
- **MLflow Metrics**: Inference metrics (latency, predictions, confidence) tied to model's run ID
- **Background Serving**: Model server can run in background for production use
- **Manual Deployment**: Students trigger deployment after training (no auto-polling)

## Two-Step Workflow

```
┌─────────────────────────────────────────────────────────────┐
│               Step 1: Training & Registration               │
│               (python -m src.pipeline)                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
              Train CNN Model
                          ↓
          Register to MLflow Model Registry
                          ↓
              (Staging stage by default)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               Step 2: Deployment & Serving                  │
│      (python -m src.serve_model)                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
              Get Best Model from Registry
                          ↓
          Wrap with SimpleLoggingWrapper
                          ↓
          Log to MLflow as Servable Model
                          ↓
          Start MLflow Model Server (port 5001)
```

## Step 1: Training & Registration

Run the complete pipeline to train the model and register it to MLflow:

```bash
# Basic training and registration to Staging
python -m src.pipeline

# Directly deploy to Production after training
python -m src.pipeline --deploy-to-prod
```

**What happens:**
1. Data pipeline: Fetch, preprocess, and create features
2. Model pipeline: Train CNN model
3. Model registry: Register the model to MLflow Model Registry in Staging
4. Optionally transition to Production with `--deploy-to-prod`

**Outputs:**
- Model checkpoint saved to `artifacts/vial_cnn_model.pth`
- Model registered in MLflow Model Registry
- Model version in Staging (or Production if `--deploy-to-prod`)

## Step 2: Deployment & Serving

Deploy the best model from the registry and start the model server:

```bash
# Basic deployment and serving (default: from registry, Staging, port 5001)
python -m src.serve_model

# Deploy from Production instead of Staging
python -m src.serve_model --stage Production

# Deploy without starting server (MLflow only)
python -m src.serve_model --no-serve

# Start server in background (detached mode - doesn't hang terminal)
python -m src.serve_model --background
# Or use short flag:
python -m src.serve_model -d

# Deploy from latest experiment run instead of registry
python -m src.serve_model --from-experiment --from-registry False

# Custom model name and port
python -m src.serve_model --model-name my_model --port 8080

# Transition to Production after deployment
python -m src.serve_model --transition-to-prod

# Skip MLflow registration (only create wrapper and log to run)
python -m src.serve_model --no-register
```

**What happens:**
1. Get best model from MLflow Model Registry (Staging by default)
2. Load the PyTorch model
3. Wrap it with `SimpleLoggingWrapper` for inference logging (tied to model's run ID)
4. Log the wrapped model to MLflow as a servable model
5. Start MLflow model server on specified port (default: 5001)

**Outputs:**
- Wrapped model logged to MLflow
- Model server running on specified port
- Inference requests logged to `artifacts/inference_log.json`
- **Inference metrics logged to MLflow tracking** (tied to model's run ID):
  - `inference_latency_ms`: Request latency in milliseconds
  - `predictions_total`: Total number of predictions
  - `predictions_class_0`: Count of class 0 predictions
  - `predictions_class_1`: Count of class 1 predictions
  - `avg_confidence`: Average confidence score across predictions

## Command Line Arguments

### `serve_model.py` Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-name` | str | `cnn_model` | Name of the model in MLflow |
| `--experiment-name` | str | `vial_image_classification` | Name of the experiment |
| `--metric` | str | `f1_score` | Metric for selecting best run |
| `--from-registry` | flag | True | Get model from MLflow Model Registry |
| `--from-experiment` | flag | False | Get model from latest experiment run |
| `--stage` | str | `Staging` | Stage to look for (`Staging`, `Production`, `Archived`) |
| `--no-register` | flag | False | Skip registering to MLflow Model Registry |
| `--transition-to-prod` | flag | False | Transition to Production after deployment |
| `--port` | int | 5001 | Port for the model server |
| `--no-serve` | flag | False | Deploy to MLflow only, skip starting server |
| `--timeout` | int | 30 | Timeout for server startup (seconds) |
| `--background` / `-d` | flag | False | Run server in background (detached mode) |

## CI/CD Integration

### Manual Deployment Workflow (Recommended for Students)

Since deployment is **manual** (not auto-polling), students should set up their CI/CD to:

1. **Train** on push (registers model to Staging)
2. **Manually trigger deploy** after review (promotes to Production)

This gives students control over when to deploy new models.

**Example workflow:**

```bash
# Step 1: Training (registers model to Staging)
python -m src.pipeline

# Step 2: Deploy and serve in background (auto-promotes to Production)
python -m src.serve_model --background --transition-to-prod

# Step 3: Submit predictions
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"instances": ["image1.jpg", "image2.jpg"]}'

# View metrics in MLflow UI (tied to model's run)
# Open http://localhost:5002, find the training run, see inference metrics

# When training new model:
# Step 1: Training creates new version in Staging
python -m src.pipeline
# Step 2: Manually deploy new version (brief ~5s interruption while server restarts)
python -m src.serve_model --background --transition-to-prod
```

### GitHub Actions

Here's an example workflow that:
1. Runs training and registration on push to main
2. Automatically triggers deployment after successful training

**Note:** For production, consider separating train and deploy into separate workflows with manual approval.

```yaml
name: Train and Deploy Model

on:
  push:
    branches: [ main, v2 ]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Start MLflow server
        run: |
          mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002 &
          sleep 5
      - name: Run training pipeline
        run: python -m src.pipeline

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Start MLflow server
        run: |
          mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002 &
          sleep 5
      - name: Deploy and serve model
        run: python -m src.serve_model --no-serve --transition-to-prod
```

### Manual Approval Workflow (Recommended)

For better control, use GitHub Actions with manual approval:

```yaml
name: Train and Deploy with Approval

on:
  push:
    branches: [ main, v2 ]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Start MLflow server
        run: |
          mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002 &
          sleep 5
      - name: Run training pipeline
        run: python -m src.pipeline

  deploy:
    needs: train
    if: github.ref == 'refs/heads/main'  # Only deploy from main
    runs-on: ubuntu-latest
    environment:
      name: production
      url: http://your-server:5001
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Start MLflow server
        run: |
          mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002 &
          sleep 5
      - name: Deploy to Production
        run: python -m src.serve_model --no-serve --transition-to-prod
```

### GitLab CI

```yaml
stages:
  - train
  - deploy

train:
  stage: train
  script:
    - mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002 &
    - sleep 5
    - python -m src.pipeline
  artifacts:
    paths:
      - mlruns/
      - artifacts/
    expire_in: 1 week

deploy:
  stage: deploy
  script:
    - mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002 &
    - sleep 5
    - python -m src.serve_model --no-serve
  needs: [train]
```

### Makefile

```makefile
.PHONY: train deploy serve all

MLFLOW_SERVER = mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002

train:
	@echo "Starting MLflow server..."
	@$(MLFLOW_SERVER) &
	@sleep 5
	python -m src.pipeline

deploy:
	@echo "Starting MLflow server..."
	@$(MLFLOW_SERVER) &
	@sleep 5
	python -m src.serve_model --no-serve

serve: deploy
	python -m src.serve_model

all: train deploy serve
```

### Manual Trigger with curl

After deploying, you can test the model server:

```bash
# Start the server
python -m src.serve_model --port 5001

# In another terminal, test inference
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"data": ["/path/to/test/image.jpg"]}'

# Check health
curl http://localhost:5001/health
```

## Model Server API

Once the model server is running, it provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check server health |
| `/invocations` | POST | Make inference requests |
| `/model` | GET | Get model metadata |

### Inference Request Format

```json
{
  "data": ["/path/to/image1.jpg", "/path/to/image2.jpg"]
}
```

Or with instances:

```json
{
  "instances": ["/path/to/image1.jpg"]
}
```

### Inference Response Format

```json
{
  "predictions": [1, 0],
  "confidences": [0.95, 0.45],
  "timestamp": "2026-08-23T10:00:00+00:00",
  "num_samples": 2
}
```

## Inference Logging

All inference requests are automatically logged to **two locations**:

### 1. JSON File Log

Requests are logged to `artifacts/inference_log.json`:

```json
[
  {
    "timestamp": "2026-08-23T10:00:00+00:00",
    "image_path": "/path/to/image.jpg",
    "prediction": 1,
    "confidence": 0.95
  }
]
```

### 2. MLflow Metrics (New!)

**Inference metrics are now logged to MLflow tracking** and tied to the model's original run ID. This allows you to:

- View inference performance alongside training metrics in the MLflow UI
- Track model performance over time
- Monitor prediction distributions and latency

**Metrics logged:**
- `inference_latency_ms`: Request processing time in milliseconds
- `predictions_total`: Total number of valid predictions in the request
- `predictions_class_0`: Count of class 0 (non-defective) predictions
- `predictions_class_1`: Count of class 1 (defective) predictions
- `avg_confidence`: Average confidence score across all predictions

**How to view:**
1. Open MLflow UI at http://localhost:5002
2. Find the training run that produced your model
3. Click on the run to see the metrics chart
4. Inference metrics will appear as time-series data alongside training metrics

**Note:** Metrics are logged with incrementing step numbers, so each inference request creates a new data point in the time-series.

## Drift Detection

After deployment, you can check for drift using the separate CLI command:

```bash
# Check for drift in the last 7 days (default)
python -m src.deployment.check_drift

# Check for drift in the last 30 days
python -m src.deployment.check_drift --days 30

# Check with custom thresholds
python -m src.deployment.check_drift --psi-threshold 0.25 --ks-threshold 0.01
```

## Troubleshooting

### Common Issues

1. **MLflow server not running**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002
   ```

2. **Model not found in registry**
   - Make sure Step 1 (training) completed successfully
   - Check MLflow UI at `http://localhost:5002`
   - Run `python -m src.pipeline` first

3. **Port already in use**
   ```bash
   # Find and kill the process
   lsof -i :5001
   kill -9 <PID>
   ```

4. **Dependency issues**
   ```bash
   pip install -r requirements.txt
   ```

### Debug Mode

For verbose output, set the `MLFLOW_TRACKING_URI` environment variable:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5002
python -m src.serve_model --port 5001
```

## File Structure

```
src/
├── pipeline.py                    # Step 1: Training & Registration
├── serve_model.py               # Step 2: Deployment & Serving
├── deployment/
    ├── __init__.py               # Exports for deployment modules
    ├── deploy.py                 # Stage transitions (Staging, Production)
    ├── logging_wrapper.py        # SimpleLoggingWrapper for inference logging
    ├── deploy_monitored.py       # Alternative deployment script
    └── check_drift.py            # Drift detection CLI
```

## Migration from Old Workflow

If you were previously using `deploy_monitored.py`, the new `serve_model.py` provides:

- **Simpler workflow**: Single command for complete deployment
- **Better separation**: Clear two-step process
- **More flexible**: Can deploy from registry or experiment
- **Production-ready**: Built-in transition to Production

The old script is still available for backwards compatibility.

## Best Practices

1. **Always run training first**: Ensure models are in the registry before deployment
2. **Use --no-serve for CI/CD**: Let your infrastructure manage the server
3. **Test locally first**: Run both steps locally before setting up CI/CD
4. **Monitor inference logs**: Regularly check `artifacts/inference_log.json`
5. **Run drift detection**: Schedule regular drift checks
