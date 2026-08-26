# MLflow Monitoring & Serving Pipeline Documentation

This directory contains documentation for the MLflow monitoring and serving pipeline implementation.

## Documentation Files

### 1. MONITORING_SERVING.md

**Purpose:** Comprehensive guide for the monitoring and serving pipeline

**Contents:**
- **Architecture Overview**: System architecture diagram and component relationships
- **Component Descriptions**: Detailed explanation of each component's purpose and functionality
  - Configuration (`src/config.py`)
  - Drift Detector (`src/deployment/drift_detector.py`)
  - Monitored Model (`src/deployment/monitored_model.py`)
  - Monitoring Script (`src/deployment/monitor_drift.py`)
  - Deployment Script (`src/deployment/deploy_monitored.py`)
- **API Reference**: Complete reference for all classes, methods, and functions
- **Usage Examples**: Practical examples for training, deploying, monitoring, and making predictions
- **Configuration Guide**: Explanation of all configurable parameters and thresholds
- **Troubleshooting Guide**: Common issues and their solutions
- **Best Practices**: Recommendations for production use
- **Performance Considerations**: Optimization tips and memory management
- **Security Considerations**: Security best practices for production deployment
- **File Structure**: Complete project structure with file descriptions

### 2. ../IMPLEMENTATION_SUMMARY.md

**Purpose:** Summary of what was implemented

**Contents:**
- Implementation status for each phase
- Files created and modified
- Key features implemented
- Test results and verification
- Pipeline execution summary
- Usage commands for all components
- Known issues and limitations
- Files changed summary

## Quick Start

### Prerequisites
- MLflow server running on `127.0.0.1:5000`
- Python 3.13+ with all dependencies installed
- Data available in `data/images/raw/` with `data/images/metadata.csv`

### Run the Complete Pipeline

```bash
# Activate virtual environment
source .venv/bin/activate

# Run full pipeline (data -> train -> register -> deploy)
python -m src.pipeline

# This will:
# 1. Load and preprocess data (730 images)
# 2. Train CNN model (10 epochs)
# 3. Compute and save baseline statistics
# 4. Register model to MLflow
# 5. Deploy to Staging
```

### Deploy Monitored Model

```bash
# Deploy monitored model to MLflow
python -m src.deployment.deploy_monitored --deploy-only

# Start model server
python -m src.deployment.deploy_monitored --no-monitor --port 5001

# In another terminal, start monitoring
python -m src.deployment.monitor_drift --interval 60
```

### Make Predictions

```python
import requests

response = requests.post(
    "http://localhost:5001/invocations",
    json={"instances": ["data/images/raw/Clean_001.jpg"]}
)
# Note: a path works only where the file exists locally. For containerized
# serving, send the image as base64-encoded bytes instead:
#   import base64
#   b64 = base64.b64encode(open("data/images/raw/Clean_001.jpg","rb").read()).decode()
#   json={"instances": [b64]}

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Drift detected: {result['drift_detected']}")
```

## Component Reference

### 1. Configuration (src/config.py)

All drift detection and monitoring parameters are configured here.

**Key Parameters:**
- `DRIFT_WINDOW_SIZE = 100` - Samples before checking drift
- `DRIFT_PSI_THRESHOLD = 0.2` - PSI threshold for drift detection
- `DRIFT_KS_PVALUE_THRESHOLD = 0.05` - KS test p-value threshold
- `DRIFT_CHECK_INTERVAL = 60` - Seconds between monitoring checks
- `DRIFT_FEATURE_LAYER = "fc1"` - Layer for embedding extraction
- `MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"` - MLflow server URI

### 2. Drift Detector (src/deployment/drift_detector.py)

**Class:** `ImageDriftDetector`

**Purpose:** Detects various types of drift in image classification models

**Methods:**
- `calculate_psi(current, baseline, bins)` - Calculate Population Stability Index
- `detect_image_statistics_drift(...)` - Detect drift in RGB channel statistics
- `detect_class_distribution_drift(class_counts)` - Detect drift in class distribution
- `detect_confidence_drift(confidences)` - Detect drift in confidence scores
- `detect_embedding_drift(embeddings)` - Detect drift in feature embeddings
- `detect_drift(batch_data)` - Main method combining all drift types

**Usage:**
```python
from src.deployment.drift_detector import ImageDriftDetector

detector = ImageDriftDetector()
batch_data = {
    'image_means': (mean_r, mean_g, mean_b, std_r, std_g, std_b),
    'class_counts': {0: count_0, 1: count_1},
    'confidences': [0.5, 0.6, 0.7, ...],
    'embeddings': numpy_array_of_features,
}
result = detector.detect_drift(batch_data)
```

### 3. Monitored Model (src/deployment/monitored_model.py)

**Class:** `MonitoredPyTorchModel` (extends `mlflow.pyfunc.PythonModel`)

**Purpose:** Wraps PyTorch model with drift detection capabilities

**Methods:**
- `__init__(model, model_path, baseline_stats_path, feature_layer, window_size)`
- `load_context(context)` - Loads model and drift detector
- `extract_embeddings(images)` - Extracts features from specified layer
- `preprocess_image(image_path)` - Preprocesses image for inference
- `predict(context, model_input)` - Makes predictions with drift detection
- `_check_drift()` - Checks for drift when buffer is full
- `get_drift_status()` - Returns current drift detection status

**Usage:**
```python
from src.deployment.monitored_model import MonitoredPyTorchModel

monitored = MonitoredPyTorchModel(
    model=cnn_model,
    baseline_stats_path='artifacts/drift_baseline_stats.json',
    feature_layer='fc1',
    window_size=100,
)
monitored.load_context(None)
result = monitored.predict(None, "path/to/image.jpg")
```

### 4. Monitoring Script (src/deployment/monitor_drift.py)

**Functions:**
- `setup_mlflow_tracking()` - Sets up MLflow tracking
- `get_latest_monitored_run(model_name)` - Gets latest run
- `check_drift_alerts(run_id)` - Checks for drift alerts
- `send_email_alert(alerts)` - Example alert callback
- `monitor_continuously(run_id, interval, callback)` - Main monitoring loop
- `get_monitoring_run_id()` / `save_monitoring_run_id(run_id)` - Run ID management
- `start_monitoring_run()` - Starts new monitoring run
- `get_model_run_id(model_name)` - Gets run ID from model registry

**CLI Usage:**
```bash
# Monitor the latest run
python -m src.deployment.monitor_drift

# Monitor specific run with custom interval
python -m src.deployment.monitor_drift --run-id <run_id> --interval 30

# Start new monitoring run
python -m src.deployment.monitor_drift --start-run
```

### 5. Deployment Script (src/deployment/deploy_monitored.py)

**Functions:**
- `deploy_monitored_model(model_uri, port, monitor)` - Deploy and optionally monitor
- `deploy_and_log_monitored_model(pytorch_model_path)` - Log monitored model to MLflow
- `check_server_health(port)` - Check if server is running
- `stop_server(server_process)` - Stop the server

**CLI Usage:**
```bash
# Deploy monitored model to MLflow (no server)
python -m src.deployment.deploy_monitored --deploy-only

# Deploy and start server
python -m src.deployment.deploy_monitored --no-monitor --port 5001

# Deploy with monitoring
python -m src.deployment.deploy_monitored --port 5001
```

## Testing

### Test Files

1. **test_drift_detector.py** - Tests for drift detection functionality
2. **test_monitored_model.py** - Tests for monitored model
3. **test_monitoring.py** - Tests for monitoring script
4. **test_integration.py** - End-to-end integration tests

### Running Tests

```bash
# Run all tests
python -m pytest src/tests/ -v

# Run specific test file
python -m pytest src/tests/test_drift_detector.py -v

# Run with coverage
python -m pytest src/tests/ --cov=src --cov-report=html
```

## Configuration Guide

### Drift Detection Thresholds

Adjust these in `src/config.py`:

- **`DRIFT_PSI_THRESHOLD`**: Lower for more sensitive drift detection (default: 0.2)
- **`DRIFT_KS_PVALUE_THRESHOLD`**: Lower for more sensitive KS test (default: 0.05)
- **`DRIFT_WINDOW_SIZE`**: Number of samples before checking (default: 100)
- **`DRIFT_CHECK_INTERVAL`**: Seconds between monitoring checks (default: 60)

### Model Configuration

- **`DRIFT_FEATURE_LAYER`**: The layer name to extract embeddings from (default: "fc1")
- Must match a layer name in your PyTorch model

## Troubleshooting

### Common Issues

1. **Model Server Fails to Start**
   - Check MLflow server: `mlflow server --host 127.0.0.1 --port 5000`
   - Check port: `lsof -i :5001`
   - Verify model URI: `models:/{model_name}/latest`

2. **Drift Detection Not Working**
   - Verify baseline stats: `ls -la artifacts/drift_baseline_stats.json`
   - Check drift detector: `monitored.drift_detector is not None`
   - Ensure window size reached

3. **Prediction Returns Null**
   - Verify model path
   - Check model checkpoint exists
   - Ensure image paths are valid

4. **Import Errors**
   - Install dependencies
   - Run from project root or install as package

## Best Practices

1. Train with representative data for accurate baseline statistics
2. Set window size based on expected data volume
3. Tune thresholds based on domain tolerance for drift
4. Start monitoring only after model is in production/staging
5. Implement custom alert callbacks for notifications
6. Use MLflow model registry to track versions

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Serving](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)
- [Population Stability Index (PSI)](https://en.wikipedia.org/wiki/Population_stability_index)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test)

## Version Information

- **Implementation Date:** 2026-08-22
- **Status:** Production Ready
- **All Phases:** Complete
