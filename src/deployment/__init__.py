"""
Deployment Modules for Image Classification

Contains modules for PyTorch CNN model deployment, stage management, and monitoring.

Modules:
- deploy.py: Model stage transitions (Staging, Production) for PyTorch models
- drift_detector.py: Drift detection functionality for monitoring data drift
- monitored_model.py: Monitored PyTorch model wrapper with drift detection
- monitor_drift.py: Monitoring script for periodic drift checking
- deploy_monitored.py: Deployment script for monitored models
"""

from .deploy import (
    deploy_model,
    transition_to_staging,
    transition_to_production,
    set_model_alias,
    wait_for_deployment,
)
from .drift_detector import ImageDriftDetector
from .monitored_model import MonitoredPyTorchModel
from .monitor_drift import (
    check_drift_alerts,
    monitor_continuously,
    send_email_alert,
    get_latest_monitored_run,
    start_monitoring_run,
    get_model_run_id,
    get_monitoring_run_id,
    save_monitoring_run_id,
)
from .deploy_monitored import (
    deploy_monitored_model,
    deploy_and_log_monitored_model,
    check_server_health,
    stop_server,
)

__all__ = [
    # Deployment
    'deploy_model',
    'transition_to_staging',
    'transition_to_production',
    'set_model_alias',
    'wait_for_deployment',
    # Drift detection
    'ImageDriftDetector',
    # Monitored model
    'MonitoredPyTorchModel',
    # Monitoring
    'check_drift_alerts',
    'monitor_continuously',
    'send_email_alert',
    'get_latest_monitored_run',
    'start_monitoring_run',
    'get_model_run_id',
    'get_monitoring_run_id',
    'save_monitoring_run_id',
    # Deploy monitored
    'deploy_monitored_model',
    'deploy_and_log_monitored_model',
    'check_server_health',
    'stop_server',
]
