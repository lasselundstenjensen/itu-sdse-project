
from pathlib import Path

root = Path(__file__).resolve().parents[1]
# Paths (relative to the project root)
artifact_dir = f"{root}/artifacts"
mlflow_state_dir = f"{root}/mlflow-state"
image_data_dir = f"{root}/data/images/raw"
metadata_path = f"{root}/data/images/metadata.csv"
test_data_dir = f"{root}/data/test"
pytorch_model_path = artifact_dir + "/vial_cnn_model.pth"
model_results_path = artifact_dir + "/model_results.json"
drift_baseline_stats_path = artifact_dir + "/drift_baseline_stats.json"
class_names_path = artifact_dir + "/class_names.json"
image_statistics_path = artifact_dir + "/image_statistics.json"
inference_log_path = artifact_dir + "/inference_log.json"

