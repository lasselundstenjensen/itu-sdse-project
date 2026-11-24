from ..config.paths import artifact_dir, mlflow_state_dir, image_data_dir, test_data_dir
from pathlib import Path
def initialize_directories():

    directories = [artifact_dir, mlflow_state_dir, f"{mlflow_state_dir}/artifacts", image_data_dir, test_data_dir]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    return 1
initialize_directories()