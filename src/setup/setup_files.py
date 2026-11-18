from ..config.paths import artifact_dir, mlflow_state_dir, image_data_dir, test_data_dir

def initialize_directories():

    directories = [artifact_dir, mlflow_state_dir, f"{mlflow_state_dir}/artifacts", image_data_dir, test_data_dir]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return 1