import shutil
import json
from pathlib import Path
from ..config.paths import artifact_dir, class_names_path 


def clean_artifacts():

    if Path(artifact_dir).exists():
        shutil.rmtree(artifact_dir)

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    class_names = {0: "good", 1: "defective"}
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {class_names_path}")
    return 1
clean_artifacts()