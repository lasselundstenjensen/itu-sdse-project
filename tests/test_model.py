import json
from pathlib import Path
import base64
import pytest
import mlflow.pyfunc
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent#---- get root of repo - regardless of where pytest was run from

AUC_THRESHOLD = 0.02#---- threshold for ROC AUC
ACC_THRESHOLD = 0.02#---- threshold for Accuracy

def _load_json(path: Path) -> dict:
    """Loads a json file from a path and returns it as a dictionary."""
    result = {}#---- empty dict for result
    with open(path) as f:
        result = json.load(f)
    
    return result

def _b64(img_path):
    """Read an image from the input path and return it's base64 encoding."""
    return base64.b64encode(img_path.read_bytes()).decode()


def test_json_results():
    """Computes absolute differences in ROC AUC & Accuracy from the current \"artifacts/model_results.json\"
    to the reference results saved in \"tests/fixtures/<commit-hash>/model_results.json\"."""
       
    #---- check if results to compare exist
    model_results_path = ROOT / "artifacts" / "model_results.json"
    if not model_results_path.exists():
        pytest.skip("No results found at \"artifacts/model_results.json\"")
    
    model_results = _load_json(model_results_path)#---- get results from current model
    reference_results = _load_json(ROOT / "tests" / "fixtures" / "974b7c2da065508646268fad001c5c166de6201d" / "artifacts" / "model_results.json")#---- dict for reference results
    
    acc_diff = abs(model_results["accuracy"] - reference_results["accuracy"])
    roc_auc_diff = abs(model_results["roc_auc"] - reference_results["roc_auc"])

    assert acc_diff <= ACC_THRESHOLD, f"Accuracy differs by {acc_diff:.4f} (limit {ACC_THRESHOLD})"
    assert roc_auc_diff <= AUC_THRESHOLD, f"ROC AUC differs by {roc_auc_diff:.4f} (limit {AUC_THRESHOLD})"
    

def test_if_model_predicts():
    """Check that the current \"models/model\" answers the 10 test images with the keys \"predictions\" & \"confidences\".\n
    Further, tests if these outputs are within the allowed values."""

    model_path = ROOT / "models" / "model"

    #---- check of the model exists
    if not (model_path / "MLmodel").exists():
        pytest.skip("No model found in \"models/model\".")

    #---- load the model with mlflow
    model = mlflow.pyfunc.load_model(str(model_path))

    #---- get the test image paths from the metadata.csv
    img_filenames = pd.read_csv(ROOT / "data" / "test" / "metadata.csv")["filename"].to_list()
    assert len(img_filenames) == 10, "Expected 10 test images in metadata.csv"
    img_paths = [ROOT / "data" / "test" / filename for filename in img_filenames]

    #---- inference the model with the base64 encoded images
    out = model.predict({"instances": [_b64(img_path) for img_path in img_paths]})

    #---- test whether the predictions match the expected schema
    assert ("predictions" in out and "confidences" in out), "expected both fields \"predictions\" and \"confidences\" in model output."
    assert len(out["predictions"]) == 10, "Expected 10 predictions in model output"
    assert len(out["confidences"]) == 10, "Expected 10 confidences in model output"
    assert all([p in (0, 1) for p in out["predictions"]]), "All predictions must be either 0 or 1."#---- this is especially to catch "None" outputs on model errors
    assert all([0.0 <= c <= 1.0 for c in out["confidences"]]), "All confidences must be in the interval (0, 1)."
