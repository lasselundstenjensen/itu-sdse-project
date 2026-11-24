import mlflow.pyfunc
import pandas as pd
import pytest

from vial_classifier.config import EXPORT_DIR, TEST_DATA_DIR, TEST_METADATA_PATH

pytestmark = pytest.mark.skipif(
    not (EXPORT_DIR / "MLmodel").exists(),
    reason="No exported model: run modeling.train and modeling.export first",
)


def test_exported_model_predicts_held_out_test_set(tmp_path):
    filenames = pd.read_csv(TEST_METADATA_PATH)["filename"]
    image_paths = [str(TEST_DATA_DIR / name) for name in filenames]

    model = mlflow.pyfunc.load_model(str(EXPORT_DIR))
    # Log to a temp file so the test does not pollute the real inference log
    model.unwrap_python_model().log_path = tmp_path / "inference_log.json"

    result = model.predict(image_paths)

    assert result["num_samples"] == len(image_paths)
    assert all(pred in (0, 1) for pred in result["predictions"])
    assert all(0.0 <= conf <= 1.0 for conf in result["confidences"])