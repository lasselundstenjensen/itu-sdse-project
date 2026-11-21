import io
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from vial_classifier.modeling.drift_baseline import compute_baseline_statistics
from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.predict import SimpleLoggingWrapper


def make_jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 768), color=(200, 200, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_baseline_statistics_have_expected_sections():
    torch.manual_seed(0)
    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    stats = compute_baseline_statistics(SimpleCNN(), loader, device=torch.device("cpu"))

    assert stats["class_distribution"]["total"] == 8
    assert stats["prediction_distribution"]["total"] == 8
    assert stats["embedding_statistics"]["shape"] == 128


def test_wrapper_predicts_and_logs_request(tmp_path):
    log_path = tmp_path / "inference_log.json"
    wrapper = SimpleLoggingWrapper(model=SimpleCNN(), log_path=log_path)
    wrapper.load_context(None)

    result = wrapper.predict(None, [make_jpeg_bytes()])

    assert result["num_samples"] == 1
    assert result["predictions"][0] in (0, 1)
    assert len(json.loads(log_path.read_text())) == 1