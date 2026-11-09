import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from vial_classifier.inference import SimpleLoggingWrapper, create_model_from_checkpoint
from vial_classifier.model import SimpleCNN


def _png_bytes(size=(160, 76), color=(120, 40, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def wrapper(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return SimpleLoggingWrapper(model=SimpleCNN(), log_path=tmp_path / "inference_log.json")


def test_preprocess_accepts_bytes_and_base64(wrapper):
    raw = _png_bytes()
    from_bytes = wrapper.preprocess_image(raw)
    from_b64 = wrapper.preprocess_image(base64.b64encode(raw).decode())
    assert tuple(from_bytes.shape) == (1, 3, 64, 64)
    assert np.allclose(from_bytes.numpy(), from_b64.numpy())


def test_preprocess_rejects_unsupported_types(wrapper):
    with pytest.raises(TypeError):
        wrapper.preprocess_image(12345)


def test_predict_returns_predictions_and_logs_each_request(wrapper):
    raw = _png_bytes()
    payload = {"instances": [base64.b64encode(raw).decode(), base64.b64encode(raw).decode()]}

    result = wrapper.predict(None, payload)

    assert result["num_samples"] == 2
    assert len(result["predictions"]) == 2
    assert all(p in (0, 1) for p in result["predictions"])
    logs = json.loads(wrapper.log_path.read_text())
    assert len(logs) == 2
    assert set(logs[0]) == {"timestamp", "image_path", "prediction", "confidence", "raw_output"}


def test_bad_image_yields_none_without_failing_batch(wrapper):
    good = base64.b64encode(_png_bytes()).decode()
    result = wrapper.predict(None, {"instances": ["not-an-image-or-path", good]})
    assert result["predictions"][0] is None
    assert result["predictions"][1] in (0, 1)


def test_extract_inputs_flattens_nested_lists():
    assert SimpleLoggingWrapper._extract_inputs({"data": [["a", "b"], "c"]}) == ["a", "b", "c"]
    assert SimpleLoggingWrapper._extract_inputs("single") == ["single"]


def test_checkpoint_loader_rejects_unknown_architecture():
    with pytest.raises(ValueError):
        create_model_from_checkpoint({"architecture": "ResNet"})
