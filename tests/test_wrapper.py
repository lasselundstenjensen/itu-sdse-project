import base64
import json

from PIL import Image
import pytest
import torch

from vial_classifier import config
from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.wrapper import SimpleLoggingWrapper


@pytest.fixture
def wrapper(tmp_path):
    torch.manual_seed(0)
    model = SimpleCNN()
    model.eval()
    return SimpleLoggingWrapper(model=model, log_path=tmp_path / "inference_log.json")


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "vial.png"
    Image.new("RGB", (100, 80), color=(120, 60, 30)).save(path)
    return path


def test_preprocess_image_returns_a_model_ready_tensor(wrapper, image_path):
    tensor = wrapper.preprocess_image(str(image_path))
    assert tuple(tensor.shape) == (1, 3, *config.IMAGE_SIZE)


def test_predict_accepts_path_bytes_and_base64(wrapper, image_path):
    raw = image_path.read_bytes()
    inputs = [str(image_path), raw, base64.b64encode(raw).decode()]

    result = wrapper.predict(None, {"instances": inputs})

    assert result["num_samples"] == 3
    assert all(p in (0, 1) for p in result["predictions"])
    assert all(0.0 <= c <= 1.0 for c in result["confidences"])
    # the same image in three different forms must give the same answer
    assert len(set(result["confidences"])) == 1


def test_predict_returns_the_expected_keys(wrapper, image_path):
    result = wrapper.predict(None, {"instances": [str(image_path)]})
    assert set(result) == {"predictions", "confidences", "raw_outputs", "num_samples"}


def test_predict_appends_each_request_to_the_log(wrapper, image_path):
    wrapper.predict(None, {"instances": [str(image_path), str(image_path)]})

    logs = json.loads(wrapper.log_path.read_text())

    assert len(logs) == 2
    assert {"timestamp", "prediction", "confidence"} <= set(logs[0])


def test_broken_input_gives_none_and_does_not_stop_the_batch(wrapper, image_path):
    result = wrapper.predict(None, {"instances": ["not-an-image!!", str(image_path)]})

    assert result["predictions"][0] is None
    assert result["confidences"][0] is None
    assert result["predictions"][1] in (0, 1)
    assert result["num_samples"] == 2
    # only the good image was logged
    assert len(json.loads(wrapper.log_path.read_text())) == 1


def test_threshold_decides_the_predicted_class(wrapper, image_path):
    wrapper.threshold = 0.0
    assert wrapper.predict(None, {"instances": [str(image_path)]})["predictions"] == [1]

    wrapper.threshold = 2.0
    assert wrapper.predict(None, {"instances": [str(image_path)]})["predictions"] == [0]
