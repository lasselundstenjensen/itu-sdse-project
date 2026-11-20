import base64
from io import BytesIO
import json

import torch
from PIL import Image

from vial_classifier.config import IMAGE_SIZE
from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.predict import SimpleLoggingWrapper


def test_prediction_preprocesses_images_and_appends_inference_logs(tmp_path):
    torch.manual_seed(42)
    image_buffer = BytesIO()
    image_size = (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2)
    Image.new("RGB", image_size, color=(0, 0, 0)).save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    log_path = tmp_path / "artifacts" / "inference_log.json"

    wrapper = SimpleLoggingWrapper(model=SimpleCNN(), log_path=log_path)
    wrapper.model.to(wrapper.device).eval()
    processed_image = wrapper.preprocess_image(encoded_image)
    assert tuple(processed_image.shape) == (1, 3, *IMAGE_SIZE)
    with torch.no_grad():
        probability = torch.sigmoid(
            wrapper.model(processed_image.to(wrapper.device))
        ).item()

    result = wrapper.predict(
        context=None,
        model_input={"instances": [encoded_image, image_bytes]},
    )

    expected_prediction = int(probability >= wrapper.threshold)
    expected_confidence = round(float(probability), 4)
    assert result["predictions"] == [expected_prediction, expected_prediction]
    assert result["confidences"] == [expected_confidence, expected_confidence]
    assert result["raw_outputs"] == [expected_confidence, expected_confidence]
    assert result["num_samples"] == 2

    logs = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(logs) == 2
    assert logs[0]["image_path"] == encoded_image
    assert logs[1]["image_path"] == "<bytes>"
    assert all(entry["prediction"] == expected_prediction for entry in logs)
    assert all(entry["confidence"] == probability for entry in logs)
    assert all(entry["timestamp"] for entry in logs)
