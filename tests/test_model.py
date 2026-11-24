import torch
from PIL import Image

from vial_classifier.features import define_transforms
from vial_classifier.modeling.model import SimpleCNN


def test_simple_cnn_outputs_one_logit_per_image():
    model = SimpleCNN()
    output = model(torch.randn(4, 3, 64, 64))
    assert output.shape == (4, 1)


def test_transforms_resize_camera_image_to_model_input():
    train_transforms, val_transforms = define_transforms()
    image = Image.new("RGB", (1600, 768))
    assert train_transforms(image).shape == (3, 64, 64)
    assert val_transforms(image).shape == (3, 64, 64)