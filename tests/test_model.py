import torch

from vial_classifier.modeling.model import SimpleCNN


def test_simple_cnn_output_shape():
    model = SimpleCNN()
    model.eval()
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1)