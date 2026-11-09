import torch

from vial_classifier.model import SimpleCNN


def test_output_is_one_logit_per_image():
    model = SimpleCNN().eval()
    out = model(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 1)


def test_parameter_count_matches_notebook_architecture():
    # conv1 896 + conv2 18,496 + fc1 2,097,280 + fc2 129
    assert sum(p.numel() for p in SimpleCNN().parameters()) == 2_116_801
