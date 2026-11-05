import torch

from vial_classifier.modeling.model import SimpleCNN, get_device


def test_forward_pass_returns_one_logit_per_image():
    output = SimpleCNN()(torch.zeros(2, 3, 64, 64))

    assert output.shape == (2, 1)


def test_architecture_matches_notebook():
    # conv1 896 + conv2 18,496 + fc1 2,097,280 + fc2 129
    assert sum(p.numel() for p in SimpleCNN().parameters()) == 2_116_801


def test_get_device_matches_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert get_device().type == expected
