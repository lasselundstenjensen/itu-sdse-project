#Here are just some tests for our 

import torch

from reestructured.model import SimpleCNN


def test_output_shape():
    model = SimpleCNN()
    images = torch.randn(2, 3, 64, 64)

    output = model(images)

    assert output.shape == (2, 1)


def test_works_with_different_batch_sizes():
    model = SimpleCNN()
    model.eval()  # disable dropout so the output is deterministic

    for batch_size in [1, 8]:
        x = torch.randn(batch_size, 3, 64, 64)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 1)