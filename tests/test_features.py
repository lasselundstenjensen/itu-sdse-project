import json

import numpy as np
from PIL import Image
import pytest
import torch

from vial_classifier import config
from vial_classifier.features import (
    calculate_image_statistics,
    define_transforms,
    save_class_names,
)


@pytest.mark.parametrize("transform_index", [0, 1], ids=["train", "val"])
def test_transforms_output_shape(transform_index):
    image = Image.new("RGB", (160, 77), color=(120, 60, 30))

    output = define_transforms()[transform_index](image)

    assert output.shape == (3, *config.IMAGE_SIZE)


def test_calculate_image_statistics_per_channel():
    # Two batches of different sizes; each channel holds a known constant value
    batch = torch.ones(2, 3, 4, 4) * torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
    loader = [(batch, None), (batch[:1], None)]

    stats = calculate_image_statistics(loader)

    assert stats["total_samples"] == 3
    assert np.allclose(stats["mean"], [1.0, 2.0, 3.0])
    assert np.allclose(stats["std"], [0.0, 0.0, 0.0])


def test_save_class_names_creates_parent_dir_and_keeps_notebook_names(tmp_path):
    path = tmp_path / "new_dir" / "class_names.json"

    save_class_names(path)

    assert json.loads(path.read_text()) == {"0": "good", "1": "defective"}
