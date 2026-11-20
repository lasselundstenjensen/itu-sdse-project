from pathlib import Path
from unittest.mock import Mock, call

import torch

from vial_classifier import dataset as dataset_module
from vial_classifier.config import DATA_DIR, MODELS_DIR, PROJ_ROOT
from vial_classifier.features import calculate_image_statistics, count_classes


def test_project_root():
    expected_root = Path(__file__).resolve().parents[1]
    assert PROJ_ROOT == expected_root


def test_project_paths():
    assert DATA_DIR == PROJ_ROOT / "data"
    assert MODELS_DIR == PROJ_ROOT / "models"


def test_count_classes():
    assert count_classes([0, 1, 1, 0, 1]) == {0: 2, 1: 3}


def test_calculate_image_statistics():
    images = torch.tensor([
        [[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]],
        [[[4.0, 4.0]], [[5.0, 5.0]], [[6.0, 6.0]]],
    ])

    statistics = calculate_image_statistics([(images, torch.tensor([0, 1]))])

    assert statistics == {
        "mean": [2.5, 3.5, 4.5],
        "std": [0.0, 0.0, 0.0],
        "total_samples": 2,
    }


def test_pull_data_uses_dvc_and_mocked_image_downloads(monkeypatch):
    run_command = Mock()
    download_metadata = Mock()
    monkeypatch.setattr(dataset_module.subprocess, "run", run_command)
    monkeypatch.setattr(
        dataset_module, "download_files_from_metadata", download_metadata
    )

    dataset_module.pull_data_from_dvc()

    run_command.assert_called_once_with(
        ["dvc", "pull"], cwd=PROJ_ROOT, check=True
    )
    assert download_metadata.call_args_list == [
        call(
            dataset_module.METADATA_PATH,
            dataset_module.IMAGE_DATA_DIR,
            "images/raw/",
        ),
        call(
            dataset_module.TEST_METADATA_PATH,
            dataset_module.TEST_DATA_DIR,
            "test/",
        ),
    ]
