from pathlib import Path

from vial_classifier.config import DATA_DIR, MODELS_DIR, PROJ_ROOT


def test_project_root():
    expected_root = Path(__file__).resolve().parents[1]
    assert PROJ_ROOT == expected_root


def test_project_paths():
    assert DATA_DIR == PROJ_ROOT / "data"
    assert MODELS_DIR == PROJ_ROOT / "models"