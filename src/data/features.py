"""
Image Feature Engineering Module

Handles image-specific artifact creation and feature engineering.
This module creates and saves artifacts for the image classification pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..config import (
    ARTIFACT_DIR,
    CLASS_NAMES_PATH,
    IMAGE_STATISTICS_PATH,
)
from ..utils import print_section_header


def save_image_artifacts(
    class_names: Dict[int, str] = None,
    image_statistics: Dict = None,
) -> None:
    """
    Save image-specific artifacts for the classification pipeline.

    Saves:
    - class_names.json: Mapping of class indices to human-readable names
    - image_statistics.json: Statistics computed from training images

    Parameters
    ----------
    class_names : Dict[int, str], optional
        Dictionary mapping class indices to names (e.g., {0: "good", 1: "defective"}).
        If None, uses default vial classification names.
    image_statistics : Dict, optional
        Dictionary with image statistics (mean, std per channel).
        If None, will be loaded from existing file or computed later.
    """
    print("\nSaving image artifacts...")

    # Default class names for vial classification
    if class_names is None:
        class_names = {
            0: "good",
            1: "defective"
        }

    # Save class names
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {CLASS_NAMES_PATH}")

    # Save image statistics if provided
    if image_statistics is not None:
        IMAGE_STATISTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(IMAGE_STATISTICS_PATH, "w") as f:
            json.dump(image_statistics, f, indent=2)
        print(f"Image statistics saved to {IMAGE_STATISTICS_PATH}")


def create_features() -> Dict[str, Path]:
    """
    Complete feature engineering pipeline for image classification.

    This function creates and saves all image-specific artifacts:
    1. Class names mapping
    2. Image statistics (if not already computed)

    Returns
    -------
    Dict[str, Path]
        Dictionary with paths to saved artifacts.
    """
    print_section_header("IMAGE FEATURE ENGINEERING")

    # Define and save class names
    class_names = {
        0: "good",
        1: "defective"
    }

    # Save artifacts
    save_image_artifacts(class_names)

    print(f"\nFeature engineering complete.")
    return {
        "class_names": CLASS_NAMES_PATH,
        "image_statistics": IMAGE_STATISTICS_PATH,
    }


if __name__ == "__main__":
    """
    Run image feature engineering module directly.

    This will create and save image-specific artifacts.

    Usage:
        python -m src.data.features
    """
    print("Running image feature engineering module...")
    try:
        # Create features (artifacts)
        artifacts = create_features()
        print(f"\nFeature engineering complete.")
        print(f"Artifacts saved:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
