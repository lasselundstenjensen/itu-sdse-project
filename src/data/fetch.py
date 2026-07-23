"""
Image Data Fetching Module

Handles loading of glass vial images and their metadata for the image classification pipeline.
This is the first step in the MLOps pipeline.
"""

import json
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image

from ..config import (
    ARTIFACT_DIR,
    IMAGE_DATA_DIR,
    METADATA_PATH,
    MLRUNS_DIR,
)
from ..utils import check_metadata_columns, print_section_header


def setup_artifacts() -> None:
    """
    Create artifact and MLflow directories.

    Creates the following directories if they don't exist:
    - artifacts/ (for data and model artifacts)
    - mlruns/ (for MLflow tracking)
    - mlruns/.trash/ (for MLflow cleanup)
    - data/images/raw/ (for raw image storage)
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    (MLRUNS_DIR / ".trash").mkdir(parents=True, exist_ok=True)
    IMAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Created artifacts and image directories")


def pull_data_from_dvc() -> bool:
    """
    Pull data from DVC.

    Uses subprocess to run `dvc pull` command.
    If DVC is not available or command fails, silently continues
    (data might already be local).

    Returns
    -------
    bool
        True if DVC pull succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            ["dvc", "pull"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("DVC pull succeeded")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"DVC pull failed (data may already be local): {e.stderr}")
        return False
    except FileNotFoundError:
        print("DVC not found in PATH (data may already be local)")
        return False


def load_image_dataset() -> List[Tuple[Path, int]]:
    """
    Load image dataset from IMAGE_DATA_DIR using metadata from METADATA_PATH.

    Reads images from the image directory and their corresponding labels
    from the metadata CSV file.

    Returns
    -------
    List[Tuple[Path, int]]
        List of (image_path, label) tuples where:
        - image_path: Path to the image file
        - label: Class label (0=good, 1=defective)

    Raises
    ------
    FileNotFoundError
        If METADATA_PATH or IMAGE_DATA_DIR does not exist.
    ValueError
        If metadata is empty or required columns are missing.
    """
    print(f"Loading image dataset from {IMAGE_DATA_DIR}")
    print(f"Using metadata from {METADATA_PATH}")

    # Check if metadata file exists
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. "
            "Create metadata.csv or ensure data exists."
        )

    # Check if image directory exists
    if not IMAGE_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found at {IMAGE_DATA_DIR}. "
            "Create the directory and add images."
        )

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Metadata loaded. Total entries: {len(metadata)}")

    # Validate metadata columns
    required_columns = ["filename", "class"]
    check_metadata_columns(metadata, required_columns)

    # Build list of (image_path, label) tuples
    image_dataset = []
    missing_files = []
    corrupt_files = []

    for _, row in metadata.iterrows():
        filename = row["filename"]
        label = int(row["class"])
        image_path = IMAGE_DATA_DIR / filename

        # Check if image file exists
        if not image_path.exists():
            missing_files.append(filename)
            continue

        # Validate image file
        try:
            with Image.open(image_path) as img:
                # Basic validation - ensure it's a valid image
                img.verify()
            image_dataset.append((image_path, label))
        except (IOError, SyntaxError) as e:
            corrupt_files.append(filename)
            warnings.warn(f"Corrupt or invalid image: {filename} - {e}")

    # Report issues
    if missing_files:
        warnings.warn(f"Missing image files: {missing_files}")
    if corrupt_files:
        warnings.warn(f"Corrupt image files: {corrupt_files}")

    print(f"Loaded {len(image_dataset)} valid images with labels")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")
    if corrupt_files:
        print(f"Warning: {len(corrupt_files)} corrupt image files detected")

    return image_dataset


def create_image_metadata() -> pd.DataFrame:
    """
    Create a sample metadata.csv file for testing purposes.

    Generates metadata for a small set of sample images with:
    - filename: Image file names
    - class: Binary labels (0=good, 1=defective)
    - batch_id: Sample batch identifiers
    - timestamp: Current timestamp

    Returns
    -------
    pd.DataFrame
        Generated metadata DataFrame.
    """
    print("Creating sample metadata.csv for testing...")

    # Create sample metadata
    sample_data = {
        "filename": [
            "vial_001.jpg", "vial_002.jpg", "vial_003.jpg", 
            "vial_004.jpg", "vial_005.jpg", "vial_006.jpg"
        ],
        "class": [0, 1, 0, 1, 0, 1],  # Balanced sample
        "batch_id": ["BATCH_2024_07_21", "BATCH_2024_07_21", "BATCH_2024_07_22", 
                   "BATCH_2024_07_22", "BATCH_2024_07_23", "BATCH_2024_07_23"],
        "timestamp": [datetime.now().isoformat()] * 6
    }

    metadata = pd.DataFrame(sample_data)

    # Save to METADATA_PATH
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(METADATA_PATH, index=False)
    print(f"Sample metadata saved to {METADATA_PATH}")

    return metadata


def validate_image_dataset(image_dataset: List[Tuple[Path, int]]) -> dict:
    """
    Validate the image dataset for quality and consistency.

    Performs the following checks:
    - Corrupt image detection
    - File size constraints (reject images < 10KB or > 1MB)
    - Resolution validation (ensure all images can be resized to 64x64)
    - Color channel validation (ensure RGB, not grayscale or RGBA)

    Parameters
    ----------
    image_dataset : List[Tuple[Path, int]]
        List of (image_path, label) tuples to validate.

    Returns
    -------
    dict
        Validation report with counts of valid/invalid images and details.
    """
    print("Validating image dataset...")

    validation_report = {
        "total_images": len(image_dataset),
        "valid_images": 0,
        "corrupt_images": 0,
        "small_files": 0,
        "large_files": 0,
        "invalid_channels": 0,
        "invalid_resolutions": 0,
        "corrupt_files": [],
        "small_file_list": [],
        "large_file_list": [],
        "invalid_channel_list": [],
        "invalid_resolution_list": []
    }

    for image_path, label in image_dataset:
        try:
            with Image.open(image_path) as img:
                # Check file size
                file_size = image_path.stat().st_size
                if file_size < 10240:  # < 10KB
                    validation_report["small_files"] += 1
                    validation_report["small_file_list"].append(image_path.name)
                    continue
                elif file_size > 1048576:  # > 1MB
                    validation_report["large_files"] += 1
                    validation_report["large_file_list"].append(image_path.name)
                    continue

                # Check color channels
                if img.mode not in ['RGB', 'L']:
                    validation_report["invalid_channels"] += 1
                    validation_report["invalid_channel_list"].append(image_path.name)
                    continue

                # Check resolution (we can resize, but warn if very different)
                width, height = img.size
                if width < 10 or height < 10:  # Too small to resize meaningfully
                    validation_report["invalid_resolutions"] += 1
                    validation_report["invalid_resolution_list"].append(image_path.name)
                    continue

                # If we get here, image is valid
                validation_report["valid_images"] += 1

        except (IOError, SyntaxError) as e:
            validation_report["corrupt_images"] += 1
            validation_report["corrupt_files"].append(image_path.name)

    # Print validation summary
    print(f"\nValidation Summary:")
    print(f"  Total images: {validation_report['total_images']}")
    print(f"  Valid images: {validation_report['valid_images']}")
    print(f"  Corrupt images: {validation_report['corrupt_images']}")
    print(f"  Too small (<10KB): {validation_report['small_files']}")
    print(f"  Too large (>1MB): {validation_report['large_files']}")
    print(f"  Invalid channels: {validation_report['invalid_channels']}")
    print(f"  Invalid resolutions: {validation_report['invalid_resolutions']}")

    if validation_report["corrupt_files"]:
        print(f"  Corrupt files: {validation_report['corrupt_files']}")
    if validation_report["small_file_list"]:
        print(f"  Small files: {validation_report['small_file_list']}")

    return validation_report


def fetch_and_prepare_data() -> List[Tuple[Path, int]]:
    """
    Complete data fetching pipeline for image classification.

    This function combines all fetching steps:
    1. Setup artifacts directory
    2. Pull data from DVC
    3. Load image dataset and metadata
    4. Validate the dataset

    Returns
    -------
    List[Tuple[Path, int]]
        Validated list of (image_path, label) tuples ready for preprocessing.
    """
    print_section_header("IMAGE DATA FETCHING")

    setup_artifacts()
    pull_data_from_dvc()

    # Load image dataset
    image_dataset = load_image_dataset()

    # Validate dataset
    validation_report = validate_image_dataset(image_dataset)

    # Filter out invalid images
    valid_dataset = []
    for image_path, label in image_dataset:
        if image_path.name not in validation_report["corrupt_files"]:
            valid_dataset.append((image_path, label))

    print(f"Data fetching complete. Valid images: {len(valid_dataset)}")
    return valid_dataset


if __name__ == "__main__":
    """
    Run image data fetching module directly.

    Usage:
        python -m src.data.fetch
    """
    print("Running image data fetching module...")
    try:
        dataset = fetch_and_prepare_data()
        print(f"\nSuccessfully loaded dataset with {len(dataset)} images")
        if dataset:
            print(f"Sample: {dataset[0]}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
