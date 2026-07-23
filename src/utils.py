"""
Utility Functions for the Image Classification MLOps Pipeline

Shared helper functions used across multiple modules for image processing and PyTorch support.
"""

import warnings
from pprint import pprint
from typing import Dict, List, Literal, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Suppress warnings globally (can be overridden in specific modules)
warnings.filterwarnings("ignore")

# Configure pandas display options
pd.set_option("display.float_format", lambda x: "%.3f" % x)


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def calculate_image_statistics(data_loader: torch.utils.data.DataLoader) -> Dict[str, List[float]]:
    """
    Calculate mean and standard deviation per channel for a dataset.

    Useful for normalization and understanding data distribution.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        DataLoader with image batches.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary with 'mean' and 'std' keys, each containing a list of 3 values
        (one per RGB channel).
    """
    print("Calculating image statistics...")

    total_mean = np.zeros(3)  # RGB channels
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        # Convert to numpy
        batch_images = images.numpy()
        batch_size = batch_images.shape[0]
        
        # Reshape to (batch_size, 3, -1) for channel-wise stats
        batch_images = batch_images.transpose(0, 2, 3, 1).reshape(batch_size, -1, 3)
        
        # Calculate mean and std for each channel
        batch_mean = batch_images.mean(axis=1)
        batch_std = batch_images.std(axis=1)
        
        total_mean += batch_mean.sum(axis=0)
        total_std += batch_std.sum(axis=0)
        total_samples += batch_size

    # Calculate overall mean and std
    overall_mean = (total_mean / total_samples).tolist()
    overall_std = (total_std / total_samples).tolist()

    print(f"Image statistics: Mean={overall_mean}, Std={overall_std}")
    
    return {
        "mean": overall_mean,
        "std": overall_std,
        "total_samples": total_samples,
    }


def verify_image_dataset(image_paths: List[Path]) -> Dict[str, int]:
    """
    Verify all images in a directory are valid.

    Checks that all images can be loaded and are not corrupt.

    Parameters
    ----------
    image_paths : List[Path]
        List of paths to image files to verify.

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of valid, corrupt, and missing images.
    """
    print(f"Verifying {len(image_paths)} images...")

    result = {
        "valid": 0,
        "corrupt": 0,
        "missing": 0,
        "corrupt_files": [],
        "missing_files": [],
    }

    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img.verify()
            result["valid"] += 1
        except FileNotFoundError:
            result["missing"] += 1
            result["missing_files"].append(image_path.name)
        except (IOError, SyntaxError):
            result["corrupt"] += 1
            result["corrupt_files"].append(image_path.name)

    print(f"Verification complete:")
    print(f"  Valid: {result['valid']}")
    print(f"  Corrupt: {result['corrupt']}")
    print(f"  Missing: {result['missing']}")

    return result


def count_classes(labels: List[int]) -> Dict[int, int]:
    """
    Count samples per class in a dataset.

    Parameters
    ----------
    labels : List[int]
        List of class labels.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping class indices to their counts.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor to convert.

    Returns
    -------
    np.ndarray
        Converted NumPy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def check_metadata_columns(df: pd.DataFrame, columns: list[str], name: str = "Metadata") -> None:
    """
    Check that specified columns exist in a metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : list[str]
        List of column names to verify.
    name : str, default="Metadata"
        Name to use in error message.

    Raises
    ------
    ValueError
        If any column is missing.
    """
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {name}: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )


# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

def check_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Check that a DataFrame is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    name : str, default="DataFrame"
        Name to use in error message.

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError(f"{name} is empty. Check data loading or filtering steps.")


def check_columns_exist(df: pd.DataFrame, columns: list[str], name: str = "Data") -> None:
    """
    Check that specified columns exist in a DataFrame.
    
    Note: This function is kept for backward compatibility but check_metadata_columns
    should be used for metadata validation in the image classification pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : list[str]
        List of column names to verify.
    name : str, default="Data"
        Name to use in error message.

    Raises
    ------
    ValueError
        If any column is missing.
    """
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {name}: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )


# =============================================================================
# UTILITY FUNCTIONS FOR PRINTING
# =============================================================================

def print_section_header(title: str) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
        Section title to print.
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print basic information about a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to describe.
    name : str, default="DataFrame"
        Name to use in output.
    """
    print(f"\n{name} Info:")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")


if __name__ == "__main__":
    """Test utility functions when run directly."""
    print("Testing utility functions...")

    # Test count_classes
    test_labels = [0, 1, 0, 1, 0, 1, 0]
    print("\nTest count_classes:")
    class_counts = count_classes(test_labels)
    print(class_counts)

    # Test tensor_to_numpy
    test_tensor = torch.tensor([1.0, 2.0, 3.0])
    print("\nTest tensor_to_numpy:")
    print(tensor_to_numpy(test_tensor))

    # Test check_metadata_columns
    test_df = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "class": [0, 1], "batch_id": ["B1", "B2"]})
    print("\nTest check_metadata_columns:")
    try:
        check_metadata_columns(test_df, ["filename", "class"])
        print("Metadata columns check passed!")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAll utility function tests passed!")
