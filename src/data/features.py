"""
Feature Engineering Module

Handles feature transformations including binning and encoding.
This is the final step in the data pipeline before model training.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from ..config import (
    COLUMNS_DRIFT_PATH,
    TRAIN_DATA_GOLD_PATH,
)
from ..utils import check_dataframe_not_empty, print_section_header


def bin_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create binned versions of categorical columns.

    Specifically creates a 'bin_source' column that groups source values:
    - 'li' and 'fb' → 'socials'
    - 'organic' and 'signup' → 'group1'
    - All other values → 'Others'

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'source' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'bin_source' column.

    Raises
    ------
    ValueError
        If 'source' column is missing.
    """
    print("Binning categorical columns...")

    if "source" not in data.columns:
        raise ValueError("Column 'source' not found. Cannot create bin_source.")

    # Create bin_source column as copy of source
    data["bin_source"] = data["source"]

    # Define valid values and mapping
    values_list = ["li", "organic", "signup", "fb"]
    
    mapping = {
        "li": "socials",
        "fb": "socials",
        "organic": "group1",
        "signup": "group1",
    }

    # Mark non-listed values as 'Others'
    data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"

    # Apply mapping to valid values
    data["bin_source"] = data["source"].map(mapping)

    # Fill any remaining NaN values (from map) with 'Others'
    data["bin_source"] = data["bin_source"].fillna("Others")

    print(f"Created 'bin_source' column with values: {data['bin_source'].unique().tolist()}")
    return data


def save_artifacts(data: pd.DataFrame) -> None:
    """
    Save data artifacts for drift detection and training.

    Saves:
    - columns_drift.json: List of column names for data drift detection
    - train_data_gold.csv: The gold medallion dataset (fully processed training data)

    Parameters
    ----------
    data : pd.DataFrame
        Fully processed training data.

    Raises
    ------
    ValueError
        If data is empty.
    """
    check_dataframe_not_empty(data, "Training data")

    print("\nSaving data artifacts...")

    # Save columns for drift detection
    data_columns = list(data.columns)
    COLUMNS_DRIFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COLUMNS_DRIFT_PATH, "w") as f:
        json.dump(data_columns, f, indent=2)
    print(f"Columns list saved to {COLUMNS_DRIFT_PATH}")

    # Save training data (gold medallion)
    TRAIN_DATA_GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(TRAIN_DATA_GOLD_PATH, index=False)
    print(f"Training data saved to {TRAIN_DATA_GOLD_PATH}")
    print(f"  Shape: {data.shape}")


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    This function combines all feature engineering steps:
    1. Binning categorical columns
    2. Saving artifacts

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data.

    Returns
    -------
    pd.DataFrame
        Data with engineered features, ready for model training.
    """
    print_section_header("FEATURE ENGINEERING")

    # Binning
    data = bin_categorical_columns(data)

    # Save artifacts
    save_artifacts(data)

    check_dataframe_not_empty(data, "Feature engineered data")
    print(f"\nFeature engineering complete. Shape: {data.shape}")

    return data


if __name__ == "__main__":
    """
    Run feature engineering module directly.

    This will load preprocessed data and run feature engineering.

    Usage:
        python -m src.data.features
    """
    from .preprocess import preprocess_data
    from .fetch import load_raw_data

    print("Running feature engineering module...")
    try:
        # Load and preprocess data
        raw_data = load_raw_data()
        print(f"Loaded raw data with shape: {raw_data.shape}")

        preprocessed_data = preprocess_data(raw_data)
        print(f"Preprocessed data shape: {preprocessed_data.shape}")

        # Create features
        data = create_features(preprocessed_data)
        print(f"\nFeature engineering complete. Final shape: {data.shape}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
