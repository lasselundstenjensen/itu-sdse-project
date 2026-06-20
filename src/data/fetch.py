"""
Data Fetching Module

Handles data loading from DVC and local CSV sources.
This is the first step in the MLOps pipeline.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..config import (
    ARTIFACT_DIR,
    DATE_LIMITS_PATH,
    MAX_DATE,
    MIN_DATE,
    MLRUNS_DIR,
    RAW_DATA_PATH,
)
from ..utils import check_dataframe_not_empty, print_section_header


def setup_artifacts() -> None:
    """
    Create artifact and MLflow directories.

    Creates the following directories if they don't exist:
    - artifacts/ (for data and model artifacts)
    - mlruns/ (for MLflow tracking)
    - mlruns/.trash/ (for MLflow cleanup)
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    (MLRUNS_DIR / ".trash").mkdir(parents=True, exist_ok=True)
    print("Created artifacts directory")


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


def load_raw_data() -> pd.DataFrame:
    """
    Load raw training data from CSV.

    Returns
    -------
    pd.DataFrame
        Raw training data loaded from RAW_DATA_PATH.

    Raises
    ------
    FileNotFoundError
        If RAW_DATA_PATH does not exist.
    ValueError
        If loaded data is empty.
    """
    print(f"Loading training data from {RAW_DATA_PATH}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            "Run 'dvc pull' or ensure data exists."
        )

    data = pd.read_csv(RAW_DATA_PATH)
    print(f"Total rows: {data.shape[0]}")
    check_dataframe_not_empty(data, "Raw training data")

    return data


def filter_data_by_date(
    data: pd.DataFrame,
    min_date: str = None,
    max_date: str = None,
) -> tuple[pd.DataFrame, pd.Timedelta, pd.Timedelta]:
    """
    Filter data by date range and save date limits.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'date_part' column to filter.
    min_date : str, optional
        Minimum date in YYYY-MM-DD format. Uses MIN_DATE from config if None.
    max_date : str, optional
        Maximum date in YYYY-MM-DD format. Uses MAX_DATE from config if None.

    Returns
    -------
    tuple[pd.DataFrame, pd.Timedelta, pd.Timedelta]
        - Filtered DataFrame
        - Actual min_date used
        - Actual max_date used

    Raises
    ------
    ValueError
        If 'date_part' column is missing from data.
    """
    from ..config import MIN_DATE as config_min_date
    from ..config import MAX_DATE as config_max_date

    # Use provided dates or fall back to config
    if min_date is None:
        min_date = config_min_date
    if max_date is None:
        max_date = config_max_date

    # Convert to datetime.date objects
    import datetime

    if max_date == config_max_date and config_max_date == "2024-01-31":
        # Use current date if the default is still set
        actual_max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        actual_max_date = pd.to_datetime(max_date).date()

    actual_min_date = pd.to_datetime(min_date).date()

    # Check for date_part column
    if "date_part" not in data.columns:
        raise ValueError(
            "Column 'date_part' not found in data. "
            "Cannot filter by date."
        )

    # Convert date_part to date and filter
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[
        (data["date_part"] >= actual_min_date) &
        (data["date_part"] <= actual_max_date)
    ]

    # Save date limits for reproducibility
    date_limits = {
        "min_date": str(actual_min_date),
        "max_date": str(actual_max_date),
    }
    DATE_LIMITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATE_LIMITS_PATH, "w") as f:
        json.dump(date_limits, f, indent=2)

    print(f"Filtered data to date range: {actual_min_date} to {actual_max_date}")
    print(f"Date limits saved to {DATE_LIMITS_PATH}")

    return data, actual_min_date, actual_max_date


def fetch_and_prepare_data(
    min_date: str = None,
    max_date: str = None,
) -> pd.DataFrame:
    """
    Complete data fetching pipeline.

    This function combines all fetching steps:
    1. Setup artifacts directory
    2. Pull data from DVC
    3. Load raw data
    4. Filter by date range

    Parameters
    ----------
    min_date : str, optional
        Minimum date for filtering.
    max_date : str, optional
        Maximum date for filtering.

    Returns
    -------
    pd.DataFrame
        Loaded and filtered training data.
    """
    print_section_header("DATA FETCHING")

    setup_artifacts()
    pull_data_from_dvc()

    data = load_raw_data()
    data, actual_min, actual_max = filter_data_by_date(data, min_date, max_date)

    print(f"Data fetching complete. Shape: {data.shape}")
    return data


if __name__ == "__main__":
    """
    Run data fetching module directly.

    Usage:
        python -m src.data.fetch
    """
    print("Running data fetching module...")
    try:
        data = fetch_and_prepare_data()
        print(f"\nSuccessfully loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
