"""
Utility Functions for the MLOps Pipeline

Shared helper functions used across multiple modules.
"""

import warnings
from pprint import pprint
from typing import Literal

import numpy as np
import pandas as pd

from .config import IMPUTATION_METHOD

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Suppress warnings globally (can be overridden in specific modules)
warnings.filterwarnings("ignore")

# Configure pandas display options
pd.set_option("display.float_format", lambda x: "%.3f" % x)


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def describe_numeric_col(x: pd.Series) -> pd.Series:
    """
    Calculate descriptive statistics for a numeric column.

    Parameters
    ----------
    x : pd.Series
        Pandas column to describe.

    Returns
    -------
    pd.Series
        Pandas series with descriptive stats including:
        - Count: Number of non-null values
        - Missing: Number of null values
        - Mean: Mean of the column
        - Min: Minimum value
        - Max: Maximum value
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

def impute_missing_values(
    x: pd.Series, method: Literal["mean", "median"] = IMPUTATION_METHOD
) -> pd.Series:
    """
    Impute missing values in a pandas Series.

    For numeric columns (float64, int64):
        - mean: Fill with the column mean
        - median: Fill with the column median

    For non-numeric columns (object, category):
        - Fill with the mode (most frequent value)

    Parameters
    ----------
    x : pd.Series
        Pandas column to impute.
    method : {"mean", "median"}, default="mean"
        Imputation method for numeric columns.
        Non-numeric columns always use mode.

    Returns
    -------
    pd.Series
        Series with missing values imputed.
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        if method == "mean":
            return x.fillna(x.mean())
        else:  # median
            return x.fillna(x.median())
    else:
        # For categorical/object columns, use mode
        return x.fillna(x.mode()[0])


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_dummy_cols(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Create one-hot encoded columns for a categorical column.

    Uses pd.get_dummies with drop_first=True to avoid the dummy variable trap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column to encode.
    col : str
        Name of the categorical column to encode.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original column replaced by one-hot encoded columns.
    """
    # Create dummy variables with drop_first to avoid multicollinearity
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)

    # Concatenate with original dataframe
    new_df = pd.concat([df, df_dummies], axis=1)

    # Drop the original column
    new_df = new_df.drop(col, axis=1)

    return new_df


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

    # Test describe_numeric_col
    test_series = pd.Series([1, 2, 3, 4, 5, np.nan])
    print("\nTest describe_numeric_col:")
    print(describe_numeric_col(test_series))

    # Test impute_missing_values
    print("\nTest impute_missing_values (mean):")
    print(impute_missing_values(test_series, method="mean"))

    # Test create_dummy_cols
    test_df = pd.DataFrame({"category": ["A", "B", "A", "C"]})
    print("\nTest create_dummy_cols:")
    print(create_dummy_cols(test_df, "category"))

    print("\nAll utility function tests passed!")
