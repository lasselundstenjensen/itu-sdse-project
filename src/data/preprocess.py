"""
Data Preprocessing Module

Handles data cleaning, transformation, and preparation for modeling.
This includes feature selection, missing value handling, outlier treatment,
and standardization.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..config import (
    ARTIFACT_DIR,
    CAT_MISSING_IMPUTE_PATH,
    OUTLIER_SUMMARY_PATH,
    RANDOM_STATE,
    SCALER_PATH,
)
from ..utils import (
    check_columns_exist,
    check_dataframe_not_empty,
    describe_numeric_col,
    impute_missing_values,
    print_section_header,
)


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove irrelevant columns for modeling.

    Drops columns that are not needed for the ML model:
    - is_active, marketing_consent, first_booking, existing_customer, last_seen
    - domain, country, visited_learn_more_before_booking, visited_faq

    Parameters
    ----------
    data : pd.DataFrame
        Raw data with all columns.

    Returns
    -------
    pd.DataFrame
        Data with only relevant columns for modeling.
    """
    print("Selecting features...")

    # Drop columns that are not needed for modeling
    drop_cols_1 = [
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
    ]
    data = data.drop(drop_cols_1, axis=1)

    # Remove columns that will be added back after EDA
    drop_cols_2 = [
        "domain",
        "country",
        "visited_learn_more_before_booking",
        "visited_faq",
    ]
    data = data.drop(drop_cols_2, axis=1)

    print(f"Dropped {len(drop_cols_1) + len(drop_cols_2)} columns. Remaining: {data.shape[1]}")
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing rows with invalid values.

    Steps:
    1. Replace empty strings with NaN in key columns
    2. Drop rows with missing target (lead_indicator) or ID (lead_id)
    3. Filter to only "signup" source
    4. Print target distribution

    Parameters
    ----------
    data : pd.DataFrame
        Raw data with potential issues.

    Returns
    -------
    pd.DataFrame
        Cleaned data with valid values.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    print("Cleaning data...")

    # Check for required columns
    required_cols = ["lead_indicator", "lead_id", "customer_code", "source"]
    check_columns_exist(data, required_cols)

    # Replace empty strings with NaN
    data["lead_indicator"] = data["lead_indicator"].replace("", np.nan)
    data["lead_id"] = data["lead_id"].replace("", np.nan)
    data["customer_code"] = data["customer_code"].replace("", np.nan)

    # Drop rows with missing target or ID
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    # Filter by source
    data = data[data.source == "signup"]

    # Print target distribution
    result = data.lead_indicator.value_counts(normalize=True)
    print("\nTarget value distribution:")
    for val, n in zip(result.index, result):
        print(f"  {val}: {n:.4f}")

    return data


def create_categorical_columns(
    data: pd.DataFrame,
    cols: list[str] = None,
) -> pd.DataFrame:
    """
    Convert specified columns to categorical (object) type.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns to convert.
    cols : list[str], optional
        List of column names to convert. If None, uses default list.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns converted to object type.
    """
    if cols is None:
        cols = [
            "lead_id",
            "lead_indicator",
            "customer_group",
            "onboarding",
            "source",
            "customer_code",
        ]

    print("Creating categorical columns...")
    for col in cols:
        if col in data.columns:
            data[col] = data[col].astype("object")
            print(f"  Changed {col} to object type")

    return data


def separate_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate continuous and categorical columns.

    Continuous: float64 or int64 dtype
    Categorical: object dtype

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with mixed column types.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Continuous variables DataFrame
        - Categorical variables DataFrame
    """
    print("\nSeparating continuous and categorical columns...")

    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    print("\nContinuous columns:")
    for col in cont_vars.columns:
        print(f"  - {col}")

    print("\nCategorical columns:")
    for col in cat_vars.columns:
        print(f"  - {col}")

    return cont_vars, cat_vars


def handle_outliers(
    cont_vars: pd.DataFrame,
    z_threshold: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle outliers using z-score method (clipping).

    Clips values that are more than z_threshold standard deviations
    from the mean.

    Parameters
    ----------
    cont_vars : pd.DataFrame
        DataFrame with continuous variables.
    z_threshold : float, default=2.0
        Number of standard deviations from mean to use as threshold.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame with outliers clipped
        - Summary DataFrame with outlier statistics
    """
    print(f"\nHandling outliers (z-score threshold: {z_threshold})...")

    cont_vars = cont_vars.apply(
        lambda x: x.clip(
            lower=(x.mean() - z_threshold * x.std()),
            upper=(x.mean() + z_threshold * x.std()),
        )
    )

    # Calculate outlier summary
    outlier_summary = cont_vars.apply(describe_numeric_col).T

    # Save summary
    OUTLIER_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    outlier_summary.to_csv(OUTLIER_SUMMARY_PATH)
    print(f"Outlier summary saved to {OUTLIER_SUMMARY_PATH}")

    return cont_vars, outlier_summary


def impute_missing_values_data(
    cont_vars: pd.DataFrame,
    cat_vars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in continuous and categorical variables.

    For continuous: uses mean (configurable via IMPUTATION_METHOD)
    For categorical: uses mode, with special handling for customer_code

    Parameters
    ----------
    cont_vars : pd.DataFrame
        Continuous variables to impute.
    cat_vars : pd.DataFrame
        Categorical variables to impute.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Continuous variables with missing values imputed
        - Categorical variables with missing values imputed
    """
    print("\nImputing missing values...")

    # Save categorical missing imputation info (mode values)
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    CAT_MISSING_IMPUTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cat_missing_impute.to_csv(CAT_MISSING_IMPUTE_PATH)
    print(f"Categorical imputation modes saved to {CAT_MISSING_IMPUTE_PATH}")

    # Impute continuous variables
    cont_vars = cont_vars.apply(impute_missing_values)

    # Handle customer_code specially - fill NaN with 'None' string
    if "customer_code" in cat_vars.columns:
        cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"

    # Impute categorical variables
    cat_vars = cat_vars.apply(impute_missing_values)

    # Print missing value counts after imputation
    print("\nMissing values after imputation:")
    cont_missing = cont_vars.isnull().sum().sum()
    cat_missing = cat_vars.isnull().sum().sum()
    print(f"  Continuous: {cont_missing}")
    print(f"  Categorical: {cat_missing}")

    return cont_vars, cat_vars


def standardize_data(
    cont_vars: pd.DataFrame,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Standardize continuous variables using MinMaxScaler.

    Scales all continuous variables to [0, 1] range.
    Saves the fitted scaler for later use in inference.

    Parameters
    ----------
    cont_vars : pd.DataFrame
        Continuous variables to standardize.

    Returns
    -------
    tuple[pd.DataFrame, MinMaxScaler]
        - Standardized continuous variables
        - Fitted MinMaxScaler instance
    """
    print("\nStandardizing continuous variables...")

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    # Save scaler
    import joblib

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value=scaler, filename=SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Transform data
    cont_vars = pd.DataFrame(
        scaler.transform(cont_vars),
        columns=cont_vars.columns,
    )

    return cont_vars, scaler


def combine_data(
    cont_vars: pd.DataFrame,
    cat_vars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine continuous and categorical data back together.

    Resets indexes to ensure alignment before concatenation.

    Parameters
    ----------
    cont_vars : pd.DataFrame
        Standardized continuous variables.
    cat_vars : pd.DataFrame
        Imputed categorical variables.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all variables.
    """
    print("\nCombining continuous and categorical data...")

    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)

    # Categorical first, then continuous (as in original notebook)
    data = pd.concat([cat_vars, cont_vars], axis=1)

    print(f"Data combined. Shape: {data.shape}")
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data preprocessing pipeline.

    This function combines all preprocessing steps in sequence:
    1. Feature selection
    2. Data cleaning
    3. Create categorical columns
    4. Separate continuous and categorical
    5. Handle outliers
    6. Impute missing values
    7. Standardize continuous variables
    8. Combine data

    Parameters
    ----------
    data : pd.DataFrame
        Raw data to preprocess.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed data ready for feature engineering.
    """
    print_section_header("DATA PREPROCESSING")

    # Feature selection
    data = select_features(data)

    # Data cleaning
    data = clean_data(data)

    # Create categorical columns
    data = create_categorical_columns(data)

    # Separate columns
    cont_vars, cat_vars = separate_columns(data)

    # Handle outliers
    cont_vars, outlier_summary = handle_outliers(cont_vars)

    # Impute missing values
    cont_vars, cat_vars = impute_missing_values_data(cont_vars, cat_vars)

    # Standardize data
    cont_vars, scaler = standardize_data(cont_vars)

    # Combine data
    data = combine_data(cont_vars, cat_vars)

    check_dataframe_not_empty(data, "Preprocessed data")
    print(f"\nPreprocessing complete. Shape: {data.shape}")

    return data


if __name__ == "__main__":
    """
    Run data preprocessing module directly.

    This will load data from the default location and run preprocessing.

    Usage:
        python -m src.data.preprocess
    """
    from .fetch import load_raw_data

    print("Running data preprocessing module...")
    try:
        # Load data
        data = load_raw_data()
        print(f"Loaded raw data with shape: {data.shape}")

        # Preprocess
        data = preprocess_data(data)
        print(f"\nPreprocessing complete. Final shape: {data.shape}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
