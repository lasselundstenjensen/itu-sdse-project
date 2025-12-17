from __future__ import annotations
import pandas as pd
from pathlib import Path
import datetime
import json
import numpy as np
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
import joblib
from loguru import logger
import argparse


# Internal imports
from mlops_pipeline.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PRINTING_STATE, MODELS_DIR


# Two helper functions for the data preparation.
def describe_numeric_col(x):
    """
    Calculates various descriptive stats for a numeric column in a dataframe.

    Input:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Imputes the mean/median for numeric columns or the mode for other types.


    Input:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    Output:
        x (pd.Series): Pandas series with added mean/median
    """
    if method not in ["mean", "median"]:
        raise ValueError("Method must be either mean or median.")

    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x



def make_dataset(raw_csv_path: Path, out_training_csv_path: Path, min_date: str, max_date: str | None, printing: bool) -> None:
    '''
    Function for importing  the raw dataset and applying filters to have the training data ready for feature building.
    
    Input:
        raw_csv_path          - The pathlib Path to the raw csv
        out_training_csv_path - The pathlib Path to where we save the cleaned dataset, used for feature building
        min_date              - Minimum str date to include from the raw dataset. Can not be empty.
        max_date              - Maximum str date to include from the raw dataset. If left empty, will set to max date in dataset.
    
    Output:
        No direct output, but files created for features/build_features.py
    '''
    logger.info(f"Reading raw data: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    # Apply the date filtering
    if "date_part" in df.columns:
        df["date_part"] = pd.to_datetime(df["date_part"]).dt.date
        min_d = pd.to_datetime(min_date).date()
        max_d = pd.to_datetime(max_date).date() if max_date else pd.to_datetime(datetime.datetime.now().date()).date()

        # Time limit data
        #data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
        #data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]


        df = df[(df["date_part"] >= min_d) & (df["date_part"] <= max_d)]
        date_limits = {"min_date": str(df["date_part"].min()), "max_date": str(df["date_part"].max())}
        # We dump the json file of the date limits in the interim data folder. (will create folder if it is absent)
        (INTERIM_DATA_DIR / "metadata").mkdir(parents=True, exist_ok=True)
        with open(INTERIM_DATA_DIR / "metadata" / "date_limits.json", "w") as f:
            json.dump(date_limits, f, indent=2)

    # Drop columns only if they exist
    cols_to_drop = ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    #Removing columns that will be added back after the EDA (MATTI: Don't think this is ever used again..? )
    cols_to_drop2 = ["domain", "country", "visited_learn_more_before_booking", "visited_faq"]
    cols_to_drop2 = [col for col in cols_to_drop2 if col in df.columns]
    if cols_to_drop2:
        df = df.drop(cols_to_drop2, axis=1)
    # END OF CHANGE

    # Fill up empty fields with NaN
    for col in ["lead_indicator", "lead_id", "customer_code"]:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan)
    # We remove rows where lead_indicator or lead_id is NaN
    df = df.dropna(axis=0, subset=[c for c in ["lead_indicator", "lead_id"] if c in df.columns])

    # We want to make sure the data in these columns is understood as pandas objects (so not continuous or time-based)
    obj_cols = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in obj_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # Now we can take all the numeric columns, and then capping the outlier values to within mean +/- 2 standard deviations. 
    cont = df.select_dtypes(include=[np.number]).copy()
    cont = cont.apply(lambda x: x.clip(lower=x.mean() - 2*x.std(), upper=x.mean() + 2*x.std()))  # x here is a pandas Series for each column
    # And then for the missing values, we add the mean (would be mode if categorical)
    cont = cont.apply(impute_missing_values)

    # Now we take all the object columns as defined above.
    cat = df.select_dtypes(include=["object"]).copy()
    if "customer_code" in cat.columns:
        cat.loc[cat["customer_code"].isna(), "customer_code"] = "None"  # For customer_code specifically, we set missing values to None
    # For NaN in object columns, we assign the mode (most frequent category for that column)
    cat = cat.apply(impute_missing_values)

    # Now we recombine the whole thing.
    df_clean = pd.concat([cat.reset_index(drop=True), cont.reset_index(drop=True)], axis=1)

    # This we can output as the training data set for feature building.
    out_training_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_training_csv_path, index=False)

    # We save the columns as metadata, to test for column drift in case we get new data in the future.
    with open(INTERIM_DATA_DIR / "metadata" / "columns_drift.json", "w") as f:
        json.dump(list(df_clean.columns), f, indent=2)

    # As always, we can print.
    if printing:
        logger.info(f"Interim saved: {out_training_csv_path} (rows={len(df_clean)}, cols={df_clean.shape[1]})")

'''
    DEPRECATED MAIN FUNCTIONALITY (now we deal with these scripts as modules)

def main():
    # We use the parser here so that code can be configurable from the terminal. Defaults set as we have them in the original script.
    parser = argparse.ArgumentParser(description="Prepares and makes the dataset for feature building from the raw_data.csv")
    parser.add_argument("--raw", type=Path, default=RAW_DATA_DIR / "raw_data.csv", help="Sets the path to the raw_data.csv. Default is in data/raw")
    parser.add_argument("--out", type=Path, default=INTERIM_DATA_DIR / "training_data.csv", help="Sets the path for the output of the data cleaning. Default is in data/interim")
    parser.add_argument("--min-date", type=str, default="2024-01-01", help="Sets the minimum date to be handled (Using format YYYY-MM-DD). Must be defined, default is 2024-01-01")
    parser.add_argument("--max-date", type=str, default="2024-01-31", help="Sets the maximum date to be handled (Using format YYYY-MM-DD). Defaults to 2024-01-31, but if explicitly undefined, max will be set to the latest date in raw_data.csv")
    parser.add_argument("--printing", action="store_true", default=PRINTING_STATE, help="Sets the printing state for logger, to see which parts are running. Default set by flag in mlops_pipeline/config.py as False. If called by --printing, will be set to true.")
    args = parser.parse_args()
    make_dataset(args.raw, args.out, args.min_date, args.max_date, args.printing)

if __name__ == "__main__":
    main()
'''

