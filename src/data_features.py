import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FILTERED_BY_DATE_FILE = ARTIFACTS_DIR / "data_filtered_by_date.csv"
OUTLIER_SUMMARY_FILE = ARTIFACTS_DIR / "outlier_summary.csv"
CATEGORICAL_IMPUTATION_FILE = ARTIFACTS_DIR / "categorical_imputation_values.csv"
FEATURE_SCALER_FILE = ARTIFACTS_DIR / "feature_scaler.pkl"
FEATURE_COLUMNS_FILE = ARTIFACTS_DIR / "feature_columns.json"
MODEL_TRAINING_DATA_FILE = ARTIFACTS_DIR / "model_training_data.csv"
TRAINING_GOLD_DATA_FILE = ARTIFACTS_DIR / "training_data_gold.csv"



def _numeric_summary(series):
    return pd.Series(
        [
            series.count(),
            series.isna().sum(),
            series.mean(),
            series.min(),
            series.max(),
        ],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


def clean_base_data(df):
    df = df.copy()
    df = df.drop(columns=[
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
        "domain",
        "country",
        "visited_learn_more_before_booking",
        "visited_faq",
    ])

    cols = ["lead_indicator", "lead_id", "customer_code"]
    df[cols] = df[cols].replace("", np.nan)

    df = df.dropna(subset=["lead_indicator", "lead_id"])
    df = df[df.source == "signup"]
    return df


def split_feature_types(df):
    df = df.copy()
    categorical_cols = [
        "lead_id",
        "lead_indicator",
        "customer_group",
        "onboarding",
        "source",
        "customer_code",
    ]
    df[categorical_cols] = df[categorical_cols].astype("object")

    continuous = df.select_dtypes(include="number")
    categorical = df.select_dtypes(include=["object"])

    return categorical, continuous


def cap_outliers(continuous, n_std=2):
    def cap_series(x):
        mean = x.mean()
        std = x.std()
        return x.clip(mean - n_std * std, mean + n_std * std)

    capped = continuous.apply(cap_series)

    capped.apply(_numeric_summary).T.to_csv(
        OUTLIER_SUMMARY_FILE, index=False
    )
    return capped

def impute_series(series, numeric_strategy="mean"):

    if series.dtype in ("float64", "int64"):
        if numeric_strategy == "median":
            return series.fillna(series.median())
        return series.fillna(series.mean())

    return series.fillna(series.mode().iloc[0])


def impute_features(categorical, continuous):
    categorical = categorical.copy()
    continuous = continuous.copy()

    categorical.mode(dropna=True).to_csv(
        CATEGORICAL_IMPUTATION_FILE, index=False
    )

    continuous = continuous.apply(impute_series)

    categorical["customer_code"] = (
    categorical["customer_code"].fillna("None")
)
    categorical = categorical.apply(impute_series)

    return categorical, continuous


def scale_continuous_features(continuous):
    continuous = continuous.copy()

    scaler = MinMaxScaler()
    scaler.fit(continuous)
    joblib.dump(scaler, FEATURE_SCALER_FILE)

    return pd.DataFrame(
        scaler.transform(continuous),
        columns=continuous.columns,
    )


def combine_and_record_columns(categorical, continuous):
    categorical = categorical.copy()
    continuous = continuous.copy()

    data = pd.concat(
        [
            categorical.reset_index(drop=True),
            continuous.reset_index(drop=True),
        ],
        axis=1,
    )

    with open(FEATURE_COLUMNS_FILE, "w") as f:
        json.dump(list(data.columns), f)

    data.to_csv(MODEL_TRAINING_DATA_FILE, index=False)
    return data


def bin_source_feature(df):
    df = df.copy()

    mapping = {
        "li": "socials",
        "fb": "socials",
        "organic": "group1",
        "signup": "group1",
    }
    df["bin_source"] = df["source"].map(mapping).fillna("Others")
    return df


def run_feature_engineering(df): 
    df = df.copy()
    
    df = clean_base_data(df)
    categorical, continuous = split_feature_types(df)
    continuous = cap_outliers(continuous)
    categorical, continuous = impute_features(categorical, continuous)
    continuous = scale_continuous_features(continuous)
    combined = combine_and_record_columns(categorical, continuous)
    final = bin_source_feature(combined)

    final.to_csv(TRAINING_GOLD_DATA_FILE, index=False)
    return final


if __name__ == "__main__":
    raw = pd.read_csv(FILTERED_BY_DATE_FILE)
    run_feature_engineering(raw)
