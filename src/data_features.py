import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_imputation import impute_series
from paths import (
    DATE_FILTERED_DATA_FILE,
    OUTLIER_SUMMARY_FILE,
    CAT_MISSING_IMPUTE_FILE,
    SCALER_FILE,
    COLUMNS_DRIFT_FILE,
    TRAINING_DATA_FILE,
    TRAIN_GOLD_FILE,
)


def _numeric_summary(series):
    return pd.Series(
        [
            series.count(),
            series.isna().sum()
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


def cap_outliers(continuous):
    def cap_series(x):
        mean = x.mean()
        std = x.std()
        return x.clip(mean - n_std * std, mean + n_std * std)

    capped = continuous.apply(cap_series)

    capped.apply(_numeric_summary).T.to_csv(
        OUTLIER_SUMMARY_FILE, index=False
    )
    return capped


def impute_features(categorical, continuous):
    categorical = categorical.copy()
    continuous = continuous.copy()

    categorical.mode(dropna=True).to_csv(
        CAT_MISSING_IMPUTE_FILE, index=False
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
    joblib.dump(scaler, SCALER_FILE)

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

    with open(COLUMNS_DRIFT_FILE, "w") as f:
        json.dump(list(data.columns), f)

    data.to_csv(TRAINING_DATA_FILE, index=False)
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

    final.to_csv(TRAIN_GOLD_FILE, index=False)
    return final


if __name__ == "__main__":
    raw = pd.read_csv(DATE_FILTERED_DATA_FILE)
    run_feature_engineering(raw)
