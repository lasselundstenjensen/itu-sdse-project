import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from mlops_project.config import (
    COLUMNS_TO_DROP_INITIAL, 
    COLUMNS_TO_DROP_EDA,
    SCALER_PATH,
    ARTIFACTS_DIR
)
from mlops_project.utils.helpers import describe_numeric_col, impute_missing_values


def drop_columns(data):
    """Deleting unused columns from data."""
    data = data.drop(COLUMNS_TO_DROP_INITIAL, axis=1)
    data = data.drop(COLUMNS_TO_DROP_EDA, axis=1)
    return data


def clean_data(data):
    """Cleaning empty targets and invalid rows."""
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]
    
    return data


def handle_outliers(cont_vars):
    """Clipping outliers with z-score."""
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(f"{ARTIFACTS_DIR}/outlier_summary.csv")
    return cont_vars


def impute_data(cont_vars, cat_vars):
    """Filling the values."""
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(f"{ARTIFACTS_DIR}/cat_missing_impute.csv")
    
    cat_vars.loc[cat_vars['customer_code'].isna(), 'customer_code'] = 'None'
    
    cat_vars = cat_vars.apply(impute_missing_values)

    cont_vars = cont_vars.apply(impute_missing_values)
    
    return cont_vars, cat_vars

def scale_data(cont_vars):
    """Normalize with MinMaxScaler"""
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(value=scaler, filename=SCALER_PATH)
    print("Saved scaler in artifacts")

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    return cont_vars


def create_bins(data):
    """Binning source"""
    data['bin_source'] = data['source']
    values_list = ['li', 'organic', 'signup', 'fb']
    data.loc[~data['source'].isin(values_list), 'bin_source'] = 'Others'
    
    mapping = {
        'li': 'socials', 
        'fb': 'socials', 
        'organic': 'group1', 
        'signup': 'group1'
    }
    data['bin_source'] = data['source'].map(mapping)
    
    return data



def preprocess(data):
    """Ana preprocessing pipeline"""
    data = drop_columns(data)
    data = clean_data(data)

    # categorical colums
    cat_cols = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in cat_cols:
        data[col] = data[col].astype("object")

    # seperate the data
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]


    # scale continuous vars
    cont_vars = scale_data(cont_vars)

    # combine the data
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    return data



