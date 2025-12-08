# imports
import pandas as pd
import os
import warnings
import datetime
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import utils
import sys


os.makedirs("output", exist_ok=True)

# we can have this step to ensure the correct formatting, but maybe not necessary

warnings.filterwarnings('ignore')


# to make sure the artifacts directory exists
os.makedirs("artifacts",exist_ok=True)

# load the raw data
data = pd.read_csv(sys.argv[1])
#print("Total rows:", data.count())
#display(data.head(5))


# Time limit the data
max_date = "2024-01-31"
min_date = "2024-01-01"
if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()

data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
with open("./data/interrim/date_limits.json", "w") as f:
    json.dump(date_limits, f)


# remove irrelevant columns
data = data.drop(
    [
        "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"
    ],
    axis=1
)



# data cleaning
data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)
data["customer_code"].replace("", np.nan, inplace=True)
data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])


# create categorical data columns
vars = [
    "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
]

for col in vars:
    data[col] = data[col].astype("object")
    print(f"Changed {col} to object type")


# separate categorical and continuous columns
cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
cat_vars = data.loc[:, (data.dtypes=="object")]


# deal with outliers
cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
outlier_summary = cont_vars.apply(utils.describe_numeric_col).T
outlier_summary.to_csv('./data/interrim/outlier_summary.csv')

# impute the data
cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
cat_missing_impute.to_csv("./data/interrim/cat_missing_impute.csv")

# continuous variables missing values
cont_vars = cont_vars.apply(utils.impute_missing_values)

# categorical variables missing values
cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
cat_vars = cat_vars.apply(utils.impute_missing_values)
cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T


# data standardisation
scaler_path = "./data/interrim/scaler.pkl"

scaler = MinMaxScaler()
scaler.fit(cont_vars)

joblib.dump(value=scaler, filename=scaler_path)
print("Saved scaler in artifacts")

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
#cont_vars


# combine data
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)


# create binary column of "source"
data['bin_source'] = data['source']
values_list = ['li', 'organic','signup','fb']
data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
mapping = {'li' : 'socials', 
           'fb' : 'socials', 
           'organic': 'group1', 
           'signup': 'group1'
           }

data['bin_source'] = data['source'].map(mapping)


# data columns drift
data_columns = list(data.columns)
with open('./data/interrim/columns_drift.json','w+') as f:           
    json.dump(data_columns,f)

# save the cleaned data
data.to_csv('./data/processed/train_data_gold.csv', index=False)