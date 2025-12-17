## mathm: Helper functions.
import subprocess
import pandas as pd
import os
import datetime
import json
import warnings
import numpy as np
#from IPython.display import display
from pprint import pprint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib

print("=== SCRIPT STARTED ===")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current dir: {os.listdir('.')}")

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





def data_extraction():
    """
    Look for raw_data.csv without using DVC.
    We run from /repo/notebooks, so the correct local path is:
      - ./artifacts/raw_data.csv
    Also allow:
      - ../artifacts/raw_data.csv  (repo root artifacts)
      - RAW_DATA_PATH env var (absolute override)
    """
    import os
    import pandas as pd

    cwd = os.getcwd()  # should be /repo/notebooks
    candidates = [
        os.path.join(cwd, "artifacts", "raw_data.csv"),                # ./artifacts/raw_data.csv
        os.path.abspath(os.path.join(cwd, "..", "artifacts", "raw_data.csv")),  # ../artifacts/raw_data.csv
        os.environ.get("RAW_DATA_PATH", ""),                           # optional absolute path
    ]

    for path in candidates:
        if path and os.path.exists(path):
            print(f"✓ Loading data from: {path}")
            return pd.read_csv(path)

def data_preparation(data, printing = False):
    '''
    Data prep

    Input:

    
    Output:

    '''
    max_date = "2024-01-31"
    min_date = "2024-01-01"


    if not max_date:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date()

    # Time limit data
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)

    # AS FAR AS I REMEMBER, ONLY CHANGE BETWEEN ORIGINAL 01_data.py AND THIS SCRIPT ARE THESE LINES
    # The original script was:
    # data = data.drop(
    #     [
    #         "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"
    #     ],
    #     axis=1
    # )   
    # data = data.drop(
    #     ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
    #     axis=1
    # )
    # Therefore, below we have the same logic, but with the addition of checking if the columns exist
    # in the dataframe before dropping them to avoid keyerrors which was messing up dagger pipeline 
    # (although this doesn't affect the individual scripts if ran seperately without dagger)
    # Drop columns only if they exist
    cols_to_drop = ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"]
    cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    if cols_to_drop:
        data = data.drop(cols_to_drop, axis=1)

    #Removing columns that will be added back after the EDA (MATTI: Don't think this is ever used again..? )
    cols_to_drop2 = ["domain", "country", "visited_learn_more_before_booking", "visited_faq"]
    cols_to_drop2 = [col for col in cols_to_drop2 if col in data.columns]
    if cols_to_drop2:
        data = data.drop(cols_to_drop2, axis=1)
    # END OF CHANGE

    #Remove rows with empty target variable
    #Remove rows with other invalid column data
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]
    result=data.lead_indicator.value_counts(normalize = True)

    if printing:
        print("Target value counter")
        for val, n in zip(result.index, result):
            print(val, ": ", n)
        print(data)


    vars = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]

    for col in vars:
        data[col] = data[col].astype("object")
        if printing:
           print(f"Changed {col} to object type")

    
    # Separate categorical and continuous columns
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]

    if printing:
        print("\nContinuous columns: \n")
        pprint(list(cont_vars.columns), indent=4)
        print("\n Categorical columns: \n")
        pprint(list(cat_vars.columns), indent=4)


    ## NOW TO HANDLE THE OUTLIERS/NaN
    # Outliers:
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv('./artifacts/outlier_summary.csv')
    if printing:
        outlier_summary

    # Missing values:
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")
    if printing:
        cat_missing_impute

    # Continuous variables missing values using helper functions
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T
    if printing:
        cont_vars

    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T
    if printing:
        cat_vars


    ## Added a scaler for data standardisation
    scaler_path = "./artifacts/scaler.pkl"

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(value=scaler, filename=scaler_path)
    if printing:
        print("Saved scaler in artifacts")

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    if printing:
        cont_vars

    # Now we combine the data into one final dataframe
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    if printing:
        print(f"Data cleansed and combined.\nRows: {len(data)}")
        data

    ## We add the artifact drift columns for future use
    data_columns = list(data.columns)
    with open('./artifacts/columns_drift.json','w+') as f:           
        json.dump(data_columns,f)
        
    
    data.to_csv('./artifacts/training_data.csv', index=False)

    if printing:
        data.columns

    # We bin based on the Source column (But all the data in our dataset is 'signup')
    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li' : 'socials', 
            'fb' : 'socials', 
            'organic': 'group1', 
            'signup': 'group1'
            }

    data['bin_source'] = data['source'].map(mapping)
    


    return data


        
def data_analysis(data):
    '''
    A function for doing EDA. This shows the data structure

    Input:  
        data: a pandas dataframe

    Output:
        
    '''

    print("Total rows:", data.count())
    #print(data.head(5))



def main():
    os.makedirs("artifacts",exist_ok=True)
    warnings.filterwarnings('ignore')
    pd.set_option('display.float_format',lambda x: "%.3f" % x)
    



    #print("Created artifacts directory")
    df_data = data_extraction()

    # DO WE WANT THE EDA?
    data_analysis(df_data)
    df_data = data_preparation(df_data)

        # Finally we can save the gold data
    df_data.to_csv('./artifacts/train_data_gold.csv', index=False)
    print("Data extraction completed.")

if __name__ == "__main__":
    main()