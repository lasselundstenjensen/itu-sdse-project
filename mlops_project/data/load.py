import pandas as pd
import subprocess
import json
import os
from mlops_project.config import (
    RAW_DATA_PATH, MIN_DATE, MAX_DATE, ARTIFACTS_DIR
)
def pull_data():
    """Pulling the data with dvc"""
    subprocess.run(["dvc", "pull"], check=True)



def load_raw_data() -> pd.DataFrame:
    """
    Loading raw csv data into pandas dataframe.
    
    """
    data = pd.read_csv(RAW_DATA_PATH)

    min_date = pd.to_datetime(MIN_DATE).date()
    max_date = pd.to_datetime(MAX_DATE).date()

    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    date_limits = {
        "min_date": str(data["date_part"].min()),
        "max_date": str(data["date_part"].max())
    }
    with open(f"{ARTIFACTS_DIR}/date_limits.json", "w") as f:
        json.dump(date_limits, f)
    
    return data
