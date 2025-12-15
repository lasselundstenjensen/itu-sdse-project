# src/data_ingestion.py

import os
import json
import datetime
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_FILE = DATA_DIR / "raw_data.csv"
FILTERED_BY_DATE_FILE = ARTIFACTS_DIR / "data_filtered_by_date.csv"
DATE_FILTER_METADATA_FILE = ARTIFACTS_DIR / "date_filter_metadata.json"


RAW_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Jeppe-T-K/itu-sdse-project-data/main/raw_data.csv"
)


def pull_with_dvc():
    try:
        subprocess.run(
            ["dvc", "pull"],
            check=True,
            cwd=PROJECT_ROOT
        )
        return True
    except subprocess.CalledProcessError:
        return False


def download_directly():
    print("Downloading raw dataset directly (CI fallback)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urlretrieve(RAW_DATA_URL, RAW_DATA_FILE)


def filter_dataset_by_date(df, min_date, max_date):
    df = df.copy()

    min_date = pd.to_datetime(min_date).date()
    max_date = pd.to_datetime(max_date).date()

    dates = pd.to_datetime(df["date_part"]).dt.date
    return df[(dates >= min_date) & (dates <= max_date)]

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Data ingestion started")

    pulled = pull_with_dvc()
    if not pulled:
        if "GITHUB_ACTIONS" in os.environ:
            download_directly()
        else:
            raise RuntimeError("DVC pull failed locally")

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError("raw_data.csv not available")

    data = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded dataset with {len(data)} rows")

    min_date = datetime.date(2024, 1, 1)
    max_date = datetime.date.today()

    mlflow.set_experiment("data-ingestion")

    with mlflow.start_run(run_name="date_filtering"):
        mlflow.log_param("min_date", min_date.isoformat())
        mlflow.log_param("max_date", max_date.isoformat())

        filtered = filter_dataset_by_date(data, min_date, max_date)

        filtered.to_csv(FILTERED_BY_DATE_FILE, index=False)
        mlflow.log_artifact(FILTERED_BY_DATE_FILE)

        metadata = {
            "requested_min_date": min_date.isoformat(),
            "requested_max_date": max_date.isoformat(),
            "actual_min_date": str(filtered["date_part"].min()),
            "actual_max_date": str(filtered["date_part"].max()),
            "rows_after_filtering": len(filtered),
        }

        with open(DATE_FILTER_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(DATE_FILTER_METADATA_FILE)

    if True:
        print("Data ingestion completed successfully")


if __name__ == "__main__":
    main()

