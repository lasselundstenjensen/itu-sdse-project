
# src/mlops_pipeline/features/build_features.py
from __future__ import annotations
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from mlops_pipeline.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PRINTING_STATE

def build_features(training_csv_path: Path, out_processed_csv_path: Path, printing: bool = PRINTING_STATE) -> None:
    df = pd.read_csv(training_csv_path)

    cont = df.select_dtypes(include=[np.number]).copy()
    cat = df.select_dtypes(include=["object"]).copy()

    df_feat = pd.concat([cat, cont], axis=1)

    # binning
    mapping = {"li": "socials", "fb": "socials", "organic": "group1", "signup": "group1"}
    if "source" in df_feat.columns:
        df_feat["bin_source"] = df_feat["source"].map(mapping).fillna("Others")

    out_processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_processed_csv_path, index=False)

    if printing:
        logger.info(f"Processed features: {out_processed_csv_path} (rows={len(df_feat)}, cols={df_feat.shape[1]})")
        logger.info(f"Scaler saved: {scaler_path}")



'''
    DEPRECATED MAIN FUNCTIONALITY (now we deal with these scripts as modules)
def main():
    parser = argparse.ArgumentParser(description="Function that takes cleaned data from data/make_dataset.py and creates the pickled Scaler model and outputs the processed csv.")
    parser.add_argument("--interim", type=Path, default=INTERIM_DATA_DIR / "training_data.csv", help="Sets a pathlib Path to the cleaned dataset. Defaults to data/interim/training_data.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DATA_DIR / "training_features_gold.csv", help="Sets a pathlib Path for the outputted scaled feature dataset. Defaults to data/processed/training_features_gold.csv")
    parser.add_argument("--scaler", type=Path, default=MODELS_DIR / "scaler.pkl", help="Sets a pathlib Path for saving the pickled Scaler, as trained on the training_data.csv. Defaults to models/scaler.pkl")
    parser.add_argument("--printing", action="store_true", default=PRINTING_STATE, help="Sets the printing state for logger, to see which parts are running. Default set by flag in mlops_pipeline/config.py as False. If called by --printing, will be set to true.")
    args = parser.parse_args()
    build_features(args.interim, args.out, args.scaler, args.printing)

if __name__ == "__main__":
    main()
'''