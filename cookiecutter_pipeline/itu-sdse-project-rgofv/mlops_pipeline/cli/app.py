import typer
from pathlib import Path
from loguru import logger
import subprocess


#from mlops_pipeline.data.prepare import load_and_prepare_data
from mlops_pipeline.data.prepare02 import load_and_prepare_data
from mlops_pipeline.features_cli import feature_app
from mlops_pipeline.dataset import data_app
from mlops_pipeline.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, PRINTING_STATE, MODELS_DIR, BASE_DATA
#from mlops_pipeline.data.make_dataset import make_dataset
from mlops_pipeline.data.make_dataset02 import make_dataset
from mlops_pipeline.features.build_features import build_features
from mlops_pipeline.modeling.train import train_models
from mlops_pipeline.full_pipeline import run_full_pipeline

app = typer.Typer(help="Hi, and welcome to the RGoFV MLOps app! We are happy to see you here. \n This is a combined MLOps pipeline CLI. Commands are listed in order of operation.")
#app.add_typer(data_app, name="data")
#app.add_typer(feature_app, name="features")

@app.command("data")
def data(
    raw: Path = typer.Option(
        BASE_DATA / "raw_data.csv",
        help="Sets the path to the raw_data.csv. Default is in data/raw",
        #exists=True,
        readable=True,
    ),
    out: Path = typer.Option(
        INTERIM_DATA_DIR / "training_data.csv",
        help="Sets the path for the output of the data cleaning. Default is in data/interim",
    ),
    mindate: str = typer.Option(
        "2024-01-01",
        help="Sets the minimum date to be handled (Using format YYYY-MM-DD). Must be defined. Default is 2024-01-01",
    ),
    maxdate: str = typer.Option(
        "2024-01-31",
        help="Sets the maximum date to be handled (Using format YYYY-MM-DD). Defaults to 2024-01-31, but if explicitly undefined, max will be set to the latest date in raw_data.csv",
    ),
    printing_bool: bool = typer.Option(
        PRINTING_STATE,
        help="Sets the printing state for logger, to see which parts are running. Default set by flag in mlops_pipeline/config.py as False. If called by --printing, will be set to true.",
    ),

):
    '''
    Prepares and makes the dataset for feature building from the raw_data.csv
    '''
    subprocess.run(
        ["dvc", "pull"],
        check=True,
        cwd=BASE_DATA
    )
    logger.info("Converting the raw data...")
    make_dataset(
        raw_csv_path=raw,
        out_training_csv_path=out,
        min_date=mindate,
        max_date=maxdate,
        printing=printing_bool,
    )
    logger.success(f"Raw data has been converted and saved at {out}.")



@app.command("build")
def build_features_cli(
    interim: Path = typer.Option(
        INTERIM_DATA_DIR / "training_data.csv",
        help="Sets a pathlib Path to the cleaned dataset. Defaults to data/interim/training_data.csv",
        exists=True,
        readable=True,
    ),
    out: Path = typer.Option(
        PROCESSED_DATA_DIR / "training_features_gold.csv",
        help="Sets a pathlib Path for the outputted scaled feature dataset. Defaults to data/processed/training_features_gold.csv",
    ),
    printing_bool: bool = typer.Option(
        PRINTING_STATE,
        help="Sets the printing state for logger, to see which parts are running. Default set by flag in mlops_pipeline/config.py as False. If called by --printing, will be set to true.",
    ),
):
    '''
    Takes cleaned data from data/make_dataset.py and outputs the processed csv.
    '''
    logger.info("Building the features...")
    build_features(
        training_csv_path=interim,
        out_processed_csv_path=out,
        printing=printing_bool,
    )
    logger.success(f"Features have been written to {out}.")


@app.command("prepare")
def prepare(
    data_gold_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "train_data_gold.csv", 
        help="Sets the path for finding the gold data, as created by the build command. Default is processed/training_features_gold.csv" ),
    data_out_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        help="Directory to write the X/y train/test to. Defaults to data/interim.",
    ),
    scaler: Path = typer.Option(
        MODELS_DIR / "scaler.pkl",
        help="Sets a pathlib Path for saving the pickled MinMaxScaler, as trained on the training_data.csv. Defaults to models/scaler.pkl",
    ),
    testing_size: float = typer.Option(
        0.15,
        help="Sets the size of the test set. Default is 15% (0.15).",
    ),
    random_state_set: int = typer.Option(
        42,
        help="Sets the random state for the split. Default is 42.",
    ),
    printing_state: bool = typer.Option(
        PRINTING_STATE,
        help="Sets the printing state for the script. If set to True, prints as the script runs for troubleshooting.",
    ),
):
    '''
    Runs the data separation scripts in data/prepare.py
    '''
    logger.info("Making the test/train split...")
    load_and_prepare_data(
        data_gold_path=data_gold_path,
        out_dir=data_out_dir,
        scaler_path=scaler,
        test_size=testing_size,
        random_state=random_state_set,
        printing=printing_state,
    )
    logger.success(f"Split train/test-files have been created at {data_out_dir}.")
    logger.success(f"The MinMaxScaler has been pickled and saved at {scaler}.")


@app.command("train-model")
def train_model_cli(
    X_train_path: Path = INTERIM_DATA_DIR / "X_train.csv",
    X_test_path: Path = INTERIM_DATA_DIR / "X_test.csv",
    y_train_path: Path = INTERIM_DATA_DIR / "y_train.csv",
    y_test_path: Path = INTERIM_DATA_DIR / "y_test.csv",
    printing_bool: bool = typer.Option(
        PRINTING_STATE,
        help="Enable verbose logging",
    ),
):
    """
    Train models (XGBoost + Logistic Regression), log to MLflow,
    and save the best model.
    """
    train_models(
        X_train_path=X_train_path,
        X_test_path=X_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        printing_bool=printing_bool,
    )

@app.command("run-all")
def run_all():
    """
    Run the entire pipeline using subprocess calls.
    """
    run_full_pipeline()

if __name__ == "__main__":
    app()