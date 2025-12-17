from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from mlops_pipeline.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR, PRINTING_STATE
from mlops_pipeline.features.build_features import build_features

feature_app = typer.Typer(help="Feature engineering for handling importing and exporting a csv.")

# Let's define the CLI for doing the feature building (second part of 01_data.py)
@feature_app.command("build")
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



'''
#Original template command here, if we need it 
@feature_app.command("template demo")
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------

'''


if __name__ == "__main__":
    feature_app()
