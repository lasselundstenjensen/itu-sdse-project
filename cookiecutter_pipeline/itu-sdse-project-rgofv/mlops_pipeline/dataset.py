from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mlops_pipeline.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PRINTING_STATE, BASE_DATA
from mlops_pipeline.data.make_dataset import make_dataset

data_app = typer.Typer(help="Initial command for taking raw dataset and cleaning it, making it ready for feature building.")


@data_app.command("data_init")
def data_init(
    raw: Path = typer.Option(
        BASE_DATA / "raw_data.csv",
        help="Sets the path to the raw_data.csv. Default is in data/raw",
        exists=True,
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
    logger.info("Converting the raw data...")
    make_dataset(
        raw_csv_path=raw,
        out_training_csv_path=out,
        min_date=mindate,
        max_date=maxdate,
        printing=printing_bool,
    )
    logger.success(f"Raw data has been converted and saved at {out}.")



'''
## ORIGINAL APP KEPT IN CASE
@data_app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------




'''

if __name__ == "__main__":
    data_app()
