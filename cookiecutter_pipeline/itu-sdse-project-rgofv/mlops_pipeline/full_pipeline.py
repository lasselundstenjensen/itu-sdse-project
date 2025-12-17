import subprocess
from loguru import logger
import sys


def run_full_pipeline():
    # We have to set the current .venv python
    py_venv = sys.executable
    commands = [
        [py_venv, "-m", "mlops_pipeline.cli.app", "data"],
        [py_venv, "-m", "mlops_pipeline.cli.app", "build"],
        [py_venv, "-m", "mlops_pipeline.cli.app", "prepare"],
        [py_venv, "-m", "mlops_pipeline.cli.app", "train-model"],
    ]

    for cmd in commands:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    logger.success("Full pipeline completed successfully")