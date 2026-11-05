"""Batch inference with the exported model.                    Owner: Person B

Loads the exported MLflow pyfunc model (config.EXPORT_DIR) and predicts on a
directory of images, using the same base64 contract as POST /invocations.
Used locally and by scripts/validate_model.py (CI model validation).

CLI: `python -m vial_classifier.modeling.predict --images-dir data/raw/test`
"""

from pathlib import Path

import typer

from vial_classifier.config import EXPORT_DIR, TEST_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    model_dir: Path = EXPORT_DIR,
    images_dir: Path = TEST_DATA_DIR,
):
    """Predict clean/defective for every image in images_dir."""
    raise NotImplementedError("TODO(Person B): load pyfunc export and predict")


if __name__ == "__main__":
    app()
