"""Model selection, promotion and export.                     Owner: Person B

To be extracted from notebooks/02_deployment.ipynb:
- cells 5-6:   pick best run by test F1
- cells 9-11:  compare with the current Production model
- cell 13:     promote via Staging -> Production (+ alias, tags)
- cell 15:     export the winning pyfunc model to config.EXPORT_DIR + version.json

CLI: `python -m vial_classifier.modeling.registry` promotes and exports the model.
The exported directory is what CI uploads as the `model` artifact.
"""

import typer

app = typer.Typer()


@app.command()
def main():
    """Select, promote and export the best model to models/model."""
    raise NotImplementedError("TODO(Person B): port notebook 02 cells 5-15")


if __name__ == "__main__":
    app()
