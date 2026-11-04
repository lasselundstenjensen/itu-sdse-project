"""Training pipeline entry point.                             Owner: Person B

To be extracted from notebooks/01_training.ipynb:
- cell 8:  MLflow tracking setup (local SQLite store)
- cell 27: train_cnn_model (weighted BCE, ReduceLROnPlateau, early stopping on macro F1)
- cell 29: tune_threshold
- cell 31: run orchestration + MLflow params
- cell 37: save checkpoint / results, log pyfunc model
- cell 39: register model, Staging transition, `best` alias

CLI: `python -m vial_classifier.modeling.train` runs the full training job.
Behaviour (seeds, order of operations) must match the notebook exactly.
"""

import typer

app = typer.Typer()


@app.command()
def main():
    """Run the training job: data -> train -> tune -> evaluate -> log -> register."""
    raise NotImplementedError("TODO(Person B): port notebook 01")


if __name__ == "__main__":
    app()
