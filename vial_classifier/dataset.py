"""Data acquisition and loading.                              Owner: Person A

To be extracted from notebooks/01_training.ipynb:
- cells 10-11: S3 client, download images listed in the metadata CSVs
               (`dvc pull` fetches the metadata CSVs first)
- cell 13:     load_image_dataset -> list of (image_path, label)
- cell 22:     GlassVialDataset (torch Dataset)
- cell 23:     split_dataset (stratified 70/15/15, seeded)
- cell 24:     create_dataloaders

CLI: `python -m vial_classifier.dataset` downloads the training and test images.
"""

import typer

app = typer.Typer()


@app.command()
def main():
    """Download the training and held-out test images from the object store."""
    raise NotImplementedError("TODO(Person A): port notebook 01 cells 10-11")


if __name__ == "__main__":
    app()
