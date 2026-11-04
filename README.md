# Vial impurity classification - MLOps pipeline

ITU BDS MLOps 2026 exam project. A small CNN classifies images of pharmaceutical
vials as **clean (0)** or **defective (1)**; the model is trained, validated and
deployed to DigitalOcean App Platform by a Dagger (Go) pipeline orchestrated from
GitHub Actions.

- Assignment: [docs/TASK.md](docs/TASK.md)
- Dataset: [docs/pharmaceutical-vial-dataset.md](docs/pharmaceutical-vial-dataset.md)
- Original monolithic notebooks (reference only): [notebooks/](notebooks/)

> Work in progress - sections marked TODO are filled in as the pipeline lands.

## Project organization

Based on the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) template.

```
├── Makefile               <- Convenience commands (`make data`, `make train`, `make test`, ...)
├── README.md
├── pyproject.toml         <- Package metadata, dependencies, ruff/pytest config
├── uv.lock                <- Locked dependency versions (uv)
├── data
│   ├── external           <- Data from third-party sources
│   ├── interim            <- Intermediate transformed data
│   ├── processed          <- Final data sets for modeling
│   └── raw
│       ├── images         <- 730 training images; metadata.csv is DVC-tracked
│       └── test           <- 10 held-out test images; metadata.csv is DVC-tracked
├── docs                   <- Assignment, dataset documentation, diagrams
├── infra/digitalocean     <- App Platform spec for the serving container
├── models                 <- Exported MLflow pyfunc model (models/model), not in git
├── notebooks              <- Original training/deployment notebooks (reference)
├── references             <- Data dictionaries, manuals, explanatory material
├── reports/figures        <- Generated figures
├── tests                  <- pytest test suite
└── vial_classifier        <- Source code
    ├── config.py          <- Paths, data source, hyperparameters, serving settings
    ├── dataset.py         <- Download images, Dataset class, stratified split, loaders
    ├── features.py        <- Image transforms and dataset statistics
    ├── modeling
    │   ├── model.py           <- SimpleCNN architecture
    │   ├── train.py           <- Training entry point (MLflow logging + registration)
    │   ├── evaluate.py        <- Test-set evaluation
    │   ├── drift_baseline.py  <- Baseline statistics for drift detection
    │   ├── wrapper.py         <- MLflow pyfunc serving wrapper (inference logging)
    │   ├── registry.py        <- Model selection, promotion and export to models/model
    │   └── predict.py         <- Batch inference with the exported model
    ├── plots.py           <- Report figures (training curves, confusion matrix)
    ├── monitoring
    │   └── drift.py       <- KS-test drift detection over the inference log
    └── serving
        └── app.py         <- FastAPI app: /health, /invocations, /infer, /version, /drift
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                # create .venv and install all dependencies (incl. dev tools)
make data              # dvc pull the metadata, then download the images
make train             # train the CNN, log to mlflow-state/
make export            # promote the best model and export it to models/model
make test              # run the test suite
make lint              # ruff format check + lint
```

On Linux (CI, Dagger, Docker) torch is installed from the CPU-only PyTorch index.

## Pipeline

TODO: Dagger functions, GitHub workflows, model artifact and validation.

## Deployment and monitoring

TODO: Docker image, GHCR, DigitalOcean App Platform, endpoints, drift-triggered retraining.
