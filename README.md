# Vial image classification - MLOps project

A binary image classifier that flags defective glass vials coming off a production line. The project is a refactoring of two notebooks into a standard Python package with a CLI, tests, and an HTTP service that serves the model and reports drift.

The original task brief is kept in [`docs/ASSIGNMENT.md`](docs/ASSIGNMENT.md). The architecture diagram it refers to is [`docs/project-architecture.png`](docs/project-architecture.png).

## What the system does

1. **Train** a small CNN (`SimpleCNN`) on vial images and log runs to a local MLflow store (SQLite). The model is registered as `cnn_model` in the **Staging** stage.
2. **Promote** the best run by test F1-score. It moves to **Production** only if it beats the current Production model. The winning model is exported to `models/model/`, which is the directory that gets baked into the serving image.
3. **Serve** the exported model over HTTP. The service exposes `/health`, `/invocations`, `/infer`, `/version` and `/drift`.
4. **Detect drift**. Every prediction is logged. `/drift` runs two-sample Kolmogorov-Smirnov tests on the logged confidences and the predicted-class mix, against a baseline saved at training time.

## Repository layout

```
src/vial_classifier/
  config.py       hyperparameters, paths and remote locations (single source of truth)
  data.py         S3 download, dataset loading, stratified split, transforms, loaders
  model.py        SimpleCNN architecture
  train.py        training loop, threshold tuning, evaluation, drift baseline
  inference.py    MLflow pyfunc wrapper: preprocessing, prediction, request logging
  drift.py        KS-test drift detection (pure functions, no I/O side effects)
  pipelines.py    run_training() and run_promotion(): MLflow registry workflow
  serving.py      FastAPI front end and the MLflow scoring subprocess
  __main__.py     CLI: pull-data | train | promote | serve
tests/            unit tests and HTTP contract tests (pytest)
notebooks/        original Databricks notebooks, kept for reference (see mapping below)
infra/digitalocean/app.yaml   App Platform spec for the deployed endpoint
data/             DVC-tracked metadata; images are downloaded, not committed
docs/             assignment brief and architecture diagrams
```

### Mapping from the notebooks

| Notebook section | Now in |
| --- | --- |
| Configuration cells | `config.py` |
| Read data (DVC pull, S3 download) | `data.py` (`pull_data`, `download_images_from_metadata`) |
| Load dataset, class counts, split | `data.py` (`load_image_dataset`, `split_dataset`) |
| Transforms, datasets, loaders, image statistics | `data.py` |
| SimpleCNN | `model.py` |
| Training loop, threshold tuning, evaluation | `train.py` |
| Drift baseline | `train.py` (`compute_baseline_statistics`) |
| Logging wrapper | `inference.py` (`SimpleLoggingWrapper`) |
| Training run and model registration | `pipelines.py` (`run_training`) |
| Model selection, production comparison, promotion | `pipelines.py` (`select_best_run`, `should_promote`, `promote_version`) |
| Export serving model and version info | `pipelines.py` (`export_serving_model`) |
| Serving (FastAPI + scoring subprocess) | `serving.py` |
| Drift detection functions | `drift.py` |

The legacy tabular-model code at the end of the training notebook was dropped, since it was commented out and not part of the image pipeline.

## Setup

Requires Python 3.13.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the pipeline

All commands are run from the project root.

```bash
# 1. Fetch the DVC-tracked metadata and download the images it lists
python -m vial_classifier pull-data

# 2. Train, log to MLflow, and register the model in Staging
python -m vial_classifier train            # add --no-pull to skip step 1

# 3. Promote the best run if it beats Production, and export the served model
python -m vial_classifier promote

# 4. Serve the exported model on port 5001
python -m vial_classifier serve            # optional: --port, --model-dir
```

Outputs:

- `artifacts/` - model checkpoint, test results, class names, image statistics, drift baseline, and the inference request log
- `mlflow-state/` - MLflow run history (SQLite) and run artifacts
- `models/model/` - the exported serving model plus `version.json`

These directories are git-ignored because they are reproducible.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok"}` once the model is loaded. `503` with `"loading"` during startup, or `"error"` if the scorer has died. |
| POST | `/invocations` | Forwards the request to the MLflow scoring server. Body: `{"instances": [<base64 JPEG/PNG>]}`. |
| POST | `/infer` | Alias of `/invocations`. |
| GET | `/version` | Model version and training timestamp, read from `version.json`. |
| GET | `/drift?days=7` | Drift report over the requests logged in the last N days. |

Example:

```bash
B64=$(base64 -w0 data/test/some_image.jpg)
curl -s -X POST -H 'Content-Type: application/json' \
  -d "{\"instances\": [\"$B64\"]}" http://localhost:5001/infer
```

The response wraps MLflow's result, so predictions appear under `predictions.predictions`.

## Tests

```bash
pytest
```

The suite has 29 tests covering the promotion rule, the drift checks, the stratified split, the model's output shape and parameter count (checked against the notebook architecture), request preprocessing and logging in the wrapper, and the HTTP contract of the front end.

The full pipeline was also run end to end on a synthetic dataset of 120 images at the original 1600x768 resolution: training, promotion, export, and serving with `/health`, `/infer`, `/version` and `/drift` all returned the expected responses.

## Known issues and next steps

These are deliberately left for the next pass, so that behaviour stays identical to the notebooks until the refactor is verified.

- **Automation is not done yet.** There are no GitHub Actions workflows and no Dagger pipeline in Go. The CLI is designed so each stage can be one Dagger step: `pull-data`, `train`, `promote`, and then a container build and push to GHCR.
- **No Dockerfile yet.** The serving image needs `models/model/`, this package, and its dependencies. The `__OWNER__` and `__TAG__` placeholders in `infra/digitalocean/app.yaml` have to be filled in from the pushed image.
- **Request logs store the full image.** `SimpleLoggingWrapper` records the base64 payload as `image_path`, so `artifacts/inference_log.json` grows quickly. Logging a short identifier instead would fix this, but it changes the log format that drift reads.
- **Pickled wrapper needs the package.** MLflow serializes the wrapper by reference, so `vial_classifier` must be importable wherever the exported model is loaded.
- **MLflow APIs are partly deprecated.** `transition_model_version_stage` and the `artifact_path` argument emit warnings on MLflow 3. Moving to model aliases only would remove them.
- **Drift baseline is approximate.** The confidence check samples from a normal distribution fitted to the baseline mean and std. The sampling RNG is unseeded by default, so drift results can vary slightly between calls.
- **Real-data metrics not reproduced here.** The synthetic run only verifies that the pipeline works. Metrics on the real dataset still need to be checked against the original notebook's results.
