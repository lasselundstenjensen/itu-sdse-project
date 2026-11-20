# Stage 3: Integration testing and handover

## Scope

Integration tests cover the existing Python data, feature, and model-training modules. They use synthetic images created in a pytest temporary directory, so no production image download, S3 credentials, Docker, Go, or Dagger is required. The original notebook and CNN architecture are unchanged.

## Python environment

`pyproject.toml` specifies Python `~=3.13.0`. Install the project's runtime dependencies as agreed with the team, plus `requirements-dev.txt` for pytest. Confirm the actual runtime dependency lock or installation instructions with the team; the root development requirements alone do not list all imports used by the training pipeline (including torch, torchvision, numpy, pandas, Pillow, boto3, scikit-learn, and mlflow).

From the repository root:

```powershell
python -m pytest -q tests/test_integration.py
python -m pytest -q tests
```

## Training pipeline

Entry point: `python -m vial_classifier.modeling.train`

1. `pull_data_from_dvc()` updates DVC-tracked metadata and downloads images from the configured object store.
2. `load_image_dataset()` reads `data/images/metadata.csv` and images under `data/images/raw/`.
3. `split_dataset()` creates stratified training, validation, and test partitions.
4. `create_dataloaders()` applies image transforms and creates PyTorch DataLoaders.
5. `calculate_image_statistics()` measures training-image statistics.
6. `train_cnn_model()` trains the CNN; `tune_threshold()` selects the validation threshold; `evaluate_cnn_model()` evaluates the held-out test split.
7. The entry point saves artifacts and logs the model and metrics to MLflow.

The real training command downloads data and needs the project dependencies, DVC, object-store access, and writable output directories. Do not run it merely to execute the synthetic integration tests.

## Artifact paths and expected contents

| Path | Expected contents |
| --- | --- |
| `artifacts/vial_cnn_model.pth` | PyTorch checkpoint with `model_state_dict`, `architecture`, `input_shape`, `output_shape`, `threshold`, `num_classes` |
| `artifacts/model_results.json` | Accuracy, F1, precision, recall, ROC AUC, confusion matrix, classification report |
| `artifacts/class_names.json` | JSON mapping of `0` to `good` and `1` to `defective` (JSON keys are strings) |
| `artifacts/image_statistics.json` | `mean`, `std` (three channels each), `total_samples` |
| `artifacts/drift_baseline_stats.json` | **Needs confirmation with Person 2**: `compute_baseline_statistics()` exists, but the inspected training entry point does not save this file |

The above are expected schemas from code inspection, **not proof that current artifact files exist or pass verification**. Verify actual outputs with Person 2 or from a completed training run. Avoid committing large generated model weights unless the group explicitly requires it.

## Known issues / handover questions

- The training entry point calls `pull_data_from_dvc()` and is not a data-free smoke test.
- `drift_baseline_stats.json` is configured and the calculation function exists, but the inspected training entry point does not call it or write the file. Ask Person 2 whether to add baseline generation to training, and which dataset split should be used.
- `compute_baseline_statistics()` currently defaults to extracting the `fc1` layer. Coordinate expected baseline schema with the drift/inference implementation.
- Person 4 owns Go/Dagger orchestration. They can run `python -m pytest -q tests` as a test step and use the documented Python training entry point as the training step.

## Test evidence

## Test evidence

Environment:
- Windows
- Conda environment: mlops
- Python 3.13.18
- pytest 9.2.0

Command:
python -m pytest tests/test_integration.py -v

Result:
5 passed, 7 warnings in 59.33s

All five integration tests passed using synthetic image data.
No production dataset download was required.

Note: The generated model artifacts still need their
contents verified. The drift baseline artifact also
needs confirmation with Person 2.