# Training pipeline plan (notebook 01 → `vial_classifier`)

**Scope:** everything in `notebooks/01_training.ipynb`, from downloading the data to
registering the model in MLflow, plus a Docker container that runs that pipeline.

**Done when:** a single command (`make train`, or the container) downloads the data,
trains, evaluates, computes the drift baseline, logs the pyfunc model and registers it
in Staging with the `best` alias. The results must match the notebook.

**Rules from the assignment:** keep behaviour identical to the notebook: same seeds,
same order of operations, and the quirks too (e.g. `CLASS_NAMES = {0: "good", ...}`).
Don't improve the model. Do remove the noise: commented-out code, `print` banners,
legacy tabular code, pandas display options.

---

## Where each notebook cell goes

Cell numbers are 0-based positions in the notebook, markdown cells included.

| Cell   | Notebook content                                                                   | Target                                            |
| ------ | ---------------------------------------------------------------------------------- | ------------------------------------------------- |
| 2      | configuration constants                                                            | `config.py` ✅ done                             |
| 4      | `os.makedirs(...)`, `chdir`                                                    | created where needed in each step; no `chdir`   |
| 6      | pandas options, `warnings`                                                       | drop                                              |
| 8      | `setup_mlflow_tracking`                                                          | `modeling/train.py`                             |
| 10–11 | S3 client, `download_files_from_metadata`, `pull_data_from_dvc`, `!dvc pull` | `dataset.py`                                    |
| 13     | `load_image_dataset`                                                             | `dataset.py`                                    |
| 15     | class distribution (EDA)                                                           | `dataset.py`: log the counts with loguru        |
| 17     | `print_section_header` → drop; `calculate_image_statistics`                   | `features.py`                                   |
| 19     | `class_names.json`                                                               | `features.py`                                   |
| 21     | `define_transforms`                                                              | `features.py`                                   |
| 22     | `GlassVialDataset`                                                               | `dataset.py`                                    |
| 23     | `split_dataset`                                                                  | `dataset.py`                                    |
| 24     | `create_dataloaders` + `image_statistics.json`                                 | `dataset.py` / `features.py`                  |
| 26     | `SimpleCNN`, `device`                                                          | `modeling/model.py`                             |
| 27     | `train_cnn_model`                                                                | `modeling/train.py`                             |
| 28     | `evaluate_cnn_model`                                                             | `modeling/evaluate.py`                          |
| 29     | `tune_threshold`                                                                 | `modeling/train.py`                             |
| 31     | run orchestration + MLflow params                                                  | `modeling/train.py` → `main()`               |
| 33     | `compute_baseline_statistics` + MLflow metrics                                   | `modeling/drift_baseline.py`                    |
| 35     | `SimpleLoggingWrapper`                                                           | `modeling/wrapper.py`                           |
| 37     | save checkpoint/results, `log_model`                                             | `modeling/train.py`                             |
| 39     | `register_model`, `wait_until_ready`, Staging, `best` alias                  | `modeling/registry.py` (registration part only) |
| 40     | `end_run`, summary                                                               | `modeling/train.py` → `main()`               |
| 41     | legacy tabular models (commented out)                                              | drop                                              |

---

## Phase 0: Baseline (reference results)

- [X] 0.1 In VS Code: **File: Revert File** on the notebook (don't save the stale copy), then restart the kernel
- [X] 0.2 Run `01_training.ipynb` from start to finish with the new `data/raw/images/{train,test}` paths
- [X] 0.3 Write down the reference numbers below. Every later parity check compares against them.

| Metric                                         | Notebook result        |
| ---------------------------------------------- | ---------------------- |
| Train / val / test split sizes                 | 510 / 108 / 112        |
| Epochs run (early stopping?)                   | 3 (early stop)         |
| Best val macro F1                              | 0.4433                 |
| Optimal threshold                              | 0.025                  |
| Test accuracy / F1 / ROC-AUC                   | 0.7857 / 0.88 / 0.8537 |
| Prediction for `Clean_151_…` (true label 0) | 1 (raw output 0.5144)  |
| Prediction for `Glass_039_…` (true label 1) | 1 (raw output 0.5152)  |

The predictions come from the registered `cnn_model@best` pyfunc model. The wrapper uses its default
threshold of 0.5, not the tuned 0.025. Parity means matching these values, even though `Clean_151` is wrong.

---

## Phase 1: Data (`feat/data-pipeline`)

- [X] 1.1 `dataset.py`: `get_s3_client()` (unsigned, Hetzner endpoint from config)
- [X] 1.2 `dataset.py`: `download_files_from_metadata(metadata_path, output_dir, prefix)` (parallel, skips existing files)
- [X] 1.3 `dataset.py`: `pull_metadata()`, which runs `dvc pull` via `subprocess` instead of `!dvc pull`
- [X] 1.4 `dataset.py`: `main()` CLI = pull the metadata, then download train + test images (`python -m vial_classifier.dataset`)
- [X] 1.5 `dataset.py`: `load_image_dataset(metadata_path, image_dir)` → `[(path, label)]`, logs class distribution
- [X] 1.6 `dataset.py`: `GlassVialDataset`
- [X] 1.7 `dataset.py`: `split_dataset(...)` (stratified, seeded, **same shuffle calls in the same order**)
- [X] 1.8 `features.py`: `define_transforms()` → `(train_transforms, val_transforms)`
- [X] 1.9 `dataset.py`: `create_dataloaders(...)`, with transforms passed in (no globals)
- [X] 1.10 `features.py`: `calculate_image_statistics(loader)`, `save_class_names(path)`
- [X] 1.11 Tests (`tests/test_dataset.py`, `tests/test_features.py`), using a few tiny generated JPEGs in `tmp_path` and no network:
  - `load_image_dataset` skips missing files
  - split sizes 70/15/15 and the class balance is kept in each set
  - same seed → same split
  - transforms output `3×64×64`
  - `GlassVialDataset` returns `(tensor, label)`
- [X] 1.12 Check: `make data` downloads 730 + 10 images into `data/raw/images/{train,test}`

## Phase 2: Model, training, evaluation (`feat/training-core`)

- [X] 2.1 `modeling/model.py`: `SimpleCNN` + `get_device()`
- [X] 2.2 `modeling/train.py`: `setup_mlflow_tracking(experiment_name)`
- [X] 2.3 `modeling/train.py`: `train_cnn_model(...)`, with hyperparameters as arguments (defaults from config)
  - keep `torch.manual_seed` / `np.random.seed` at the start (image stats consume random numbers before training)
  - keep the MLflow logging guarded by `mlflow.active_run()`
- [X] 2.4 `modeling/train.py`: `tune_threshold(model, val_loader, ...)`
- [X] 2.5 `modeling/evaluate.py`: `evaluate_cnn_model(model, test_loader, threshold)`
- [X] 2.6 Tests (`tests/test_model.py`, `tests/test_train.py`):
  - forward pass: `(2,3,64,64)` → `(2,1)`
  - 2 epochs on ~8 fake images return a history and restore the best weights
  - `tune_threshold` picks the expected threshold on a hand-made case
  - `evaluate_cnn_model` returns all the metric keys

## Phases 3–6: one branch

Do Phases 3–6 on a single branch, `feat/training-pipeline`, with one commit per phase. Start it from `main`
once `refactor/data-layout`, `feat/data-pipeline` and `feat/training-core` are merged, because Phase 4 needs all of them.

## Phase 3: Artifacts and pyfunc wrapper

- [ ] 3.1 `modeling/drift_baseline.py`: `compute_baseline_statistics(...)` + `log_baseline_metrics(stats)`
- [ ] 3.2 `modeling/wrapper.py`: `SimpleLoggingWrapper`
  - ⚠️ `log_path` default must stay **relative** (`artifacts/inference_log.json`), not `config.INFERENCE_LOG_PATH`
    (absolute), or the training machine's path is pickled into the served model
- [ ] 3.3 `log_model(..., code_paths=["vial_classifier"])` so the exported model contains the package and loads
  anywhere (serving image, CI validator)
- [ ] 3.4 Tests (`tests/test_wrapper.py`, `tests/test_drift_baseline.py`):
  - wrapper accepts bytes / base64 / path / `{"instances": [...]}` and returns `predictions`, `confidences`, `raw_outputs`, `num_samples`
  - a bad input gives `None` instead of crashing
  - each request is appended to the log file
  - baseline JSON has all the keys the `/drift` endpoint reads

## Phase 4: Pipeline entry point and registration

- [ ] 4.1 `modeling/registry.py`: `wait_until_ready(...)`, `register_model(run_id) -> version` (Staging, tags, `best` alias)
- [ ] 4.2 `modeling/train.py`: `main()`, the full run in notebook order:
  seed → load → split → loaders → image stats → class names → start run → log params → train → tune →
  evaluate → baseline → save checkpoint + results → log pyfunc → register → end run
- [ ] 4.3 CLI options: `--epochs`, `--batch-size`, `--limit` (subset of images for smoke tests)
- [ ] 4.4 `make train` runs data + training
- [ ] 4.5 Slow integration test (`@pytest.mark.slow`): `--epochs 1 --limit 20` end to end, then load the logged pyfunc model and predict
- [ ] 4.6 **Parity check**: `make train` against the Phase 0 table. Investigate any difference before moving on.
- [ ] 4.7 Delete the `NotImplementedError` stubs that are now implemented

## Phase 5: Training container

- [ ] 5.1 `docker/Dockerfile.train`:
  - `python:3.13-slim` + uv
  - `COPY pyproject.toml uv.lock` → `uv sync --frozen --no-dev` (dependencies cached in their own layer; CPU torch)
  - `COPY vial_classifier/ .dvc/ data/**/*.dvc` → install the project
  - `ENTRYPOINT ["python", "-m", "vial_classifier.modeling.train"]`
- [ ] 5.2 `.dockerignore`: `.venv`, `.git`, data images, `artifacts/`, `mlflow-state/`, `models/`, notebooks
- [ ] 5.3 DVC without git in the container: run `dvc config core.no_scm true` in the image (the build context has no `.git`)
- [ ] 5.4 Outputs to mounted volumes: `artifacts/`, `mlflow-state/` (and `data/` as a cache so images aren't re-downloaded)
  `docker run --rm -v ./artifacts:/app/artifacts -v ./mlflow-state:/app/mlflow-state -v ./data:/app/data vial-train`
- [ ] 5.5 Smoke run: `docker run ... vial-train --epochs 1 --limit 20`
- [ ] 5.6 `make train-docker` target
- [ ] 5.7 Hand-off note for Person D (Dagger): image build context, command, args, env vars, output directories

## Phase 6: Docs and clean-up

- [ ] 6.1 README: "Training" section (local + container), artifacts produced, MLflow state
- [ ] 6.2 Keep the notebook as reference only; make sure nothing imports from it
- [ ] 6.3 `ruff` clean, all tests green

---

## Hand-offs to the team

| Who                        | Needs from me                                                                                                                                                | When          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| Notebook 02 / export owner | `mlflow-state/` layout, `MODEL_NAME`, `best` alias, run metric `test_f1_score`, `registry.py` shared (I add registration, they add promote/export) | after Phase 4 |
| Person C (serving / drift) | `drift_baseline_stats.json` format, wrapper predict contract, inference log format                                                                         | after Phase 3 |
| Person D (Dagger / CI)     | container command, args, output dirs; `--epochs`/`--limit` for fast CI runs                                                                              | after Phase 5 |

## Known gotchas

- Notebook comments can be wrong; the **code** is the reference, not the comments or markdown.
- `num_workers=0` in the DataLoaders: keep it for parity, even though the Dataset class is now importable.
- MLflow stage transitions (`Staging`) are deprecated in recent MLflow and only produce warnings. Keep them; the deployment side relies on them.
- `MLFLOW_ARTIFACT_LOCATION` in config is absolute. That's fine locally and in the container, but don't copy an
  `mlflow-state/` from one machine to another and expect the artifact paths to still resolve.
- Windows line endings: make sure new files are LF (consider a `.gitattributes` with `* text=auto eol=lf`).
