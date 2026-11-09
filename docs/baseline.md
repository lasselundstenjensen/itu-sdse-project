# Baseline: original training notebook

Reference results from running the original `notebooks/01_training.ipynb` unchanged, before any refactoring. The refactored pipeline must reproduce these numbers: the task is to make the code "run as before", not to improve the model.

## How it was produced

From the project root, with the virtual environment active:

```
jupyter nbconvert --to notebook --execute notebooks/01_training.ipynb --output 01_training_baseline.ipynb
```

- Date: 2026-11-03
- Base commit: `24cdda5`
- Data: all 730 training images and 10 test images downloaded (`failed 0`)

### Environment

| Package | Version |
|---|---|
| OS | Windows 11 (CPU only) |
| Python | 3.13.15 |
| torch | 2.14.1+cpu |
| torchvision | 0.29.1+cpu |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| mlflow | 3.17.2 |
| pillow | 13.0.0 |
| dvc | 3.67.1 |

## Results

### Data split (stratified 70/15/15, seed 42)

| Set | Total | Class 0 (clean) | Class 1 (defective) |
|---|---|---|---|
| Train | 510 | 107 | 403 |
| Validation | 108 | 22 | 86 |
| Test | 112 | 24 | 88 |

### Training

- Loss `pos_weight`: 0.2655
- Early stopping after epoch 3 (no validation macro-F1 improvement for 2 epochs)
- Best validation macro-F1: 0.4433

| Epoch | Train loss | Train acc | Val loss | Val acc | Val macro-F1 |
|---|---|---|---|---|---|
| 1 | 0.3143 | 0.4098 | 0.2986 | 0.7963 | 0.4433 |
| 2 | 0.2919 | 0.7275 | 0.2981 | 0.7963 | 0.4433 |

### Test set (`artifacts/model_results.json`)

| Metric | Value |
|---|---|
| Optimal threshold | 0.025 |
| Accuracy | 0.7857 |
| F1 (defective) | 0.8800 |
| Precision | 0.7857 |
| Recall | 1.0000 |
| ROC-AUC | 0.8537 |
| Macro-F1 | 0.44 |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[0, 24], [0, 88]]` |

### Drift baseline (`artifacts/drift_baseline_stats.json`)

- Predictions on validation: 0 clean, 108 defective
- Confidence: mean 0.5144, std 0.0004, min 0.5138, max 0.5162

### Model checkpoint

- `artifacts/vial_cnn_model.pth` MD5: `fc7bae2cc82df8e5ab99e1a96bae6474`

## Notes

- **Deterministic locally.** Two consecutive runs with the same data produced byte-identical artifacts (all JSON files and the `.pth` checkpoint). Bitwise equality is not guaranteed on other machines (e.g. the Linux containers used by Dagger and GitHub Actions) or with other package versions, so the first containerised run must be compared against this file.
- **The model is degenerate.** It predicts "defective" for every image: confidences collapse into a narrow band around 0.514, and threshold tuning falls back to the first value tried (0.025). This is the behaviour of the original code with this data. It is kept as-is and not fixed as part of the refactor.
- **Sensitive to the exact data.** An earlier run where one image (`Sand_021.jpg`) failed to download (729 images) changed the shuffle and produced a non-degenerate model (threshold 0.5, test F1 0.929, ROC-AUC 0.973). Always check that the download step reports `failed 0` before comparing results.
