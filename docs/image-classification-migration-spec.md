# **Image Classification Migration Specification Document**

---

## **Overview**
This document specifies the **technical changes** required to migrate the existing MLOps pipeline from **tabular data classification (XGBoost)** to **image classification (PyTorch CNN)**. The goal is to enable **automated visual inspection of glass vials** for quality control.

---

## **Current State**
The current pipeline processes **tabular data (CSV files)** with:
- XGBoost and Logistic Regression models
- Feature engineering for categorical/continuous columns
- Standardization, outlier handling, missing value imputation
- MLflow tracking and model registry

---

## **Target State**
The new pipeline will process **image data** with:
- PyTorch CNN model for binary classification
- Image loading, resizing (to **64x64 RGB**), and preprocessing
- No tabular-specific preprocessing (outliers, imputation, dummy variables)
- Same MLflow tracking and model registry infrastructure

---

## **Problem Context**
- **Domain**: Pharmaceutical manufacturing quality control
- **Use Case**: Binary classification of glass vial images
  - **Input**: Images of produced glass vials (fixed size: **64x64 RGB**)
  - **Classes**:
    - `0`: Good quality (acceptable)
    - `1`: Not good quality (defective/rejected)
  - **Goal**: Automated visual inspection to classify vial quality
  - **NUM_CLASSES**: 2 (binary classification)

---
---
## **Data Specifications**

### **Image Properties**
| **Property**       | **Value**               | **Notes**                                  |
|--------------------|-------------------------|--------------------------------------------|
| **Format**         | JPEG or PNG             |                                            |
| **Resolution**     | Fixed **64x64**         | All images **must** be resized to this      |
| **Color Channels** | RGB                     |                                            |
| **Bit Depth**      | 8-bit                   | Standard for most cameras                  |

### **Dataset Statistics**
- **Expected Volume**: ~100 images/day
- **Class Distribution**:
  - Raw data: ~10% defective (class `1`), 90% good (class `0`)
  - **Training Set**: Resampled to **50/50 split** (balanced classes)
- **Train/Val/Test Split**: Standard ratios (e.g., 70%/15%/15%), **stratified**

### **Directory Structure**
**Option 2 (Single directory with metadata file)**:
```bash
/data/images/
├── raw/
│   ├── vial_001.jpg
│   ├── vial_002.jpg
│   └── ...
└── metadata.csv  # Columns: filename, class, batch_id, timestamp
```

### **Metadata**
| **Field**      | **Type**   | **Example**          | **Purpose**                     |
|----------------|------------|----------------------|---------------------------------|
| `filename`     | str        | `vial_001.jpg`       | Unique identifier               |
| `class`        | int        | `0` or `1`           | Label (0=good, 1=defective)     |
| `batch_id`     | str        | `BATCH_2024_07_21`   | Traceability                    |
| `timestamp`    | datetime   | `2024-07-21T14:30:00`| Temporal analysis              |

### **Data Quality Checks**
Implement **automated checks** in code to validate:
- **Corrupt image detection** (PIL/OpenCV validation)
- **File size constraints** (e.g., reject images < 10KB or > 1MB)
- **Resolution validation** (ensure all images are **64x64** after resizing)
- **Color channel validation** (ensure RGB, not grayscale or RGBA)

### **Data Augmentation**
Use the following **PyTorch transforms** for training:
```python
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),               # Fixed size
    transforms.RandomHorizontalFlip(p=0.5),   # Mirror images
    transforms.RandomRotation(10),             # ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
- **Validation/Testing**: Only `Resize`, `ToTensor`, and `Normalize` (no random augmentations)

---

## **Model Architecture**
The **PyTorch CNN** must use the following **deterministic architecture** for consistency and validation:

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 64x64 -> 8x8 after 3 pools
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x
```

### **Training Configuration**
- **Loss**: `BCELoss` (Binary Cross Entropy Loss)
- **Optimizer**: `Adam` (default `lr=0.001`)
- **Metrics**: Accuracy, F1-score, Precision, Recall
- **Device**: Automatic detection (`cuda` if available, else `cpu`)
- **Determinism**: Set `torch.manual_seed(42)` and `numpy.random.seed(42)` for reproducibility

### **Threshold Selection**
- **Default Threshold**: `0.5` (output ≥ 0.5 → class `0` (good), else class `1` (defective))
- **Threshold Tuning**:
  - Evaluate thresholds `[0.3, 0.4, 0.5, 0.6, 0.7]` on the **validation set**
  - Select threshold that **maximizes F1-score**
  - Save optimal threshold as part of model artifacts (e.g., in `config.py` or MLflow params)

---
---
## **Evaluation Metrics**
The following **metrics** must be logged and reported for model evaluation:

| **Metric**               | **Description**                                  | **Implementation**                          |
|--------------------------|--------------------------------------------------|---------------------------------------------|
| **Accuracy**             | Overall correctness                              | `(TP + TN) / (TP + TN + FP + FN)`            |
| **Precision**            | % of predicted defectives that are actual defectives | `TP / (TP + FP)`                     |
| **Recall**               | % of actual defectives correctly identified      | `TP / (TP + FN)`                     |
| **F1-Score**             | Harmonic mean of precision and recall            | `2 * (Precision * Recall) / (Precision + Recall)` |
| **Confusion Matrix**    | Per-class true/false positives/negatives         | `sklearn.metrics.confusion_matrix`          |
| **ROC-AUC**              | Area under the ROC curve                         | `sklearn.metrics.roc_auc_score`              |
| **Precision-Recall Curve** | Trade-off between precision and recall        | `sklearn.metrics.PrecisionRecallDisplay`    |

- **Primary Metric**: **F1-score** (critical for imbalanced data, even after resampling)
- **Secondary Metrics**: Precision, Recall, ROC-AUC
- **Business Context for Metrics**:
  - High **recall** (minimize false negatives) is **more important** than high precision (false positives are less costly than false negatives in this context).

---
---
## **File-by-File Changes Required**

---

### **1. `src/config.py`**
**Purpose**: Central configuration constants

**Changes Required**:
- **Remove**:
  - Tabular-specific constants: `SCALER_PATH`, `COLUMNS_DRIFT_PATH`, `COLUMNS_LIST_PATH`, `CAT_MISSING_IMPUTE_PATH`, `OUTLIER_SUMMARY_PATH`
  - Tabular model paths: `XGBOOST_MODEL_PATH`, `LR_MODEL_PATH`
  - Tabular parameters: `OUTLIER_Z_THRESHOLD`, `IMPUTATION_METHOD`
- **Add**:
  - `IMAGE_DATA_DIR`: Path to directory containing glass vial images (e.g., `data/images/raw/`)
  - `METADATA_PATH`: Path to `metadata.csv` (e.g., `data/images/metadata.csv`)
  - `IMAGE_SIZE`: `(64, 64)` (fixed)
  - `NUM_CLASSES`: `2`
  - `BATCH_SIZE`: `32` (default)
  - `NUM_EPOCHS`: `50` (default)
  - `LEARNING_RATE`: `0.001` (default)
  - `DEVICE`: `'cuda'` or `'cpu'` (auto-detected)
  - `PYTORCH_MODEL_PATH`: Path to save PyTorch model (e.g., `models/vial_cnn_model.pth`)
  - `TRANSFORM_MEAN`: `[0.485, 0.456, 0.406]` (ImageNet stats)
  - `TRANSFORM_STD`: `[0.229, 0.224, 0.225]` (ImageNet stats)
  - `THRESHOLD`: `0.5` (default, but tunable)
- **Update**:
  - `EXPERIMENT_NAME`: `"vial_image_classification"`
  - `MODEL_NAME`: `"cnn_model"`

---

### **2. `src/data/fetch.py`**
**Purpose**: Data loading and preparation

**Changes Required**:
- **Remove**:
  - `load_raw_data()` (CSV loading)
  - `filter_data_by_date()` (if not applicable to images)
  - DVC-specific code (if not used for images)
- **Add**:
  - `load_image_dataset()`:
    - Reads images from `IMAGE_DATA_DIR` and labels from `METADATA_PATH`
    - Returns a list of `(image_path, label)` tuples
    - Supports `.jpg` and `.png` formats
  - `create_image_metadata()`:
    - Generates `metadata.csv` if not provided (for testing)
    - Columns: `filename`, `class`, `batch_id`, `timestamp`
  - `validate_image_dataset()`:
    - Checks for corrupt images, invalid sizes, or missing files
    - Logs warnings for any issues found
- **Update**:
  - `fetch_and_prepare_data()`:
    - Calls `load_image_dataset()` and `validate_image_dataset()`
    - Returns validated dataset

---

### **3. `src/data/preprocess.py`**
**Purpose**: Data preprocessing

**Changes Required**:
- **Remove**:
  - All tabular-specific functions:
    - `select_features()`, `clean_data()`, `create_categorical_columns()`, `separate_columns()`
    - `handle_outliers()`, `impute_missing_values_data()`, `standardize_data()`, `combine_data()`
  - `sklearn` imports (e.g., `MinMaxScaler`)
- **Add**:
  - `define_transforms()`:
    - Returns `train_transforms` and `val_transforms` (as specified above)
  - `create_dataloaders()`:
    - Creates `DataLoader` objects for training, validation, and testing
    - Uses `torchvision.datasets.DatasetFolder` or a custom dataset class
    - Applies `train_transforms`/`val_transforms` as appropriate
    - Handles batching (`BATCH_SIZE`) and shuffling (only for training)
  - `split_dataset()`:
    - Splits dataset into train/val/test sets (70%/15%/15%)
    - Uses **stratified sampling** to maintain 50/50 class balance
- **Update**:
  - `preprocess_data()`:
    - Takes raw image dataset as input
    - Calls `split_dataset()` and `create_dataloaders()`
    - Returns `train_loader`, `val_loader`, `test_loader`
    - Saves preprocessing metadata (e.g., class distribution, image stats)

---

### **4. `src/data/features.py`**
**Purpose**: Feature engineering (minimal changes)

**Changes Required**:
- **Remove**:
  - `bin_categorical_columns()`, `save_artifacts()` (tabular-specific)
- **Update**:
  - `create_features()`:
    - Repurpose to save **image-specific artifacts**:
      - `class_names.json` (e.g., `{"0": "good", "1": "defective"}`)
      - `image_statistics.json` (mean/std per channel, computed from training set)
- **Alternative**: Merge this functionality into `preprocess.py` and **delete this file**

---

### **5. `src/models/train.py`**
**Purpose**: Model training

**Changes Required**:
- **Remove**:
  - All XGBoost/Logistic Regression code:
    - `train_xgboost()`, `evaluate_and_save_xgboost()`, `train_logistic_regression()`
- **Add**:
  - `SimpleCNN` class (as defined above)
  - `train_cnn_model()`:
    - Initializes `SimpleCNN`, `BCELoss`, `Adam` optimizer
    - Implements training loop:
      - Iterates over `train_loader` in batches
      - Computes loss, backpropagates, updates weights
      - Logs **loss, accuracy, F1-score** to MLflow every epoch
      - Validates on `val_loader` every epoch
      - Saves **best model** (based on validation F1-score) to `PYTORCH_MODEL_PATH`
    - Returns trained model and training history
  - `evaluate_cnn_model()`:
    - Evaluates model on `test_loader`
    - Computes **all metrics** (accuracy, F1, precision, recall, ROC-AUC)
    - Generates **confusion matrix** and **classification report**
    - Logs metrics to MLflow
  - `tune_threshold()`:
    - Evaluates model on `val_loader` with thresholds `[0.3, 0.4, 0.5, 0.6, 0.7]`
    - Selects threshold that **maximizes F1-score**
    - Saves optimal threshold to `config.py` or MLflow
- **Update**:
  - `train_models()`:
    - Calls `train_cnn_model()` (replaces XGBoost/LR training)
    - Passes `train_loader`, `val_loader`, and hyperparameters
  - `save_model_artifacts()`:
    - Saves model state dict, architecture, and threshold
    - Logs to MLflow as a PyTorch model
- **Imports**:
  - **Remove**: `xgboost`, `sklearn`, `joblib`
  - **Add**: `torch`, `torch.nn`, `torch.optim`, `torchvision.transforms`

---

### **6. `src/models/evaluate.py`**
**Purpose**: Model evaluation

**Changes Required**:
- **Keep**:
  - Generic functions (`calculate_accuracy`, `get_confusion_matrix`, `get_classification_report`, etc.)
- **Update**:
  - Add helper functions for **tensor handling**:
    - `tensor_to_numpy()`: Converts PyTorch tensors to NumPy arrays
    - `get_predictions()`: Runs model on `DataLoader` and returns predictions/labels
  - Update `evaluate_model()` and `evaluate_both_datasets()` to:
    - Handle PyTorch tensor outputs
    - Use `tensor_to_numpy()` where needed
- **Add**:
  - `plot_confusion_matrix()`: Visualizes confusion matrix (optional for students)
  - `plot_roc_curve()`: Plots ROC curve (optional for students)

---

### **7. `src/models/registry.py`**
**Purpose**: Model registration with MLflow

**Changes Required**:
- **Update**:
  - Replace `mlflow.sklearn` with `mlflow.pytorch` for model logging
  - Update `load_model_results()` to handle PyTorch model outputs
  - No other changes needed (MLflow is model-agnostic for tracking)

---
---
### **8. `src/deployment/deploy.py`**
**Purpose**: Model deployment and stage management

**Changes Required**:
- **Minimal changes**:
  - Update comments/docstrings to reflect image classification
  - Ensure deployment logic can handle **image input** (not tabular data)

---

### **9. `src/pipeline.py`**
**Purpose**: Main pipeline orchestration

**Changes Required**:
- **Update**:
  - `run_data_pipeline()`:
    - Returns `train_loader`, `val_loader`, `test_loader` (instead of DataFrame)
  - `run_model_pipeline()`:
    - Calls `train_cnn_model()` (instead of XGBoost/LR)
    - Passes `DataLoader` objects and hyperparameters
  - `run_full_pipeline()`:
    - Updates data flow from CSV → images
    - Updates model flow from XGBoost/LR → CNN

---
---
### **10. `src/utils.py`**
**Purpose**: Utility functions

**Changes Required**:
- **Remove**:
  - Tabular-specific utilities:
    - `create_dummy_cols()`, `describe_numeric_col()`, `impute_missing_values()`
- **Add**:
  - `calculate_image_statistics()`:
    - Computes mean/std per channel for a dataset
    - Used for normalization
  - `verify_image_dataset()`:
    - Checks all images in a directory are valid (not corrupt, correct size)
  - `count_classes()`:
    - Counts samples per class in a dataset
  - `tensor_to_numpy()`:
    - Converts PyTorch tensors to NumPy arrays
- **Update**:
  - `check_dataframe_not_empty()` → Keep for metadata validation
  - `check_columns_exist()` → Replace with `check_metadata_columns()` (for `metadata.csv`)

---
---
### **11. `__init__.py` Files**
**Files**: `src/__init__.py`, `src/models/__init__.py`, `src/data/__init__.py`, `src/deployment/__init__.py`

**Changes Required**:
- Update **docstrings** to reflect:
  - Image classification (not tabular)
  - PyTorch CNN (not XGBoost/LR)

---
---
## **Files Summary Table**

| **File**               | **Current Purpose**          | **New Purpose**               | **Major Changes**                                  |
|------------------------|------------------------------|------------------------------|----------------------------------------------------|
| `config.py`            | Tabular config               | Image config                 | Replace tabular paths/params with image equivalents|
| `data/fetch.py`        | CSV loading                  | Image loading                | Replace CSV reading with image/metadata loading   |
| `data/preprocess.py`   | Tabular preprocessing        | Image preprocessing          | Replace preprocessing with transforms/DataLoaders   |
| `data/features.py`     | Tabular feature engineering  | Image feature engineering    | Minimal; repurpose for image artifacts              |
| `models/train.py`      | XGBoost/LR training          | CNN training                 | Complete rewrite for PyTorch                        |
| `models/evaluate.py`   | Generic evaluation           | Generic evaluation          | Minimal; add tensor handling                        |
| `models/registry.py`   | MLflow registration          | MLflow registration          | Update for PyTorch model type                       |
| `deployment/deploy.py` | Deployment                   | Deployment                  | Minimal; ensure image input support                 |
| `pipeline.py`          | Orchestration                | Orchestration               | Update data/model flow                              |
| `utils.py`             | Utilities                    | Utilities                   | Remove tabular utils, add image utils                |
| `__init__.py` files    | Documentation                | Documentation               | Update docstrings                                   |

---
---
## **Implementation Priority**
| **Priority** | **Files**                          | **Notes**                                  |
|--------------|------------------------------------|--------------------------------------------|
| **High**     | `config.py`, `data/fetch.py`, `data/preprocess.py`, `models/train.py` | Core data and model changes |
| **Medium**   | `pipeline.py`, `models/evaluate.py`, `utils.py` | Orchestration and utilities |
| **Low**      | `models/registry.py`, `deployment/deploy.py`, `__init__.py`, `data/features.py` | Minimal changes |

---
---
## **Dependencies**
### **Add**
```text
torch == 2.2.0
torchvision == 0.17.0
pillow == 10.1.0
```

### **Remove (Optional)**
```text
xgboost
scikit-learn  # Keep if needed for metrics (e.g., classification_report)
```

---
---
## **Notes**
1. **Determinism**: All random operations (e.g., data splits, augmentations) must use **fixed seeds** (`42`) for reproducibility.
2. **Validation**: The CNN architecture and preprocessing must be **exactly as specified** to ensure trained models can be validated.
3. **Flexibility**: Students may **adjust hyperparameters** (e.g., `BATCH_SIZE`, `LEARNING_RATE`) but must document changes.
4. **Testing**: Students are responsible for adding **unit tests** (e.g., for `load_image_dataset()`, `SimpleCNN` forward pass).