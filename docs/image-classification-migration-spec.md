# **Image Classification Migration Specification Document - UPDATED**

---

## **Overview**
This document specifies the **technical changes** required to migrate the existing MLOps pipeline from **tabular data classification (XGBoost)** to **image classification (PyTorch CNN)**. The goal is to enable **automated visual inspection of pharmaceutical glass vials** for quality control.

**Dataset Source**: The implementation uses the ICPR 2024 dataset:
> G. Rosati, K. Marchesini, L. Lumetti, F. Sartori, B. Balboni, F. Begarani, L. Vescovi, F. Bolelli, C. Grana (2024). "Identifying Impurities in Liquids of Pharmaceutical Vials." 27th International Conference on Pattern Recognition (ICPR).

**Dataset URL**: https://ditto.ing.unimore.it/residual/

---

## **Current State**
The current pipeline processes **tabular data (CSV files)** with:
- XGBoost and Logistic Regression models
- Feature engineering for categorical/continuous columns
- Standardization, outlier handling, missing value imputation
- MLflow tracking and model registry

---

## **Target State**
The new pipeline will process **pharmaceutical vial images** with:
- PyTorch CNN model for binary classification (clean vs. defective)
- Image loading, resizing, and preprocessing for variable input sizes
- Automatic handling of 1600x768 RGB images (resized to model input)
- Same MLflow tracking and model registry infrastructure

---

## **Problem Context**
- **Domain**: Pharmaceutical manufacturing quality control
- **Use Case**: Binary classification of glass vial images
  - **Input**: Images of produced glass vials (1600x768 RGB, variable size handled)
  - **Classes**:
    - `0`: Good quality (clean, acceptable)
    - `1`: Not good quality (defective - contains impurities: glass, plastic, rubber, or sand)
  - **Goal**: Automated visual inspection to classify vial quality
  - **NUM_CLASSES**: 2 (binary classification)

---

## **Data Specifications**

### **Image Properties**
| **Property**       | **Value**               | **Notes**                                  |
|--------------------|-------------------------|--------------------------------------------|
| **Format**         | JPEG                    | Standard JPEG format from dataset          |
| **Resolution**     | 1600x768 (native)       | Images will be resized to model input size |
| **Color Channels** | RGB                     | 3-channel color images                    |
| **Bit Depth**      | 8-bit                   | Standard for JPEG images                  |

### **Dataset Statistics**
- **Total Vials**: 730 (1 sample per vial extracted from 13,880 total images)
- **Class Distribution**:
  - Clean: 153 vials (20.96%) - Class 0 (good)
  - Glass: 144 vials (19.73%) - Class 1 (defective)
  - Plastic: 144 vials (19.73%) - Class 1 (defective)
  - Rubber: 146 vials (20.00%) - Class 1 (defective)
  - Sand: 143 vials (19.59%) - Class 1 (defective)
  - **Effective Distribution**: ~21% clean, ~79% defective
- **Original Images per Vial**: 19 rotated images (we use 1 representative image per vial)
- **Train/Val/Test Split**: Standard ratios (70%/15%/15%), **stratified**

### **Directory Structure**
**Organized Structure**:
```bash
/data/images/
├── raw/
│   ├── Clean_001.jpg
│   ├── Clean_002.jpg
│   ├── Glass_001.jpg
│   ├── Plastic_001.jpg
│   ├── Rubber_001.jpg
│   ├── Sand_001.jpg
│   └── ... (730 total images)
└── metadata.csv  # Columns: filename, class, class_name, vial_id, impurity_type, etc.
```

### **Metadata**
| **Field**      | **Type**   | **Example**          | **Purpose**                     |
|----------------|------------|----------------------|---------------------------------|
| `filename`     | str        | `Clean_001.jpg`      | Image filename in raw directory |
| `class`        | int        | `0` or `1`           | Label (0=clean, 1=defective)    |
| `class_name`   | str        | `"clean"` or `"defective"` | Human-readable class name |
| `vial_id`     | str        | `"Clean_001"`       | Original vial identifier        |
| `impurity_type`| str       | `"none"`, `"glass"`, `"plastic"`, `"rubber"`, `"sand"` | Type of impurity (none for clean) |
| `original_path`| str       | `"data/Clean/Clean_001/TLC00-20240326154141-058901.jpg"` | Path to original image |
| `num_images_in_series` | int | `19` | Number of images in original series |
| `width`        | int        | `1600`              | Original image width            |
| `height`       | int        | `768`               | Original image height           |
| `format`       | str        | `"JPEG"`            | Image format                    |
| `batch_id`     | str        | `"BATCH_2024_ICPR"` | Batch identifier               |
| `timestamp`    | datetime   | `2026-07-23T10:26:10.757295` | Processing timestamp           |

### **Data Quality Checks**
Implement **automated checks** in code to validate:
- **Corrupt image detection** (PIL/OpenCV validation)
- **File size constraints** (e.g., reject images < 10KB or > 5MB for our larger images)
- **Resolution validation** (ensure all images can be loaded and resized)
- **Color channel validation** (ensure RGB, not grayscale or RGBA)

### **Data Augmentation**
Use the following **PyTorch transforms** for training (applied after resizing to model input size):
```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to model input (updated from 64x64)
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
The **PyTorch CNN** uses a **deterministic architecture** optimized for the 1600x768 vial images:

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(SimpleCNN, self).__init__()
        # Convolutional layers for larger images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate input size after convolutions and pooling
        # For 224x224 input: 224 -> 112 -> 56 -> 28 (after 3 pool layers)
        # Feature map: 128 channels * 28 * 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Updated for 224x224 input
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Updated for 224x224 input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x
```

### **Training Configuration**
- **Input Size**: 224x224 RGB (resized from native 1600x768)
- **Loss**: `BCELoss` (Binary Cross Entropy Loss)
- **Optimizer**: `Adam` (default `lr=0.001`)
- **Metrics**: Accuracy, F1-score, Precision, Recall
- **Device**: Automatic detection (`cuda` if available, else `cpu`)
- **Determinism**: Set `torch.manual_seed(42)` and `numpy.random.seed(42)` for reproducibility

### **Threshold Selection**
- **Default Threshold**: `0.5` (output >= 0.5 -> class `0` (clean), else class `1` (defective))
- **Threshold Tuning**:
  - Evaluate thresholds `[0.3, 0.4, 0.5, 0.6, 0.7]` on the **validation set**
  - Select threshold that **maximizes F1-score**
  - Save optimal threshold as part of model artifacts

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

- **Primary Metric**: **F1-score** (critical for imbalanced data)
- **Secondary Metrics**: Precision, Recall, ROC-AUC
- **Business Context for Metrics**:
  - High **recall** (minimize false negatives) is **more important** than high precision (false positives are less costly than missing defective vials).

---

## **Files Summary Table**

| **File**               | **Current Purpose**          | **New Purpose**               | **Major Changes**                                  |
|------------------------|------------------------------|------------------------------|----------------------------------------------------|
| `config.py`            | Tabular config               | Image config                 | Replace tabular paths/params with image equivalents|
| `data/fetch.py`        | CSV loading                  | Image loading                | Replace CSV reading with image/metadata loading   |
| `data/preprocess.py`   | Tabular preprocessing        | Image preprocessing          | Replace preprocessing with transforms/DataLoaders   |
| `data/organize_dataset.py` | N/A | Dataset organization | New file for organizing raw ICPR dataset |
| `models/train.py`      | XGBoost/LR training          | CNN training                 | Complete rewrite for PyTorch with updated sizes    |
| `models/evaluate.py`   | Generic evaluation           | Generic evaluation          | Add tensor handling for PyTorch                    |
| `models/registry.py`   | MLflow registration          | MLflow registration          | Update for PyTorch model type                       |
| `pipeline.py`          | Orchestration                | Orchestration               | Update data/model flow for images                   |
| `utils.py`             | Utilities                    | Utilities                   | Add image utilities, remove tabular-specific       |

---

## **Implementation Notes**

1. **Image Size Handling**: The model now accepts 224x224 input (resized from native 1600x768) instead of the originally specified 64x64. This provides better feature extraction for the pharmaceutical vial classification task.

2. **Dataset Attribution**: All documentation and code references the ICPR 2024 dataset with proper attribution to the authors.

3. **Class Balance**: The dataset has ~21% clean and ~79% defective vials. Training uses stratified sampling to maintain this distribution across train/val/test splits.

4. **Impurity Types**: The dataset includes 4 types of impurities (glass, plastic, rubber, sand) plus clean vials. All impurities are mapped to class 1 (defective).

5. **Image Selection**: From the original 19 rotated images per vial, we extract 1 representative image to create a manageable dataset of 730 images.

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

## **References**
1. **Dataset**: G. Rosati, K. Marchesini, L. Lumetti, F. Sartori, B. Balboni, F. Begarani, L. Vescovi, F. Bolelli, C. Grana (2024). "Identifying Impurities in Liquids of Pharmaceutical Vials." 27th International Conference on Pattern Recognition (ICPR).

2. **Dataset URL**: https://ditto.ing.unimore.it/residual/

3. **Access Date**: 2026-07-23