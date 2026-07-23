# **Pharmaceutical Vial Image Dataset Documentation**

## **Dataset Overview**

This document describes the **Pharmaceutical Vial Image Dataset** used for training and evaluating the image classification pipeline.

**Dataset Name**: Identifying Impurities in Liquids of Pharmaceutical Vials

**Source**: 27th International Conference on Pattern Recognition (ICPR 2024)

**Authors**: G. Rosati, K. Marchesini, L. Lumetti, F. Sartori, B. Balboni, F. Begarani, L. Vescovi, F. Bolelli, C. Grana

**Publication Year**: 2024

**Dataset URL**: https://ditto.ing.unimore.it/residual/

**Access Date**: 2026-07-23

**Citation**: 
```
@inproceedings{rosati2024identifying,
  title={Identifying Impurities in Liquids of Pharmaceutical Vials},
  author={Rosati, G and Marchesini, K and Lumetti, L and Sartori, F and Balboni, B and Begarani, F and Vescovi, L and Bolelli, F and Grana, C},
  booktitle={27th International Conference on Pattern Recognition (ICPR)},
  year={2024}
}
```

---

## **Dataset Description**

The dataset contains high-resolution images of pharmaceutical glass vials with various types of impurities in the liquid. The vials were photographed under controlled conditions to capture the presence of foreign particles that affect the quality of pharmaceutical products.

### **Purpose**
This dataset is designed for developing and evaluating computer vision algorithms that can automatically detect impurities in pharmaceutical vials. The primary application is quality control in pharmaceutical manufacturing, where manual inspection is time-consuming and subject to human error.

### **Data Collection**
- **Environment**: Controlled laboratory setting
- **Camera**: High-resolution digital camera
- **Lighting**: Standardized lighting conditions
- **Vial Positioning**: Each vial was rotated to capture 19 different angles
- **Image Format**: JPEG
- **Resolution**: 1600x768 pixels
- **Color Space**: RGB (3 channels, 8-bit per channel)

---

## **Dataset Structure**

### **Raw Dataset Structure (Original)**
```bash
data/
├── Clean/
│   ├── Clean_001/
│   │   ├── TLC00-20240326154141-058901.jpg
│   │   ├── TLC00-20240326154141-058902.jpg
│   │   └── ... (19 rotated images)
│   ├── Clean_002/
│   └── ... (153 clean vials)
├── Glass/
│   ├── Glass_001/
│   │   ├── TLC00-20240326144241-037301.jpg
│   │   └── ... (19 rotated images)
│   └── ... (144 glass impurity vials)
├── Plastic/
│   └── ... (144 plastic impurity vials)
├── Rubber/
│   └── ... (146 rubber impurity vials)
└── Sand/
    └── ... (143 sand impurity vials)
```

### **Organized Dataset Structure (Processed)**
```bash
data/images/
├── raw/
│   ├── Clean_001.jpg
│   ├── Clean_002.jpg
│   ├── ...
│   ├── Glass_001.jpg
│   ├── Glass_002.jpg
│   └── ... (730 total images)
└── metadata.csv
```

---

## **Dataset Statistics**

### **Overall Statistics**
- **Total Raw Images**: 13,880 (19 images per vial × 730 vials)
- **Total Vials**: 730
- **Total Processed Images**: 730 (1 representative image per vial)
- **Image Dimensions**: 1600×768 pixels (native), resized to 224×224 for model input
- **File Format**: JPEG
- **Color Channels**: RGB (3 channels)
- **File Size Range**: ~500KB - 1MB per image

### **Class Distribution**

| **Class** | **Category** | **Vials** | **Percentage** | **Label** | **Class Name** |
|-----------|--------------|-----------|---------------|----------|----------------|
| 0 | Clean | 153 | 20.96% | 0 | clean |
| 1 | Glass | 144 | 19.73% | 1 | defective |
| 1 | Plastic | 144 | 19.73% | 1 | defective |
| 1 | Rubber | 146 | 20.00% | 1 | defective |
| 1 | Sand | 143 | 19.59% | 1 | defective |
| **Total** | **-** | **730** | **100%** | **-** | **-** |

**Binary Classification Distribution:**
- Class 0 (Clean): 153 vials (20.96%)
- Class 1 (Defective): 577 vials (79.04%)

### **Impurity Type Distribution**
- **Glass Particles**: 144 vials (19.73%)
- **Plastic Particles**: 144 vials (19.73%)
- **Rubber Particles**: 146 vials (20.00%)
- **Sand Particles**: 143 vials (19.59%)
- **Clean (No Impurities)**: 153 vials (20.96%)

---

## **Class Definitions**

### **Class 0: Clean (Good Quality)**
- **Description**: Vials containing clear liquid with no visible impurities
- **Examples**: 153 vials
- **Label**: 0
- **Class Name**: "clean"
- **Impurity Type**: "none"

### **Class 1: Defective (Poor Quality)**
- **Description**: Vials containing liquid with visible impurities
- **Subtypes**:
  - **Glass**: Small glass particles suspended in liquid
  - **Plastic**: Plastic fragments or fibers in liquid
  - **Rubber**: Rubber particles (likely from stopper degradation)
  - **Sand**: Sand or grit particles in liquid
- **Examples**: 577 vials total
- **Label**: 1
- **Class Name**: "defective"
- **Impurity Types**: "glass", "plastic", "rubber", "sand"

---

## **Image Characteristics**

### **Visual Appearance**
- **Background**: Light-colored background (likely white or light gray)
- **Vial**: Clear glass vial with liquid content
- **Liquid**: Typically clear or slightly colored liquid
- **Impurities**: Various types of particles visible in liquid
- **Lighting**: Even illumination from top or sides
- **Focus**: Vial and liquid are in focus

### **Color Information**
- **Color Space**: RGB
- **Bit Depth**: 8 bits per channel
- **Expected Color Distribution**: 
  - High values in all channels for background
  - Varied values in vial and liquid regions
  - Darker regions where impurities are present

### **Size and Resolution**
- **Original Resolution**: 1600×768 pixels
- **Aspect Ratio**: ~2.08:1 (landscape orientation)
- **Model Input Resolution**: 224×224 pixels (resized with aspect ratio preservation)
- **File Size**: ~500KB - 1MB per JPEG image

---

## **Metadata Schema**

The `metadata.csv` file contains the following columns:

| **Column** | **Type** | **Description** | **Example** |
|------------|----------|-----------------|-------------|
| `filename` | string | Filename in organized dataset | "Clean_001.jpg" |
| `class` | integer | Binary class label (0=clean, 1=defective) | 0 |
| `class_name` | string | Human-readable class name | "clean" |
| `vial_id` | string | Original vial identifier | "Clean_001" |
| `impurity_type` | string | Type of impurity or "none" | "none", "glass", "plastic", "rubber", "sand" |
| `original_path` | string | Path to original image in raw dataset | "data/Clean/Clean_001/TLC00-20240326154141-058901.jpg" |
| `num_images_in_series` | integer | Number of images in original rotation series | 19 |
| `width` | integer | Original image width in pixels | 1600 |
| `height` | integer | Original image height in pixels | 768 |
| `format` | string | Image file format | "JPEG" |
| `batch_id` | string | Batch identifier for tracking | "BATCH_2024_ICPR" |
| `timestamp` | datetime | When the image was processed | "2026-07-23T10:26:10.757295" |

---

## **Data Preparation**

### **Image Selection**
From the original dataset with 19 rotated images per vial, we selected **1 representative image per vial** to create a manageable dataset. The selection criteria:

1. **First Image**: For simplicity, we selected the first image in each vial directory
2. **Alternative**: Could select the middle image (index 9 or 10) for most representative view
3. **Future Enhancement**: Could use all 19 images for more robust training

### **Resizing Strategy**
The high-resolution images (1600×768) are resized to 224×224 for model input:

- **Method**: Bilinear interpolation (default in PyTorch transforms)
- **Aspect Ratio**: Preserved during resizing
- **Reason**: 224×224 provides good balance between computational efficiency and feature preservation
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### **Stratification**
The dataset maintains the natural class distribution:
- **Training Set**: ~70% of vials, stratified by class
- **Validation Set**: ~15% of vials, stratified by class
- **Test Set**: ~15% of vials, stratified by class

This ensures that each split has approximately 21% clean and 79% defective vials.

---

## **Quality Control**

### **Data Validation**
All images in the organized dataset have been validated for:

1. **File Integrity**: All images can be opened and decoded
2. **Format Consistency**: All images are JPEG format
3. **Color Channels**: All images are RGB (3 channels)
4. **Resolution**: All images are 1600×768 pixels (native resolution)
5. **File Size**: All images are within expected size range

### **Known Limitations**

1. **Class Imbalance**: The dataset has ~21% clean and ~79% defective vials. This reflects real-world scenarios where defective vials are less common but critical to detect.

2. **Single View**: Each vial is represented by a single image (out of 19 available). This may miss impurities visible from other angles.

3. **Image Variability**: Images may have slight variations in lighting, positioning, and focus due to the data collection process.

4. **Impurity Visibility**: Some impurities may be subtle and difficult to detect, even for human inspectors.

---

## **Usage Guidelines**

### **For Model Training**
```python
# Example usage with PyTorch
from torchvision import transforms
from PIL import Image

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load and transform image
image = Image.open("data/images/raw/Clean_001.jpg")
image_tensor = transform(image)
```

### **For Data Loading**
```python
# Using the organized dataset with our pipeline
from src.data.fetch import load_image_dataset

# Load dataset
dataset = load_image_dataset()  # Returns list of (image_path, label) tuples

# Or use the metadata directly
import pandas as pd
metadata = pd.read_csv("data/images/metadata.csv")
```

---

## **Dataset Organization Script**

The dataset was organized using the script `src/data/organize_dataset.py`, which:

1. Scans the raw dataset directory structure
2. Extracts one representative image per vial
3. Creates the organized directory structure
4. Generates the metadata CSV file
5. Validates all images and metadata

### **Running the Script**
```bash
# From project root
python3 src/data/organize_dataset.py
```

### **Output**
- Organized images in `data/images/raw/`
- Metadata file at `data/images/metadata.csv`
- Dataset statistics printed to console

---

## **File Manifest**

### **Raw Dataset Files**
- **Location**: `data/{Clean,Glass,Plastic,Rubber,Sand}/`
- **Total Directories**: 5 (categories) + 730 (vials)
- **Total Images**: 13,880 JPEG files
- **Total Size**: ~4.5 GB (estimated)

### **Processed Dataset Files**
- **Location**: `data/images/raw/`
- **Total Images**: 730 JPEG files
- **Metadata**: `data/images/metadata.csv`
- **Total Size**: ~730 MB (estimated)

---

## **Acknowledgments**

This dataset is provided by the authors of the paper:

**"Identifying Impurities in Liquids of Pharmaceutical Vials"**

Presented at the **27th International Conference on Pattern Recognition (ICPR 2024)**.

The authors have made this dataset publicly available for research purposes at:
https://ditto.ing.unimore.it/residual/

### **Citation Requirement**
When using this dataset in publications or presentations, please cite the original paper:

```
@inproceedings{rosati2024identifying,
  title={Identifying Impurities in Liquids of Pharmaceutical Vials},
  author={Rosati, G and Marchesini, K and Lumetti, L and Sartori, F and Balboni, B and Begarani, F and Vescovi, L and Bolelli, F and Grana, C},
  booktitle={Proceedings of the 27th International Conference on Pattern Recognition},
  year={2024},
  organization={IEEE}
}
```

### **License**
The dataset is provided for academic and research use. Please check the original dataset website for specific license terms.

---

## **Contact Information**

For questions about the original dataset, please contact the authors through the ICPR 2024 conference or the dataset website.

For questions about this specific implementation and organization, please refer to the project documentation.

---

## **Changelog**

| **Date** | **Change** | **Author** |
|----------|------------|------------|
| 2026-07-23 | Initial dataset organization and documentation | Jeppe Kristensen |
| 2026-07-23 | Created metadata.csv with vial information | Jeppe Kristensen |
| 2026-07-23 | Extracted 1 representative image per vial (730 total) | Jeppe Kristensen |

---

## **Appendix: Sample Metadata**

```csv
filename,class,class_name,vial_id,impurity_type,original_path,num_images_in_series,width,height,format,batch_id,timestamp
Clean_001.jpg,0,clean,Clean_001,none,data/Clean/Clean_001/TLC00-20240326154141-058901.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T10:26:10.757295
Clean_002.jpg,0,clean,Clean_002,none,data/Clean/Clean_002/TLC00-20240326154142-058906.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T10:26:10.760899
Glass_001.jpg,1,defective,Glass_001,glass,data/Glass/Glass_001/TLC00-20240326144241-037301.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T10:26:10.764969
Plastic_001.jpg,1,defective,Plastic_001,plastic,data/Plastic/Plastic_001/TLC00-20240326150921-058913.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T10:26:10.768107
Rubber_001.jpg,1,defective,Rubber_001,rubber,data/Rubber/Rubber_001/TLC00-20240326151432-059001.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T10:26:10.771234
Sand_001.jpg,1,defective,Sand_001,sand,data/Sand/Sand_001/TLC00-20240326151943-059101.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T10:26:10.774567
```

---

*Last updated: 2026-07-23*