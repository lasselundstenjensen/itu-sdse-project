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
│   └── ... (145 rubber impurity vials)
└── Sand/
    └── ... (144 sand impurity vials)
```

### **Organized Dataset Structure (Processed)**
```bash
data/
├── images/
│   ├── raw/
│   │   ├── Clean_001.jpg
│   │   ├── Clean_002.jpg
│   │   ├── ...
│   │   ├── Glass_001.jpg
│   │   ├── Glass_002.jpg
│   │   └── ... (one image per vial)
│   └── metadata.csv
└── test/                     <- held-out test set (see below)
    ├── Clean_151_TLC00-20240326150919-041901.jpg
    ├── ... (2 per class)
    └── metadata.csv
```

### **Held-Out Test Set**
A small held-out test set lives at `data/test/`: **10 images, 2 per class** (clean, glass, plastic, rubber, sand), with `batch_id=TEST_SET` in its metadata. It is separate from the training images and is used to check predictions against the deployed model without touching the training data. Its metadata uses the same schema as the training metadata, but filenames include the full camera file name (e.g. `Clean_151_TLC00-20240326150919-041901.jpg`).

---

## **Dataset Statistics**

### **Overall Statistics**
- **Total Raw Images**: 13,870 (19 images per vial)
- **Total Vials**: 730
- **Total Processed Images**: 730 (1 image per vial)
- **Held-Out Test Images**: 10 (2 per class, kept separate from the training set)
- **Image Dimensions**: 1600×768 pixels (native; stored images are not resized)
- **File Format**: JPEG
- **Color Channels**: RGB (3 channels)
- **File Size Range**: ~500KB - 1MB per image

### **Class Distribution**

| **Class** | **Category** | **Vials** | **Percentage** | **Label** | **Class Name** |
|-----------|--------------|-----------|---------------|----------|----------------|
| 0 | Clean | 153 | 20.96% | 0 | clean |
| 1 | Glass | 144 | 19.73% | 1 | defective |
| 1 | Plastic | 144 | 19.73% | 1 | defective |
| 1 | Rubber | 145 | 19.86% | 1 | defective |
| 1 | Sand | 144 | 19.73% | 1 | defective |
| **Total** | **-** | **730** | **100%** | **-** | **-** |

**Binary Classification Distribution:**
- Class 0 (Clean): 153 vials (20.96%)
- Class 1 (Defective): 577 vials (79.04%)

### **Impurity Type Distribution**
- **Glass Particles**: 144 vials (19.73%)
- **Plastic Particles**: 144 vials (19.73%)
- **Rubber Particles**: 145 vials (19.86%)
- **Sand Particles**: 144 vials (19.73%)
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
| `original_path` | string | Path to original image in raw dataset | "data/Clean/Clean_001/TLC00-20240326154142-058909.jpg" |
| `num_images_in_series` | integer | Number of images in original rotation series | 19 |
| `width` | integer | Original image width in pixels | 1600 |
| `height` | integer | Original image height in pixels | 768 |
| `format` | string | Image file format | "JPEG" |
| `batch_id` | string | Batch identifier for tracking | "BATCH_2024_ICPR" (training), "TEST_SET" (held-out test) |
| `timestamp` | datetime | When the image was processed | "2026-07-23T13:03:11.963263" |

---

## **Data Preparation**

### **Image Selection**
From the original dataset with 19 rotated images per vial, we keep **1 image per vial** to create a manageable dataset. The chosen image varies per vial - it is not always the first of the rotation series; any of the 19 views serves as a representative.

### **Preprocessing**
The fetched images are stored exactly as they come from the raw dataset - native 1600×768 resolution, no resizing or other processing is applied to the files. Anything done to the images at training time happens in the notebooks, not in the dataset.

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

## **Dataset Organization**

The dataset was organized in a one-off step, which:

1. Scans the raw dataset directory structure
2. Extracts one representative image per vial
3. Creates the organized directory structure
4. Generates the metadata CSV file
5. Validates all images and metadata

### **Output**
- Organized images in `data/images/raw/`
- Metadata file at `data/images/metadata.csv`
- Dataset statistics printed to console

### **Where the data lives now**
Both the organized set and the held-out test set are hosted in the course object store (`data` bucket at `itudsip.hel1.your-objectstorage.com`, keys `images/raw/`, `images/metadata.csv`, `test/`, `test/metadata.csv`). The training notebook downloads them from there; the raw 19-image tree is not distributed with the repository. The two metadata CSVs are versioned with DVC (`data/images/metadata.csv.dvc`, `data/test/metadata.csv.dvc`), so `dvc pull` fetches them from the object store, while the JPEG images themselves are downloaded by the notebook from the URLs listed in the metadata.

---

## **File Manifest**

### **Raw Dataset Files**
- **Location**: `data/{Clean,Glass,Plastic,Rubber,Sand}/` (source dataset only, not distributed with the repository)
- **Total Directories**: 5 (categories) + 730 (vials)
- **Total Images**: 13,870 JPEG files
- **Total Size**: ~4.5 GB (estimated)

### **Processed Dataset Files**
- **Location**: object store `images/raw/` (downloaded by the training notebook to `data/images/raw/`)
- **Total Images**: 730 JPEG files
- **Metadata**: object store `images/metadata.csv`
- **Total Size**: ~730 MB (estimated)

### **Held-Out Test Set Files**
- **Location**: object store `test/` (downloaded to `data/test/`)
- **Total Images**: 10 JPEG files
- **Metadata**: object store `test/metadata.csv`

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
| 2026-10-19 | Verified counts against object-store metadata; documented held-out test set and object-store location; removed training-time preprocessing (kept in notebooks) | Jeppe Kristensen |

---

## **Appendix: Sample Metadata**

```csv
filename,class,class_name,vial_id,impurity_type,original_path,num_images_in_series,width,height,format,batch_id,timestamp
Clean_001.jpg,0,clean,Clean_001,none,data/Clean/Clean_001/TLC00-20240326154142-058909.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T13:03:11.963263
Glass_001.jpg,1,defective,Glass_001,glass,data/Glass/Glass_001/TLC00-20240327112026-064005.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T13:03:12.317892
Plastic_001.jpg,1,defective,Plastic_001,plastic,data/Plastic/Plastic_001/TLC00-20240326165823-075117.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T13:03:13.105708
Rubber_001.jpg,1,defective,Rubber_001,rubber,data/Rubber/Rubber_001/TLC00-20240325171030-072108.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T13:03:13.508619
Sand_001.jpg,1,defective,Sand_001,sand,data/Sand/Sand_001/TLC00-20240326094240-016412.jpg,19,1600,768,JPEG,BATCH_2024_ICPR,2026-07-23T13:03:13.600618
Clean_151_TLC00-20240326150919-041901.jpg,0,clean,Clean_151,none,data/Clean/Clean_151/TLC00-20240326150919-041901.jpg,19,1600,768,JPEG,TEST_SET,2026-07-29T19:11:01.414901
Glass_039_TLC00-20240327111626-061101.jpg,1,defective,Glass_039,glass,data/Glass/Glass_039/TLC00-20240327111626-061101.jpg,19,1600,768,JPEG,TEST_SET,2026-07-29T19:11:01.415854
```

The last two rows are from the held-out test set (`data/test/metadata.csv`); the rest are from the training metadata (`data/images/metadata.csv`).

---

*Last updated: 2026-10-19*