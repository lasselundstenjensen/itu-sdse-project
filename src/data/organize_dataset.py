#!/usr/bin/env python3
"""
Dataset Organization Script for Pharmaceutical Vial Images

This script organizes the raw vial images from the ICPR 2024 dataset:
- Extracts one representative image per vial (from multiple rotations)
- Creates a metadata CSV file with vial information
- Organizes images into a standardized structure

Dataset source: G. Rosati, K. Marchesini, L. Lumetti, F. Sartori, B. Balboni, 
F. Begarani, L. Vescovi, F. Bolelli, C. Grana (2024). 
"Identifying Impurities in Liquids of Pharmaceutical Vials." 
27th International Conference on Pattern Recognition (ICPR).

Dataset URL: https://ditto.ing.unimore.it/residual/
"""

import csv
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
RAW_DATA_DIR = Path("data")
ORGANIZED_DATA_DIR = Path("data/images/raw")
METADATA_PATH = Path("data/images/metadata.csv")

# Vial categories and their labels
CATEGORIES = {
    "Clean": 0,      # Good quality
    "Glass": 1,      # Defective - glass impurity
    "Plastic": 1,    # Defective - plastic impurity  
    "Rubber": 1,     # Defective - rubber impurity
    "Sand": 1,       # Defective - sand impurity
}

# Reverse mapping for metadata
CLASS_NAMES = {
    0: "clean",
    1: "defective"
}

# Impurity type mapping
IMPURITY_TYPES = {
    "Clean": "none",
    "Glass": "glass",
    "Plastic": "plastic",
    "Rubber": "rubber", 
    "Sand": "sand"
}


def get_image_info(image_path: Path) -> Tuple[int, int, str]:
    """
    Get image dimensions and mode using system tools.
    
    Parameters
    ----------
    image_path : Path
        Path to the image file.
        
    Returns
    -------
    Tuple[int, int, str]
        (width, height, mode/format)
    """
    # Use file command to get image info
    import subprocess
    try:
        result = subprocess.run(
            ["file", str(image_path)],
            capture_output=True,
            text=True
        )
        info = result.stdout.strip()
        
        # Parse dimensions from file output
        if "1600x768" in info:
            return 1600, 768, "JPEG"
        elif "x" in info:
            # Extract dimensions
            size_part = info.split("x")[0].split(", ")[1] if "x" in info else "0x0"
            if "x" in size_part:
                width, height = size_part.split("x")
                return int(width), int(height), "JPEG"
        return 0, 0, "UNKNOWN"
    except Exception:
        return 0, 0, "UNKNOWN"


def extract_sample_images(raw_dir: Path, organized_dir: Path) -> Dict[str, Dict]:
    """
    Extract one sample image per vial from the raw dataset.
    
    The raw dataset has structure: data/{category}/{vial_id}/{timestamp}-{sequence}.jpg
    We want to select one representative image per vial.
    
    Parameters
    ----------
    raw_dir : Path
        Path to the raw dataset directory.
    organized_dir : Path
        Path to save organized images.
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary with vial information: {category: {vial_id: image_path}}
    """
    vial_info = {}
    
    print("Scanning raw dataset...")
    
    # Create organized directory structure
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    for category in CATEGORIES.keys():
        category_dir = raw_dir / category
        if not category_dir.exists():
            print(f"Warning: Category directory {category_dir} not found")
            continue
            
        vial_dirs = [d for d in category_dir.iterdir() if d.is_dir()]
        print(f"Found {len(vial_dirs)} vials in {category} category")
        
        vial_info[category] = {}
        
        for vial_dir in vial_dirs:
            vial_name = vial_dir.name
            
            # Get all images in this vial directory
            images = list(vial_dir.glob("*.jpg"))
            
            if not images:
                print(f"Warning: No images found in {vial_dir}")
                continue
            
            # Select the first image as representative (or could select middle one)
            # For simplicity, we'll use the first image
            sample_image = images[0]
            
            # Create new filename: {category}_{vial_number}.jpg
            vial_number = vial_name.split("_")[1]  # Extract number from "Clean_001"
            new_filename = f"{category}_{vial_number}.jpg"
            new_path = organized_dir / new_filename
            
            # Copy the image to organized directory
            shutil.copy2(sample_image, new_path)
            
            # Store vial information
            width, height, img_format = get_image_info(sample_image)
            vial_info[category][vial_name] = {
                "path": new_path,
                "filename": new_filename,
                "category": category,
                "vial_id": vial_name,
                "class": CATEGORIES[category],
                "class_name": CLASS_NAMES[CATEGORIES[category]],
                "impurity_type": IMPURITY_TYPES[category],
                "original_path": str(sample_image),
                "num_images": len(images),
                "width": width,
                "height": height,
                "format": img_format,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"  Processed {vial_name}: {len(images)} images -> {new_filename}")
    
    return vial_info


def create_metadata_csv(vial_info: Dict[str, Dict], metadata_path: Path) -> Path:
    """
    Create metadata CSV file from vial information.
    
    Parameters
    ----------
    vial_info : Dict[str, Dict]
        Vial information dictionary from extract_sample_images.
    metadata_path : Path
        Path to save the metadata CSV file.
        
    Returns
    -------
    Path
        Path to the created metadata file.
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV header
    headers = [
        "filename",
        "class", 
        "class_name",
        "vial_id",
        "impurity_type",
        "original_path",
        "num_images_in_series",
        "width",
        "height", 
        "format",
        "batch_id",
        "timestamp"
    ]
    
    print(f"Creating metadata CSV: {metadata_path}")
    
    with open(metadata_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        row_count = 0
        for category, vials in vial_info.items():
            for vial_name, info in vials.items():
                row = {
                    "filename": info["filename"],
                    "class": info["class"],
                    "class_name": info["class_name"],
                    "vial_id": info["vial_id"],
                    "impurity_type": info["impurity_type"],
                    "original_path": info["original_path"],
                    "num_images_in_series": info["num_images"],
                    "width": info["width"],
                    "height": info["height"],
                    "format": info["format"],
                    "batch_id": f"BATCH_2024_ICPR",
                    "timestamp": info["timestamp"]
                }
                writer.writerow(row)
                row_count += 1
        
        print(f"Created metadata with {row_count} entries")
    
    return metadata_path


def analyze_dataset(vial_info: Dict[str, Dict]) -> Dict:
    """
    Analyze the dataset and print statistics.
    
    Parameters
    ----------
    vial_info : Dict[str, Dict]
        Vial information dictionary.
        
    Returns
    -------
    Dict
        Dataset statistics.
    """
    stats = {
        "total_vials": 0,
        "total_images": 0,
        "categories": {},
        "class_distribution": {0: 0, 1: 0},
        "impurity_distribution": {},
    }
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    for category, vials in vial_info.items():
        category_count = len(vials)
        total_images = sum(info["num_images"] for info in vials.values())
        class_label = CATEGORIES[category]
        
        stats["total_vials"] += category_count
        stats["total_images"] += total_images
        stats["categories"][category] = category_count
        stats["class_distribution"][class_label] += category_count
        
        impurity_type = IMPURITY_TYPES[category]
        stats["impurity_distribution"][impurity_type] = category_count
        
        print(f"\n{category}:")
        print(f"  Vials: {category_count}")
        print(f"  Total images: {total_images}")
        print(f"  Class: {class_label} ({CLASS_NAMES[class_label]})")
        print(f"  Impurity type: {impurity_type}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Total vials: {stats['total_vials']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Class distribution:")
    for class_label, count in stats["class_distribution"].items():
        percentage = (count / stats["total_vials"]) * 100 if stats["total_vials"] > 0 else 0
        print(f"  Class {class_label} ({CLASS_NAMES[class_label]}): {count} ({percentage:.1f}%)")
    
    print(f"\nImpurity distribution:")
    for impurity, count in stats["impurity_distribution"].items():
        if impurity != "none":
            print(f"  {impurity}: {count}")
    
    return stats


def main():
    """Main function to organize the dataset."""
    print("Pharmaceutical Vial Image Dataset Organization")
    print("="*60)
    print(f"Source: {RAW_DATA_DIR}")
    print(f"Destination: {ORGANIZED_DATA_DIR}")
    print(f"Metadata: {METADATA_PATH}")
    print()
    
    # Extract sample images
    vial_info = extract_sample_images(RAW_DATA_DIR, ORGANIZED_DATA_DIR)
    
    # Analyze dataset
    stats = analyze_dataset(vial_info)
    
    # Create metadata
    metadata_path = create_metadata_csv(vial_info, METADATA_PATH)
    
    print(f"\n✅ Dataset organization complete!")
    print(f"   Organized images: {ORGANIZED_DATA_DIR}")
    print(f"   Metadata file: {metadata_path}")
    print(f"   Total vials: {stats['total_vials']}")
    print(f"   Total images processed: {stats['total_images']}")


if __name__ == "__main__":
    main()