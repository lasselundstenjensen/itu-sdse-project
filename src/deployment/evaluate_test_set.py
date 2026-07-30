"""
Simplified Evaluation Script for Test Set

Loads model from data/output/, runs inference on test images, and saves results.

Usage:
    python -m src.deployment.evaluate_test_set

Output:
    - Prints per-image results to console
    - Saves results to data/output/output.csv
"""

from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from src.models.train import SimpleCNN


def main():
    """Main evaluation function."""
    # Config
    OUTPUT_DIR = Path("data/output")
    TEST_DIR = Path("data/test")
    METADATA_PATH = TEST_DIR / "metadata.csv"
    MODEL_PATH = OUTPUT_DIR / "model.pth"
    THRESHOLD_PATH = OUTPUT_DIR / "threshold.txt"
    RESULT_PATH = OUTPUT_DIR / "output.csv"

    print("=" * 60)
    print("SIMPLIFIED TEST SET EVALUATION")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Threshold path: {THRESHOLD_PATH}")
    print(f"Result path: {RESULT_PATH}")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    
    # Try to load as checkpoint dict first
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # It's a checkpoint dictionary
            model = SimpleCNN()
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # It's a full model object (from MLflow)
            model = checkpoint
    except Exception:
        # Fallback: load with safe globals for MLflow-pickled models
        torch.serialization.add_safe_globals([SimpleCNN])
        model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    model.eval()
    print("Model loaded successfully")

    # Load threshold
    if THRESHOLD_PATH.exists():
        threshold = float(THRESHOLD_PATH.read_text())
        print(f"Loaded threshold from file: {threshold}")
    else:
        # Fallback to checkpoint threshold (only available if loaded from checkpoint dict)
        threshold = 0.5
        if 'checkpoint' in locals() and isinstance(checkpoint, dict):
            threshold = checkpoint.get('threshold', 0.5)
        print(f"Using threshold from checkpoint: {threshold}")

    # Load test metadata
    print(f"\nLoading test metadata from {METADATA_PATH}...")
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(metadata)} test images")

    # Transforms (matching the training transforms from config)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Run inference
    print("\nRunning inference...")
    results = []
    for idx, row in metadata.iterrows():
        img_path = TEST_DIR / row['filename']
        
        # Load and transform image
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensor = transform(img).unsqueeze(0)

        # Run model inference
        with torch.no_grad():
            probability = model(tensor).item()

        # Predict class
        predicted_class = 1 if probability >= threshold else 0

        # Map predicted class to name
        predicted_class_name = 'clean' if predicted_class == 0 else 'defective'

        results.append({
            'filename': row['filename'],
            'class': row['class'],
            'class_name': row['class_name'],
            'vial_id': row['vial_id'],
            'impurity_type': row['impurity_type'],
            'model_probability': probability,
            'predicted_class': predicted_class,
            'predicted_class_name': predicted_class_name,
            'threshold': threshold,
        })

        # Print per-image results
        print(f"[{idx+1}/{len(metadata)}] {row['filename']}: "
              f"prob={probability:.4f}, "
              f"predicted={predicted_class_name}, "
              f"threshold={threshold}")

    # Save results
    results_df = pd.DataFrame(results)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULT_PATH, index=False)
    print(f"\nResults saved to {RESULT_PATH}")
    print(f"Results shape: {results_df.shape}")


if __name__ == "__main__":
    try:
        main()
        print("\nEvaluation complete!")
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        import traceback
        traceback.print_exc()
        __import__('sys').exit(1)
