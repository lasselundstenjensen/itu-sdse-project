"""
Image Classification MLOps Pipeline Package

This package contains the refactored code for glass vial image classification
using PyTorch CNN, organized by MLOps processes:

- config.py: Shared configuration constants for image classification
- utils.py: Common utility functions for image processing
- data/: Data pipeline modules (fetch, preprocess, features) for image data
- models/: Model pipeline modules (train, evaluate, registry) for PyTorch CNN
- deployment/: Deployment modules for MLflow model registry
- pipeline.py: Main pipeline orchestration

The pipeline supports:
- Loading and validating glass vial images (64x64 RGB)
- PyTorch CNN model with data augmentation
- MLflow tracking and model registry
- Threshold tuning for optimal F1-score
"""

__version__ = "0.2.0"
