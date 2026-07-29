"""
Model Evaluation Module for Image Classification

Provides utilities for evaluating PyTorch CNN model performance.
Includes accuracy, confusion matrix, classification report, and tensor handling.
"""

import sys
from typing import Any, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..utils import print_section_header


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Handles the conversion and ensures the result is on CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor to convert.

    Returns
    -------
    np.ndarray
        Converted NumPy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def get_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions and labels from a model on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    data_loader : DataLoader
        DataLoader with batches to predict on.
    device : torch.device, default=torch.device('cpu')
        Device to use for prediction.
    threshold : float, default=0.5
        Threshold for converting probabilities to class predictions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Array of predicted class labels
        - Array of true labels
    """
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels_np = labels.numpy()
            
            outputs = model(images)
            
            # Convert outputs to predictions using threshold
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.view(-1)
            
            preds = (outputs >= threshold).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)

    return np.array(all_preds), np.array(all_labels)


def calculate_accuracy(
    y_true: Union[pd.Series, list, Any],
    y_pred: Union[pd.Series, list, Any],
) -> float:
    """
    Calculate accuracy score.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.

    Returns
    -------
    float
        Accuracy classification score (fraction of correct predictions).
    """
    return accuracy_score(y_true, y_pred)


def get_confusion_matrix(
    y_true: Union[pd.Series, list, Any],
    y_pred: Union[pd.Series, list, Any],
) -> pd.DataFrame:
    """
    Get confusion matrix as a DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.

    Returns
    -------
    pd.DataFrame
        Confusion matrix with Actual vs Predicted labels.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        conf_matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )


def get_classification_report(
    y_true: Union[pd.Series, list, Any],
    y_pred: Union[pd.Series, list, Any],
    output_dict: bool = False,
) -> Union[dict, str]:
    """
    Get classification report.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    output_dict : bool, default=False
        If True, return output as dict. If False, return as string.

    Returns
    -------
    dict or str
        Classification report with precision, recall, f1-score, and support.
    """
    return classification_report(y_true, y_pred, output_dict=output_dict)


def print_confusion_matrix_crosstab(
    y_true: Union[pd.Series, list, Any],
    y_pred: Union[pd.Series, list, Any],
    dataset_name: str = "",
) -> None:
    """
    Print confusion matrix as a crosstab table with margins.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    dataset_name : str, optional
        Name of dataset (e.g., "Test", "Train") for labeling output.
    """
    prefix = f"{dataset_name} " if dataset_name else ""
    crosstab = pd.crosstab(
        y_true,
        y_pred,
        rownames=["Actual"],
        colnames=["Predicted"],
        margins=True,
    )
    print(f"\n{prefix}Confusion Matrix (Actual vs Predicted):\n{crosstab}")


def print_classification_report(
    y_true: Union[pd.Series, list, Any],
    y_pred: Union[pd.Series, list, Any],
    dataset_name: str = "",
) -> None:
    """
    Print classification report.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    dataset_name : str, optional
        Name of dataset (e.g., "Test", "Train") for labeling output.
    """
    prefix = f"{dataset_name} " if dataset_name else ""
    print(f"\n{prefix}Classification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred))


def evaluate_model(
    y_true: Union[pd.Series, list, Any],
    y_pred: Union[pd.Series, list, Any],
    dataset_name: str = "",
    print_crosstab: bool = True,
    print_report: bool = True,
    print_accuracy: bool = True,
) -> dict:
    """
    Comprehensive model evaluation.

    Prints confusion matrix, classification report, and accuracy score.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    dataset_name : str, optional
        Name of dataset for labeling output.
    print_crosstab : bool, default=True
        Whether to print confusion matrix crosstab.
    print_report : bool, default=True
        Whether to print classification report.
    print_accuracy : bool, default=True
        Whether to print accuracy score.

    Returns
    -------
    dict
        Dictionary with evaluation metrics (accuracy, f1_score).
    """
    metrics = {}

    if print_crosstab:
        print_confusion_matrix_crosstab(y_true, y_pred, dataset_name)

    if print_report:
        print_classification_report(y_true, y_pred, dataset_name)

    if print_accuracy:
        accuracy = calculate_accuracy(y_true, y_pred)
        prefix = f"{dataset_name} " if dataset_name else ""
        print(f"\n{prefix}Accuracy: {accuracy:.4f}")
        metrics["accuracy"] = accuracy

    # Also calculate F1 score
    f1 = f1_score(y_true, y_pred, average="weighted")
    prefix = f"{dataset_name} " if dataset_name else ""
    print(f"{prefix}F1 Score (weighted): {f1:.4f}")
    metrics["f1_score"] = f1

    return metrics


def evaluate_both_datasets(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate model on both training and test datasets.

    Parameters
    ----------
    model : estimator
        Fitted model with predict method.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test targets.

    Returns
    -------
    dict
        Dictionary with metrics for both datasets.
    """
    print_section_header("MODEL EVALUATION")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate on test set
    test_metrics = evaluate_model(
        y_test, y_pred_test, dataset_name="Test",
        print_crosstab=True, print_report=True, print_accuracy=True
    )

    # Evaluate on train set
    train_metrics = evaluate_model(
        y_train, y_pred_train, dataset_name="Train",
        print_crosstab=True, print_report=True, print_accuracy=True
    )

    return {
        "train": train_metrics,
        "test": test_metrics,
    }


def plot_confusion_matrix(
    y_true: Union[pd.Series, list, Any],
    y_pred: Union[pd.Series, list, Any],
    class_names: dict = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix visualization.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    class_names : dict, optional
        Dictionary mapping class indices to names.
    title : str, default="Confusion Matrix"
        Title for the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        print("Matplotlib or Seaborn not available. Skipping plot.")
        return

    # Convert to numpy if needed
    y_true = tensor_to_numpy(y_true) if isinstance(y_true, torch.Tensor) else y_true
    y_pred = tensor_to_numpy(y_pred) if isinstance(y_pred, torch.Tensor) else y_pred

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Default class names
    if class_names is None:
        class_names = {0: "Good", 1: "Defective"}

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(class_names.values()),
                yticklabels=list(class_names.values()))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_roc_curve(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    class_names: dict = None,
) -> None:
    """
    Plot ROC curve for the model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    data_loader : DataLoader
        DataLoader with data to evaluate on.
    device : torch.device, default=torch.device('cpu')
        Device to use for prediction.
    class_names : dict, optional
        Dictionary mapping class indices to names.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay
    except ImportError:
        print("Matplotlib not available. Skipping ROC curve plot.")
        return

    model.eval()
    
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels_np = labels.numpy()
            
            outputs = model(images)
            
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.view(-1)
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels_np)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Plot ROC curve
    RocCurveDisplay.from_predictions(all_labels, all_probs)
    plt.title('ROC Curve')
    plt.show()


if __name__ == "__main__":
    """
    Test evaluation module directly.

    Usage:
        python -m src.models.evaluate
    """
    print("Testing evaluation module...")

    # Create dummy data
    import numpy as np

    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    # Test individual functions
    print("\n--- Testing calculate_accuracy ---")
    acc = calculate_accuracy(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\n--- Testing confusion matrix ---")
    conf_df = get_confusion_matrix(y_true, y_pred)
    print(conf_df)

    print("\n--- Testing classification report ---")
    report = get_classification_report(y_true, y_pred)
    print(report)

    print("\n--- Testing evaluate_model ---")
    metrics = evaluate_model(y_true, y_pred, dataset_name="Dummy")
    print(f"Metrics: {metrics}")

    print("\nAll evaluation tests passed!")
