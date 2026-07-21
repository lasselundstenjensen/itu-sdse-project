"""
Model Evaluation Module

Provides utilities for evaluating model performance.
Includes accuracy, confusion matrix, classification report, and other metrics.
"""

import sys
from typing import Any, Literal

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..utils import print_section_header


def calculate_accuracy(
    y_true: pd.Series | list | Any,
    y_pred: pd.Series | list | Any,
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
    y_true: pd.Series | list | Any,
    y_pred: pd.Series | list | Any,
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
    y_true: pd.Series | list | Any,
    y_pred: pd.Series | list | Any,
    output_dict: bool = False,
) -> dict | str:
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
    y_true: pd.Series | list | Any,
    y_pred: pd.Series | list | Any,
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
    y_true: pd.Series | list | Any,
    y_pred: pd.Series | list | Any,
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
    y_true: pd.Series | list | Any,
    y_pred: pd.Series | list | Any,
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
