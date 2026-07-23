"""
Model Training Module

Handles training of PyTorch CNN model for glass vial image classification.
Includes training loop, evaluation, threshold tuning, and MLflow integration.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score
)
from torch.utils.data import DataLoader

from ..config import (
    ARTIFACT_DIR,
    ARTIFACT_PATH,
    BATCH_SIZE,
    DEVICE,
    EXPERIMENT_NAME,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_EPOCHS,
    PYTORCH_MODEL_PATH,
    RANDOM_STATE,
    THRESHOLD,
)
from ..utils import (
    get_predictions,
    print_section_header,
    tensor_to_numpy,
)
from .evaluate import (
    get_confusion_matrix,
    get_classification_report,
    calculate_accuracy,
)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN model for binary classification of glass vial images.
    
    Architecture:
    - 3 convolutional layers with max pooling
    - 2 fully connected layers with dropout
    - Sigmoid output for binary classification
    
    Input: 64x64 RGB images (3 channels)
    Output: Single value (0-1) representing probability of class 0 (good)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 3 max pooling layers: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, 64, 64)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1) with values in [0, 1]
        """
        # Convolutional layers with ReLU and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x


def setup_mlflow() -> str:
    """
    Setup MLflow experiment and directories for PyTorch CNN training.

    Creates necessary directories and sets the active experiment.

    Returns
    -------
    str
        The experiment name that was set.
    """
    print(f"Setting up MLflow experiment: {EXPERIMENT_NAME}")

    # Ensure directories exist
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    Path("./mlruns").mkdir(parents=True, exist_ok=True)
    Path("./mlruns/.trash").mkdir(parents=True, exist_ok=True)

    # Set experiment
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        print(f"Warning: Could not set MLflow experiment: {e}")

    print(f"MLflow experiment set to: {EXPERIMENT_NAME}")
    return EXPERIMENT_NAME


# =============================================================================
# CNN TRAINING FUNCTIONS
# =============================================================================

def train_cnn_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: torch.device = DEVICE,
    random_seed: int = RANDOM_STATE,
) -> Tuple[SimpleCNN, Dict]:
    """
    Train the SimpleCNN model on glass vial images.

    Parameters
    ----------
    train_loader : DataLoader
        Training DataLoader with image batches and labels.
    val_loader : DataLoader
        Validation DataLoader for monitoring performance.
    num_epochs : int, default=NUM_EPOCHS from config
        Number of training epochs.
    learning_rate : float, default=LEARNING_RATE from config
        Learning rate for Adam optimizer.
    device : torch.device, default=DEVICE from config
        Device to use for training (cuda or cpu).
    random_seed : int, default=RANDOM_STATE from config
        Random seed for reproducibility.

    Returns
    -------
    Tuple[SimpleCNN, Dict]
        - Trained SimpleCNN model
        - Training history with metrics per epoch
    """
    print(f"\n{'='*60}")
    print("TRAINING CNN MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Set seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    
    # Use Binary Cross Entropy loss for binary classification
    criterion = nn.BCELoss()
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1': [],
        'val_f1': [],
        'best_val_f1': 0.0,
        'best_model_state': None,
    }

    print(f"\nStarting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_running_loss += loss.item()
            predicted = (outputs >= THRESHOLD).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            # Store predictions and labels for F1 calculation
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                predicted = (outputs >= THRESHOLD).float().view(-1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.view(-1)).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='binary')

        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_accuracy'].append(epoch_train_accuracy)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['train_f1'].append(epoch_train_f1)
        history['val_f1'].append(epoch_val_f1)

        # Save best model based on validation F1 score
        if epoch_val_f1 > history['best_val_f1']:
            history['best_val_f1'] = epoch_val_f1
            history['best_model_state'] = model.state_dict().copy()

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | Train F1: {epoch_train_f1:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f} | Val F1: {epoch_val_f1:.4f}")

        # Log to MLflow
        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_train_accuracy, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
        mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
        mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)

    # Load best model weights
    if history['best_model_state'] is not None:
        model.load_state_dict(history['best_model_state'])
        print(f"\nRestored best model weights (best val F1: {history['best_val_f1']:.4f})")

    print(f"\nTraining complete!")
    print(f"Best validation F1-score: {history['best_val_f1']:.4f}")

    return model, history


def evaluate_cnn_model(
    model: SimpleCNN,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
    threshold: float = THRESHOLD,
) -> Dict:
    """
    Evaluate CNN model on test dataset.

    Computes comprehensive metrics: accuracy, F1-score, precision, recall, ROC-AUC.
    Generates confusion matrix and classification report.

    Parameters
    ----------
    model : SimpleCNN
        Trained CNN model to evaluate.
    test_loader : DataLoader
        Test DataLoader with image batches and labels.
    device : torch.device, default=DEVICE from config
        Device to use for evaluation.
    threshold : float, default=THRESHOLD from config
        Classification threshold for converting probabilities to class predictions.

    Returns
    -------
    Dict
        Dictionary with evaluation metrics.
    """
    print(f"\n{'='*60}")
    print("EVALUATING CNN MODEL")
    print(f"{'='*60}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Threshold: {threshold}")

    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            outputs = model(images)
            
            # Store probabilities and predictions
            probs = outputs.cpu().numpy()
            preds = (outputs >= threshold).float().view(-1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    # ROC-AUC requires probabilities, not binary predictions
    roc_auc = roc_auc_score(all_labels, all_probs)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_report = classification_report(all_labels, all_preds, output_dict=True)

    # Print results
    print(f"\nTest Set Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN, FP],")
    print(f"   [FP, TP]] = {conf_matrix.tolist()}")

    print(f"\nClassification Report:")
    for class_name, metrics in class_report.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"  Class {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1-score']:.4f}")
    print(f"  Accuracy: {class_report['accuracy']:.4f}")

    # Log metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_roc_auc", roc_auc)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
    }


def tune_threshold(
    model: SimpleCNN,
    val_loader: DataLoader,
    device: torch.device = DEVICE,
    threshold_range: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
) -> float:
    """
    Tune classification threshold to maximize F1-score on validation set.

    Evaluates model on validation set with different thresholds and selects
    the one that maximizes F1-score.

    Parameters
    ----------
    model : SimpleCNN
        Trained CNN model.
    val_loader : DataLoader
        Validation DataLoader.
    device : torch.device, default=DEVICE from config
        Device to use for evaluation.
    threshold_range : List[float], default=[0.3, 0.4, 0.5, 0.6, 0.7]
        Range of thresholds to evaluate.

    Returns
    -------
    float
        Optimal threshold that maximizes F1-score.
    """
    print(f"\n{'='*60}")
    print("TUNING CLASSIFICATION THRESHOLD")
    print(f"{'='*60}")
    print(f"Threshold range: {threshold_range}")

    model.eval()
    
    all_probs = []
    all_labels = []

    # Get all predictions and labels from validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            outputs = model(images)
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)

    # Test each threshold
    best_threshold = THRESHOLD
    best_f1 = 0.0
    
    results = {}

    for threshold in threshold_range:
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, average='binary')
        
        results[threshold] = {
            'f1_score': f1,
            'accuracy': accuracy_score(all_labels, preds),
            'precision': precision_score(all_labels, preds, average='binary'),
            'recall': recall_score(all_labels, preds, average='binary'),
        }

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        print(f"  Threshold {threshold}: F1={f1:.4f}, Acc={results[threshold]['accuracy']:.4f}, "
              f"Prec={results[threshold]['precision']:.4f}, Rec={results[threshold]['recall']:.4f}")

    print(f"\nOptimal threshold: {best_threshold} (F1-score: {best_f1:.4f})")

    # Log to MLflow
    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_threshold_f1", best_f1)

    return best_threshold


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    cv: int = 10,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """
    Train XGBoost model with randomized hyperparameter search.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    n_iter : int, default=10
        Number of parameter settings sampled.
    cv : int, default=10
        Number of cross-validation folds.
    random_state : int, default=RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    RandomizedSearchCV
        Fitted XGBoost model with best parameters.
    """
    print("\n" + "=" * 50)
    print("TRAINING XGBOOST MODEL")
    print("=" * 50)

    model = XGBRFClassifier(random_state=random_state, n_estimators=100)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"],
    }

    model_grid = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_jobs=-1,
        verbose=3,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
    )

    print("Starting XGBoost training with RandomizedSearchCV...")
    model_grid.fit(X_train, y_train)
    print("XGBoost training complete!")

    return model_grid


def evaluate_and_save_xgboost(
    model_grid: RandomizedSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBRFClassifier:
    """
    Evaluate XGBoost model and save the best estimator.

    Parameters
    ----------
    model_grid : RandomizedSearchCV
        Fitted XGBoost model grid.
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
    XGBRFClassifier
        Best XGBoost estimator.
    """
    print("\n" + "=" * 50)
    print("XGBOOST EVALUATION")
    print("=" * 50)

    # Get best parameters
    best_params = model_grid.best_params_
    print("\nBest XGBoost parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Predictions
    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_pred_train, y_train)
    test_acc = accuracy_score(y_pred_test, y_test)
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test: {test_acc:.4f}")

    # Classification reports
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test))

    print("\nTrain Classification Report:")
    print(classification_report(y_train, y_pred_train))

    # Save best model
    best_model = model_grid.best_estimator_
    XGBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_model.save_model(XGBOOST_MODEL_PATH)
    print(f"\nBest XGBoost model saved to {XGBOOST_MODEL_PATH}")

    return best_model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[LogisticRegression, RandomizedSearchCV]:
    """
    Train Logistic Regression model with MLflow logging.

    Uses RandomizedSearchCV for hyperparameter tuning and logs
    metrics and artifacts to MLflow.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test targets.
    n_iter : int, default=10
        Number of parameter settings sampled.
    cv : int, default=3
        Number of cross-validation folds.
    random_state : int, default=RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    tuple[LogisticRegression, RandomizedSearchCV]
        - Best Logistic Regression estimator
        - Fitted RandomizedSearchCV object
    """
    print("\n" + "=" * 50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 50)

    # Define custom MLflow wrapper for probability prediction
    class LRWrapper(mlflow.pyfunc.PythonModel):
        """Wrapper to predict probabilities instead of classes."""

        def __init__(self, model):
            self.model = model

        def predict(self, context, model_input):
            """Predict probability of positive class."""
            return self.model.predict_proba(model_input)[:, 1]

    # Setup MLflow autologging
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)

    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else None

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression(max_iter=1000, random_state=random_state)

        params = {
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "penalty": ["none", "l1", "l2", "elasticnet"],
            "C": [100, 10, 1.0, 0.1, 0.01],
        }

        model_grid = RandomizedSearchCV(
            model,
            param_distributions=params,
            verbose=3,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )

        print("Starting Logistic Regression training...")
        model_grid.fit(X_train, y_train)
        print("Logistic Regression training complete!")

        best_model = model_grid.best_estimator_

        # Predictions
        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        # Log metrics
        f1 = f1_score(y_test, y_pred_test)
        mlflow.log_metric("f1_score", f1)

        # Log artifacts
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="model")
        mlflow.log_param("data_version", "00000")

        # Save model
        LR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=best_model, filename=LR_MODEL_PATH)
        print(f"Logistic Regression model saved to {LR_MODEL_PATH}")

        # Log custom model for probability prediction
        mlflow.pyfunc.log_model("model", python_model=LRWrapper(best_model))

    # Print best parameters
    best_params = model_grid.best_params_
    print("\nBest Logistic Regression parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Print accuracy
    train_acc = accuracy_score(y_pred_train, y_train)
    test_acc = accuracy_score(y_pred_test, y_test)
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test: {test_acc:.4f}")

    # Classification reports
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test))

    print("\nTrain Classification Report:")
    print(classification_report(y_train, y_pred_train))

    return best_model, model_grid


def save_model_artifacts(
    model: SimpleCNN,
    model_results: dict,
    threshold: float = THRESHOLD,
) -> None:
    """
    Save PyTorch CNN model artifacts.

    Saves:
    - Model state dict and architecture
    - Training results and metrics
    - Optimal classification threshold

    Parameters
    ----------
    model : SimpleCNN
        Trained CNN model to save.
    model_results : dict
        Dictionary with evaluation results and metrics.
    threshold : float, default=THRESHOLD from config
        Optimal classification threshold.
    """
    print("\nSaving CNN model artifacts...")

    # Save model state dict
    PYTORCH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': SimpleCNN.__name__,
        'input_shape': (3, 64, 64),  # RGB 64x64 images
        'output_shape': (1,),  # Binary classification
        'threshold': threshold,
        'num_classes': NUM_CLASSES,
    }, PYTORCH_MODEL_PATH)
    print(f"CNN model saved to {PYTORCH_MODEL_PATH}")

    # Save model results
    MODEL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_RESULTS_PATH, "w") as results_file:
        json.dump(model_results, results_file, indent=2)
    print(f"Model results saved to {MODEL_RESULTS_PATH}")

    # Log threshold to MLflow
    mlflow.log_param("classification_threshold", threshold)


def train_models(
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
) -> dict:
    """
    Complete CNN model training pipeline for image classification.

    If DataLoaders are not provided, loads from the data pipeline.

    Trains the SimpleCNN model, evaluates it, tunes the threshold,
    and saves all artifacts with MLflow integration.

    Parameters
    ----------
    train_loader : DataLoader, optional
        Training DataLoader. If None, loads from data pipeline.
    val_loader : DataLoader, optional
        Validation DataLoader. If None, loads from data pipeline.
    test_loader : DataLoader, optional
        Test DataLoader. If None, loads from data pipeline.

    Returns
    -------
    dict
        Dictionary with trained model, evaluation results, and history.
    """
    print_section_header("CNN MODEL TRAINING")

    # Setup MLflow
    setup_mlflow()

    # Load data if not provided
    if train_loader is None or val_loader is None or test_loader is None:
        from ..data.preprocess import preprocess_data
        train_loader, val_loader, test_loader = preprocess_data()

    # Log parameters to MLflow
    mlflow.log_param("model_architecture", "SimpleCNN")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("initial_threshold", THRESHOLD)
    mlflow.log_param("num_classes", NUM_CLASSES)
    mlflow.log_param("image_size", "64x64")
    mlflow.log_param("random_seed", RANDOM_STATE)

    # Log model architecture details
    model_params = sum(p.numel() for p in SimpleCNN().parameters())
    trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", model_params)
    mlflow.log_param("trainable_parameters", trainable_params)

    # Start MLflow run
    with mlflow.start_run() as run:
        # Train CNN model
        cnn_model, training_history = train_cnn_model(
            train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            random_seed=RANDOM_STATE,
        )

        # Tune threshold on validation set
        optimal_threshold = tune_threshold(
            cnn_model, val_loader, device=DEVICE
        )

        # Evaluate on test set with optimal threshold
        test_results = evaluate_cnn_model(
            cnn_model, test_loader,
            device=DEVICE,
            threshold=optimal_threshold,
        )

        # Save model artifacts
        save_model_artifacts(cnn_model, test_results, threshold=optimal_threshold)

        # Log model to MLflow
        mlflow.pytorch.log_model(
            cnn_model,
            "model",
            registered_model_name=MODEL_NAME,
        )

    return {
        "cnn_model": cnn_model,
        "training_history": training_history,
        "test_results": test_results,
        "optimal_threshold": optimal_threshold,
    }


if __name__ == "__main__":
    """
    Run CNN model training module directly.

    Usage:
        python -m src.models.train
    """
    print("Running CNN model training module...")
    try:
        # Set seeds for reproducibility
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        
        result = train_models()
        print("\nCNN model training complete!")
        print(f"Optimal threshold: {result.get('optimal_threshold')}")
        print(f"Test results: {result.get('test_results')}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
