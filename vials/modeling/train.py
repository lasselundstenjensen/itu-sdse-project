from loguru import logger
import mlflow
import typer
import numpy as np
import json

from vials.modeling.models import SimpleCNN
from vials.modeling.eval import evaluate_cnn_model
from vials.dataset import print_section_header, create_dataloaders, load_splits
from vials.features import get_transforms
from vials.monitoring.drift import compute_baseline_statistics

import torch
import torch.optim as optim
import torch.nn as nn

from vials.config import (
    ARTIFACTS_DIR,
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_STATE_DIR,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    TRAINING,
    DRIFT_BASELINE_STATS_PATH,
    DRIFT_KS_PVALUE_THRESHOLD,
    DRIFT_DEFAULT_DAYS,
    DRIFT_MEAN_DIFF_THRESHOLD,
)

from sklearn.metrics import (
    f1_score,
    accuracy_score
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = TRAINING["num_epochs"]
num_classes = TRAINING["num_classes"]
learning_rate = TRAINING["learning_rate"]
random_seed = TRAINING["random_state"]
patience = TRAINING["scheduler_patience"]
threshold = TRAINING["threshold"]
early_stop_patience = TRAINING["early_stop_patience"]
batch_size = TRAINING["batch_size"]

def train_cnn_model(train_loader, val_loader, device=device):
    """Train the SimpleCNN model on glass vial images."""

    print(f"\n{'='*60}")
    print("TRAINING CNN MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = SimpleCNN().to(device)

    # Weighted BCE-with-logits loss for binary classification. The dataset is
    # imbalanced, so the positive (defective) class is down-weighted by the
    # negative/positive count ratio to rebalance the loss.
    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    print(f"Training class counts: clean(0)={num_negative}, defective(1)={num_positive}")
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    print(f"Loss pos_weight: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Halve the learning rate when validation macro-F1 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=patience)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': [], 'val_f1_macro': [],
        'best_val_f1_macro': 0.0, 'best_model_state': None,
    }
    epochs_without_improvement = 0

    print("\nStarting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for _, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            # Forward pass (raw logits)
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics (logits -> probabilities)
            train_running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= threshold).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

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

                logits = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item()
                probs = torch.sigmoid(logits)
                predicted = (probs >= threshold).float().view(-1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.view(-1)).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='binary')
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_accuracy'].append(epoch_train_accuracy)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['train_f1'].append(epoch_train_f1)
        history['val_f1'].append(epoch_val_f1)
        history['val_f1_macro'].append(epoch_val_f1_macro)

        # Step the LR scheduler on validation macro-F1
        scheduler.step(epoch_val_f1_macro)

        # Save best model based on validation macro-F1
        if epoch_val_f1_macro > history['best_val_f1_macro']:
            history['best_val_f1_macro'] = epoch_val_f1_macro
            # Deep-copy the weights so the "best" state does not keep changing
            history['best_model_state'] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping: no macro-F1 improvement for {early_stop_patience} consecutive epochs")
                break

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}):")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | Train F1: {epoch_train_f1:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f} | Val F1: {epoch_val_f1:.4f} | Val Macro F1: {epoch_val_f1_macro:.4f}")

        # Log to MLflow (only if run is active)
        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

    # Load best model weights
    if history['best_model_state'] is not None:
        model.load_state_dict(history['best_model_state'])
        print(f"\nRestored best model weights (best val macro F1: {history['best_val_f1_macro']:.4f})")

    print(f"\nTraining complete!")
    print(f"Best validation macro F1-score: {history['best_val_f1_macro']:.4f}")

    return model, history

def tune_threshold(model, val_loader, device=device, threshold_range=None, threshold=TRAINING["threshold"]):
    """
    Tune the classification threshold to maximize macro F1 on the validation set.

    Macro-F1 is used because maximizing the positive-class F1 alone rewards
    predicting "defective" for every image (defective is the majority class).
    """
    if threshold_range is None:
        # Fine grid: the model's probabilities can compress into a narrow band
        threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    print(f"\n{'='*60}")
    print("TUNING CLASSIFICATION THRESHOLD")
    print(f"{'='*60}")
    print(f"Threshold range: {threshold_range}")

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)

            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)

    best_threshold = threshold
    best_f1 = 0.0

    for t in threshold_range:
        preds = (all_probs >= t).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro')
        f1_positive = f1_score(all_labels, preds, average='binary')

        print(f"  Threshold {t}: MacroF1={f1_macro:.4f}, F1(defective)={f1_positive:.4f}, Acc={accuracy_score(all_labels, preds):.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = t

    print(f"\nOptimal threshold: {best_threshold} (macro F1-score: {best_f1:.4f})")

    # Log to MLflow
    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_threshold_macro_f1", best_f1)

    return best_threshold


app = typer.Typer()


def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    MLFLOW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment_name)


@app.command()
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    setup_mlflow_tracking(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: {}", mlflow.get_tracking_uri())

    print_section_header("CNN MODEL TRAINING")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if mlflow.active_run():
        mlflow.end_run()
        print("Ended previous active MLflow run")

    run = mlflow.start_run(run_name="training_notebook")

    # Log parameters to MLflow
    mlflow.log_param("model_architecture", "SimpleCNN")
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("initial_threshold", threshold)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("image_size", "64x64")
    mlflow.log_param("random_seed", random_seed)

    # Log model architecture details
    model_params = sum(p.numel() for p in SimpleCNN().parameters())
    trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", model_params)
    mlflow.log_param("trainable_parameters", trainable_params)

    # Create data splits
    train_split, val_split, test_split = load_splits()
    train_transforms, val_transforms = get_transforms()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_split, val_split, test_split,
        train_transforms, val_transforms,
    )

    cnn_model, training_history = train_cnn_model(
        train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        random_seed=random_seed,
    )

    # Tune threshold on validation set
    optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)

    # Evaluate on test set with optimal threshold
    test_results = evaluate_cnn_model(cnn_model, test_loader, device=device, threshold=optimal_threshold)

    baseline_stats = compute_baseline_statistics(cnn_model, val_loader, threshold=optimal_threshold)

    with open(DRIFT_BASELINE_STATS_PATH, 'w') as f:
        json.dump(baseline_stats, f, indent=2)

    # Log baseline statistics to MLflow
    mlflow.log_metric("baseline_image_mean_r", baseline_stats['image_statistics']['mean_r'])
    mlflow.log_metric("baseline_image_mean_g", baseline_stats['image_statistics']['mean_g'])
    mlflow.log_metric("baseline_image_mean_b", baseline_stats['image_statistics']['mean_b'])
    mlflow.log_metric("baseline_image_std_r", baseline_stats['image_statistics']['std_r'])
    mlflow.log_metric("baseline_image_std_g", baseline_stats['image_statistics']['std_g'])
    mlflow.log_metric("baseline_image_std_b", baseline_stats['image_statistics']['std_b'])
    mlflow.log_metric("baseline_class_0_count", baseline_stats['class_distribution']['class_0_count'])
    mlflow.log_metric("baseline_class_1_count", baseline_stats['class_distribution']['class_1_count'])
    mlflow.log_metric("baseline_confidence_mean", baseline_stats['confidence_statistics']['mean'])
    mlflow.log_metric("baseline_confidence_std", baseline_stats['confidence_statistics']['std'])
    mlflow.log_artifact(DRIFT_BASELINE_STATS_PATH, "drift_detection")

if __name__ == "__main__":
    app()
