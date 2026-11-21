import json

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from vial_classifier.config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    EARLY_STOP_PATIENCE,
    EXPERIMENT_NAME,
    IMAGE_SIZE,
    IMAGE_STATISTICS_PATH,
    LEARNING_RATE,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_STATE_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_RESULTS_PATH,
    NUM_CLASSES,
    NUM_EPOCHS,
    PYTORCH_MODEL_PATH,
    RANDOM_STATE,
    SCHEDULER_PATIENCE,
    THRESHOLD,
    MODEL_ARTIFACT_PATH,
    PROJ_ROOT,
)
from vial_classifier.dataset import create_dataloaders, load_image_dataset, split_dataset
from vial_classifier.modeling.model import SimpleCNN

from vial_classifier.modeling.drift_baseline import (
    compute_baseline_statistics,
    save_and_log_baseline,
)
from vial_classifier.modeling.predict import SimpleLoggingWrapper

app = typer.Typer()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn_model(
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=None,
    random_seed=RANDOM_STATE,
):
    """Train the SimpleCNN model on glass vial images."""
    if device is None:
        device = get_device()

    print(f"\n{'=' * 60}")
    print("TRAINING CNN MODEL")
    print(f"{'=' * 60}")
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_f1": [], "val_f1": [], "val_f1_macro": [],
        "best_val_f1_macro": 0.0, "best_model_state": None,
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

        for images, labels in train_loader:
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
            predicted = (probs >= THRESHOLD).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average="binary")

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
                predicted = (probs >= THRESHOLD).float().view(-1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.view(-1)).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average="binary")
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_accuracy"].append(epoch_train_accuracy)
        history["val_accuracy"].append(epoch_val_accuracy)
        history["train_f1"].append(epoch_train_f1)
        history["val_f1"].append(epoch_val_f1)
        history["val_f1_macro"].append(epoch_val_f1_macro)

        # Step the LR scheduler on validation macro-F1
        scheduler.step(epoch_val_f1_macro)

        # Save best model based on validation macro-F1
        if epoch_val_f1_macro > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = epoch_val_f1_macro
            # Deep-copy the weights so the "best" state does not keep changing
            history["best_model_state"] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"\nEarly stopping: no macro-F1 improvement for "
                    f"{EARLY_STOP_PATIENCE} consecutive epochs"
                )
                break

        # Print epoch results
        print(f"Epoch [{epoch + 1}/{num_epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}):")
        print(
            f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} "
            f"| Train F1: {epoch_train_f1:.4f}"
        )
        print(
            f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f} "
            f"| Val F1: {epoch_val_f1:.4f} | Val Macro F1: {epoch_val_f1_macro:.4f}"
        )

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
    if history["best_model_state"] is not None:
        model.load_state_dict(history["best_model_state"])
        print(
            f"\nRestored best model weights "
            f"(best val macro F1: {history['best_val_f1_macro']:.4f})"
        )

    print("\nTraining complete!")
    print(f"Best validation macro F1-score: {history['best_val_f1_macro']:.4f}")

    return model, history


def tune_threshold(model, val_loader, device=None, threshold_range=None):
    """Tune the classification threshold to maximize macro F1 on the validation set.

    Macro-F1 is used because maximizing the positive-class F1 alone rewards
    predicting "defective" for every image (defective is the majority class).
    """
    if device is None:
        device = get_device()

    if threshold_range is None:
        # Fine grid: the model's probabilities can compress into a narrow band
        threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    print(f"\n{'=' * 60}")
    print("TUNING CLASSIFICATION THRESHOLD")
    print(f"{'=' * 60}")
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

    best_threshold = THRESHOLD
    best_f1 = 0.0

    for t in threshold_range:
        preds = (all_probs >= t).astype(int)
        f1_macro = f1_score(all_labels, preds, average="macro")
        f1_positive = f1_score(all_labels, preds, average="binary")

        print(
            f"  Threshold {t}: MacroF1={f1_macro:.4f}, F1(defective)={f1_positive:.4f}, "
            f"Acc={accuracy_score(all_labels, preds):.4f}"
        )

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = t

    print(f"\nOptimal threshold: {best_threshold} (macro F1-score: {best_f1:.4f})")

    # Log to MLflow (only if run is active)
    if mlflow.active_run():
        mlflow.log_param("optimal_threshold", best_threshold)
        mlflow.log_metric("optimal_threshold_macro_f1", best_f1)

    return best_threshold


def evaluate_cnn_model(model, test_loader, device=None, threshold=THRESHOLD):
    """Evaluate the model on the test set: accuracy, F1, precision, recall, ROC-AUC."""
    if device is None:
        device = get_device()

    print(f"\n{'=' * 60}")
    print("EVALUATING CNN MODEL")
    print(f"{'=' * 60}")
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

            logits = model(images)
            probs = torch.sigmoid(logits)

            probs = probs.cpu().numpy()
            preds = (probs >= threshold).astype(float).reshape(-1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    roc_auc = roc_auc_score(all_labels, all_probs)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)

    print("\nTest Set Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print("  [[TN, FP],")
    print(f"   [FN, TP]] = {conf_matrix.tolist()}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Log metrics to MLflow (only if run is active)
    if mlflow.active_run():
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_roc_auc", roc_auc)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
    }
    
def setup_mlflow_tracking(experiment_name=EXPERIMENT_NAME):
    """Point MLflow at the local state dir and create the experiment if needed."""
    MLFLOW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment_name)


def calculate_image_statistics(data_loader):
    """Calculate mean and standard deviation per channel for a dataset."""
    print("Calculating image statistics...")

    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        batch_images = images.numpy()
        batch_size = batch_images.shape[0]

        # Reshape to (batch_size, -1, 3) for channel-wise stats
        batch_images = batch_images.transpose(0, 2, 3, 1).reshape(batch_size, -1, 3)

        batch_mean = batch_images.mean(axis=1)
        batch_std = batch_images.std(axis=1)

        total_mean += batch_mean.sum(axis=0)
        total_std += batch_std.sum(axis=0)
        total_samples += batch_size

    overall_mean = (total_mean / total_samples).tolist()
    overall_std = (total_std / total_samples).tolist()

    print(f"Image statistics: Mean={overall_mean}, Std={overall_std}")
    return {
        "mean": overall_mean,
        "std": overall_std,
        "total_samples": total_samples,
    }


# ---- CLI ----


@app.command()
def main(num_epochs: int = NUM_EPOCHS, learning_rate: float = LEARNING_RATE):
    """Train, tune and evaluate the CNN, save the artifacts and log everything to MLflow."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow_tracking()
    device = get_device()

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f"Class names saved to {CLASS_NAMES_PATH}")

    train_dataset, val_dataset, test_dataset = split_dataset(load_image_dataset())
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # Save the image statistics of the training set as an artifact
    image_stats = calculate_image_statistics(train_loader)
    with open(IMAGE_STATISTICS_PATH, "w") as f:
        json.dump(image_stats, f, indent=2)
    print(f"Image statistics saved to {IMAGE_STATISTICS_PATH}")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    with mlflow.start_run(run_name="training") as run:
        # Log parameters to MLflow
        mlflow.log_param("model_architecture", "SimpleCNN")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("initial_threshold", THRESHOLD)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("image_size", f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
        mlflow.log_param("random_seed", RANDOM_STATE)

        # Log model architecture details
        model_params = sum(p.numel() for p in SimpleCNN().parameters())
        trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
        mlflow.log_param("total_parameters", model_params)
        mlflow.log_param("trainable_parameters", trainable_params)

        cnn_model, _ = train_cnn_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            random_seed=RANDOM_STATE,
        )

        # Tune threshold on validation set
        optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)

        # Evaluate on test set with optimal threshold
        test_results = evaluate_cnn_model(
            cnn_model, test_loader, device=device, threshold=optimal_threshold
        )

        print("\nSaving model artifacts...")

        torch.save(
            {
                "model_state_dict": cnn_model.state_dict(),
                "architecture": SimpleCNN.__name__,
                "input_shape": (3, 64, 64),
                "output_shape": (1,),
                "threshold": optimal_threshold,
                "num_classes": NUM_CLASSES,
            },
            PYTORCH_MODEL_PATH,
        )
        print(f"CNN model saved to {PYTORCH_MODEL_PATH}")

        with open(MODEL_RESULTS_PATH, "w") as results_file:
            json.dump(test_results, results_file, indent=2)
        print(f"Model results saved to {MODEL_RESULTS_PATH}")

        mlflow.log_param("classification_threshold", optimal_threshold)
        
        # Drift baseline on the validation set (compared against live traffic by /drift)
        baseline_stats = compute_baseline_statistics(
            cnn_model, val_loader, device=device, threshold=optimal_threshold
        )
        save_and_log_baseline(baseline_stats)

        # Log the pyfunc model with the SimpleLoggingWrapper as the model artifact
        wrapper = SimpleLoggingWrapper(model=cnn_model)
        mlflow.pyfunc.log_model(
            MODEL_ARTIFACT_PATH,
            python_model=wrapper,
            code_paths=[str(PROJ_ROOT / "vial_classifier")],
        )
        print("Logged pyfunc model to the MLflow run")

    print("\n" + "=" * 50)
    print("TRAINING JOB COMPLETE")
    print("=" * 50)
    print("\nSummary:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Test F1-score: {test_results['f1_score']:.4f}")
    print(f"  MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    app()
