"""Training job: data -> features -> train -> tune -> evaluate -> baseline -> register.

CLI: ``python -m vial_classifier.train``
"""

import json

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from vial_classifier import config, registry
from vial_classifier.data import (
    create_dataloaders,
    load_image_dataset,
    prepare_directories,
    print_class_distribution,
    pull_data_from_dvc,
    split_dataset,
)
from vial_classifier.drift import (
    compute_baseline_statistics,
    log_baseline_to_mlflow,
    save_baseline_stats,
)
from vial_classifier.evaluate import evaluate_cnn_model, tune_threshold
from vial_classifier.features import (
    calculate_image_statistics,
    save_class_names,
    save_image_statistics,
)
from vial_classifier.model import SimpleCNN, SimpleLoggingWrapper, get_device


def print_section_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _run_epoch_eval(model, loader, criterion, device, threshold):
    """One pass over `loader` without gradients. Returns (loss, accuracy, preds, labels)."""
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= threshold).float().view(-1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), correct / total, all_preds, all_labels


def train_cnn_model(train_loader, val_loader, num_epochs=config.NUM_EPOCHS,
                    learning_rate=config.LEARNING_RATE, device=None,
                    random_seed=config.RANDOM_STATE):
    """Train the SimpleCNN model on glass vial images."""
    device = device or get_device()
    threshold = config.THRESHOLD

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

    # The dataset is imbalanced, so the positive (defective) class is
    # down-weighted by the negative/positive count ratio.
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
        optimizer, mode="max", factor=0.5, patience=config.SCHEDULER_PATIENCE,
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

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= threshold).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average="binary")

        # Validation phase
        model.eval()
        epoch_val_loss, epoch_val_accuracy, all_val_preds, all_val_labels = _run_epoch_eval(
            model, val_loader, criterion, device, threshold,
        )
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average="binary")
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_accuracy"].append(epoch_train_accuracy)
        history["val_accuracy"].append(epoch_val_accuracy)
        history["train_f1"].append(epoch_train_f1)
        history["val_f1"].append(epoch_val_f1)
        history["val_f1_macro"].append(epoch_val_f1_macro)

        scheduler.step(epoch_val_f1_macro)

        # Keep the best model based on validation macro-F1
        if epoch_val_f1_macro > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = epoch_val_f1_macro
            # Deep-copy the weights so the "best" state does not keep changing
            history["best_model_state"] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no macro-F1 improvement for {config.EARLY_STOP_PATIENCE} consecutive epochs")
                break

        print(f"Epoch [{epoch + 1}/{num_epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}):")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | Train F1: {epoch_train_f1:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f} | Val F1: {epoch_val_f1:.4f} | Val Macro F1: {epoch_val_f1_macro:.4f}")

        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

    if history["best_model_state"] is not None:
        model.load_state_dict(history["best_model_state"])
        print(f"\nRestored best model weights (best val macro F1: {history['best_val_f1_macro']:.4f})")

    print("\nTraining complete!")
    print(f"Best validation macro F1-score: {history['best_val_f1_macro']:.4f}")

    return model, history


def _log_training_params():
    mlflow.log_param("model_architecture", "SimpleCNN")
    mlflow.log_param("batch_size", config.BATCH_SIZE)
    mlflow.log_param("num_epochs", config.NUM_EPOCHS)
    mlflow.log_param("learning_rate", config.LEARNING_RATE)
    mlflow.log_param("initial_threshold", config.THRESHOLD)
    mlflow.log_param("num_classes", config.NUM_CLASSES)
    mlflow.log_param("image_size", "64x64")
    mlflow.log_param("random_seed", config.RANDOM_STATE)

    parameters = list(SimpleCNN().parameters())
    mlflow.log_param("total_parameters", sum(p.numel() for p in parameters))
    mlflow.log_param("trainable_parameters", sum(p.numel() for p in parameters if p.requires_grad))


def _save_model_artifacts(model, optimal_threshold, test_results):
    """Save the checkpoint and results locally, and log the pyfunc model to MLflow."""
    print("\nSaving model artifacts...")

    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": SimpleCNN.__name__,
        "input_shape": (3, 64, 64),
        "output_shape": (1,),
        "threshold": optimal_threshold,
        "num_classes": config.NUM_CLASSES,
    }, config.PYTORCH_MODEL_PATH)
    print(f"CNN model saved to {config.PYTORCH_MODEL_PATH}")

    with open(config.MODEL_RESULTS_PATH, "w") as results_file:
        json.dump(test_results, results_file, indent=2)
    print(f"Model results saved to {config.MODEL_RESULTS_PATH}")

    mlflow.log_param("classification_threshold", optimal_threshold)

    # NOTE: as in the original notebook, the wrapper uses the default
    # threshold (config.THRESHOLD), not optimal_threshold.
    wrapper = SimpleLoggingWrapper(model=model)
    # code_paths ships the package with the model, so the pickled wrapper
    # can be loaded wherever the model is served.
    mlflow.pyfunc.log_model(
        config.ARTIFACT_PATH,
        python_model=wrapper,
        code_paths=[str(config.PACKAGE_DIR)],
    )
    print("Logged pyfunc model to the MLflow run")


def run_training():
    """Run the whole training job and register the model in Staging."""
    prepare_directories()
    registry.setup_mlflow_tracking(config.EXPERIMENT_NAME)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # Data
    pull_data_from_dvc()
    image_dataset = load_image_dataset()
    print_class_distribution(image_dataset)

    # Features
    save_class_names()
    train_dataset, val_dataset, test_dataset = split_dataset(image_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    save_image_statistics(calculate_image_statistics(train_loader))

    # Training
    print_section_header("CNN MODEL TRAINING")
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    device = get_device()

    if mlflow.active_run():
        mlflow.end_run()
        print("Ended previous active MLflow run")

    with mlflow.start_run(run_name="training_notebook") as run:
        _log_training_params()

        cnn_model, _ = train_cnn_model(train_loader, val_loader, device=device)
        optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)
        test_results = evaluate_cnn_model(cnn_model, test_loader, device=device, threshold=optimal_threshold)

        baseline_stats = compute_baseline_statistics(
            cnn_model, val_loader, device=device, threshold=optimal_threshold,
        )
        save_baseline_stats(baseline_stats)
        log_baseline_to_mlflow(baseline_stats)

        _save_model_artifacts(cnn_model, optimal_threshold, test_results)
        model_details = registry.register_and_stage_model(run.info.run_id)

    print("\n" + "=" * 50)
    print("TRAINING JOB COMPLETE")
    print("=" * 50)
    print("\nSummary:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Test F1-score: {test_results['f1_score']:.4f}")
    print(f"  Registered model: {config.MODEL_NAME} version {model_details.version} (Staging)")


if __name__ == "__main__":
    run_training()