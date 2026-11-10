"""Run the notebook steps from the command line: data, train, export.

Run from the project root:

    python -m vial_classifier.pipeline data
    python -m vial_classifier.pipeline train
    python -m vial_classifier.pipeline export
"""

import json
from pathlib import Path
import sys

import mlflow
import torch

from vial_classifier import config
from vial_classifier.dataset import load_image_dataset, pull_data_from_dvc
from vial_classifier.features import build_features
from vial_classifier.modeling.evaluate import compute_baseline_statistics, evaluate_cnn_model
from vial_classifier.modeling.export import export_model
from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.registry import (
    get_client,
    register_and_stage_model,
    select_and_promote_best_model,
)
from vial_classifier.modeling.train import train_cnn_model, tune_threshold
from vial_classifier.modeling.wrapper import SimpleLoggingWrapper
from vial_classifier.utils import setup_mlflow_tracking


def data():
    """Fetch the DVC-tracked metadata and download the images."""
    pull_data_from_dvc()


def train():
    """Everything notebook 01 did: data -> features -> training -> registration."""
    # Setup: artifact folder and MLflow
    Path(config.ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    client = get_client()

    # Read the data and build the train / validation / test loaders
    pull_data_from_dvc()
    image_dataset = load_image_dataset()
    train_loader, val_loader, test_loader = build_features(image_dataset)

    # Start the MLflow run and log the parameters
    run = mlflow.start_run(run_name="training_pipeline")

    mlflow.log_param("model_architecture", "SimpleCNN")
    mlflow.log_param("batch_size", config.BATCH_SIZE)
    mlflow.log_param("num_epochs", config.NUM_EPOCHS)
    mlflow.log_param("learning_rate", config.LEARNING_RATE)
    mlflow.log_param("initial_threshold", config.THRESHOLD)
    mlflow.log_param("num_classes", config.NUM_CLASSES)
    mlflow.log_param("image_size", "64x64")
    mlflow.log_param("random_seed", config.RANDOM_STATE)

    model_params = sum(p.numel() for p in SimpleCNN().parameters())
    trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", model_params)
    mlflow.log_param("trainable_parameters", trainable_params)

    # Train, tune the threshold on the validation set, evaluate on the test set
    cnn_model, _training_history = train_cnn_model(train_loader, val_loader)
    optimal_threshold = tune_threshold(cnn_model, val_loader)
    test_results = evaluate_cnn_model(cnn_model, test_loader, threshold=optimal_threshold)

    # Drift baseline
    baseline_stats = compute_baseline_statistics(
        cnn_model, val_loader, threshold=optimal_threshold
    )
    with open(config.DRIFT_BASELINE_STATS_PATH, "w") as f:
        json.dump(baseline_stats, f, indent=2)

    mlflow.log_metric("baseline_image_mean_r", baseline_stats["image_statistics"]["mean_r"])
    mlflow.log_metric("baseline_image_mean_g", baseline_stats["image_statistics"]["mean_g"])
    mlflow.log_metric("baseline_image_mean_b", baseline_stats["image_statistics"]["mean_b"])
    mlflow.log_metric("baseline_image_std_r", baseline_stats["image_statistics"]["std_r"])
    mlflow.log_metric("baseline_image_std_g", baseline_stats["image_statistics"]["std_g"])
    mlflow.log_metric("baseline_image_std_b", baseline_stats["image_statistics"]["std_b"])
    mlflow.log_metric(
        "baseline_class_0_count", baseline_stats["class_distribution"]["class_0_count"]
    )
    mlflow.log_metric(
        "baseline_class_1_count", baseline_stats["class_distribution"]["class_1_count"]
    )
    mlflow.log_metric(
        "baseline_confidence_mean", baseline_stats["confidence_statistics"]["mean"]
    )
    mlflow.log_metric("baseline_confidence_std", baseline_stats["confidence_statistics"]["std"])
    mlflow.log_artifact(config.DRIFT_BASELINE_STATS_PATH, "drift_detection")



    torch.save(
        {
            "model_state_dict": cnn_model.state_dict(),
            "architecture": SimpleCNN.__name__,
            "input_shape": (3, 64, 64),
            "output_shape": (1,),
            "threshold": optimal_threshold,
            "num_classes": config.NUM_CLASSES,
        },
        config.PYTORCH_MODEL_PATH,
    )
    print(f"CNN model saved to {config.PYTORCH_MODEL_PATH}")

    with open(config.MODEL_RESULTS_PATH, "w") as results_file:
        json.dump(test_results, results_file, indent=2)
    print(f"Model results saved to {config.MODEL_RESULTS_PATH}")

    mlflow.log_param("classification_threshold", optimal_threshold)

    # Log the model with the wrapper. The notebook forgot to pass the tuned threshold here.
    wrapper = SimpleLoggingWrapper(model=cnn_model, threshold=optimal_threshold)
    mlflow.pyfunc.log_model(config.MODEL_ARTIFACT_PATH, python_model=wrapper)
    print("Logged pyfunc model to the MLflow run")

    # Register the model and move it to Staging
    model_details = register_and_stage_model(client, run.info.run_id)

    mlflow.end_run()

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


def export():
    """Promote the best model if it beats Production, then export it to models/model."""
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    select_and_promote_best_model()
    export_model()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m vial_classifier.pipeline [data|train|export]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "data":
        data()
    elif command == "train":
        train()
    elif command == "export":
        export()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m vial_classifier.pipeline [data|train|export]")
        sys.exit(1)