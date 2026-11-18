# Drift baseline

#For drift detection we store the statistics of the model on the validation set as a baseline: image statistics per channel, class distribution, prediction distribution,
#confidence statistics, and the mean/std of the embeddings from the `fc1` layer. The serving endpoint compares live traffic against this baseline.


import torch
import numpy as np
import mlflow
from ..config.constants import threshold
from tuning import tune_threshold
import json
from ..config.paths import drift_baseline_stats_path
from preprocessing import split_dataset, create_dataloaders, load_image_dataset
from model_training import SimpleCNN, train_cnn_model
from ..config.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dataset = load_image_dataset()
train_dataset, val_dataset, test_dataset = split_dataset(image_dataset)
train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
cnn_model, training_history = train_cnn_model(
    train_loader, val_loader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device,
    random_seed=random_state,
)
optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_baseline_statistics(model, data_loader, device=device, feature_layer="fc1", threshold=threshold):
    """Compute baseline statistics for drift detection from a dataset."""
    print(f"\nComputing baseline statistics for drift detection...")
    print(f"Feature layer: {feature_layer}")

    model.eval()

    image_means = []
    image_stds = []
    class_counts = {0: 0, 1: 0}
    prediction_counts = {0: 0, 1: 0}
    confidences = []
    all_embeddings = []

    # Get the feature extraction layer
    if hasattr(model, feature_layer):
        feature_extractor = getattr(model, feature_layer)
    else:
        raise ValueError(f"Model does not have layer '{feature_layer}'")

    # Hook to capture embeddings
    embeddings = []
    def embedding_hook(module, input, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = feature_extractor.register_forward_hook(embedding_hook)

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                # Get predictions (logits -> probabilities)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= threshold).astype(int)

                # Compute image statistics (per channel)
                batch_mean = images.mean(dim=(0, 2, 3)).cpu().numpy()
                batch_std = images.std(dim=(0, 2, 3)).cpu().numpy()

                image_means.append(batch_mean)
                image_stds.append(batch_std)

                # Update class counts
                for label in labels.cpu().numpy().flatten():
                    class_counts[int(label)] += 1

                # Update prediction counts
                for pred in preds:
                    prediction_counts[int(pred)] += 1

                # Collect confidences
                confidences.extend(probs)

                # Get embeddings from the hook
                if embeddings:
                    all_embeddings.extend([e for e in embeddings])
                    embeddings.clear()

        # Compute final statistics
        image_means = np.array(image_means)
        image_stds = np.array(image_stds)

        baseline_stats = {
            'image_statistics': {
                'mean_r': float(np.mean(image_means[:, 0])),
                'mean_g': float(np.mean(image_means[:, 1])),
                'mean_b': float(np.mean(image_means[:, 2])),
                'std_r': float(np.mean(image_stds[:, 0])),
                'std_g': float(np.mean(image_stds[:, 1])),
                'std_b': float(np.mean(image_stds[:, 2])),
            },
            'class_distribution': {
                'class_0_count': class_counts[0],
                'class_1_count': class_counts[1],
                'total': class_counts[0] + class_counts[1],
            },
            'prediction_distribution': {
                'pred_0_count': prediction_counts[0],
                'pred_1_count': prediction_counts[1],
                'total': prediction_counts[0] + prediction_counts[1],
            },
            'confidence_statistics': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
            },
            'embedding_statistics': {
                'mean': [],
                'std': [],
            },
        }

        # Compute embedding statistics if embeddings were captured
        if all_embeddings:
            all_embeddings_array = np.concatenate(all_embeddings, axis=0)
            embedding_mean = np.mean(all_embeddings_array, axis=0)
            embedding_std = np.std(all_embeddings_array, axis=0)
            baseline_stats['embedding_statistics'] = {
                'mean': embedding_mean.tolist(),
                'std': embedding_std.tolist(),
                'shape': int(all_embeddings_array.shape[1]),
            }

        print(f"Baseline statistics computed:")
        print(f"  Class distribution: 0={class_counts[0]}, 1={class_counts[1]}")
        print(f"  Confidence mean: {baseline_stats['confidence_statistics']['mean']:.4f}")
        if all_embeddings:
            print(f"  Embedding shape: {baseline_stats['embedding_statistics']['shape']}")

        return baseline_stats

    finally:
        handle.remove()

baseline_stats = compute_baseline_statistics(cnn_model, val_loader, threshold=optimal_threshold)

with open(drift_baseline_stats_path, 'w') as f:
    json.dump(baseline_stats, f, indent=2)
print(f"Baseline statistics saved to {drift_baseline_stats_path}")

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
mlflow.log_artifact(drift_baseline_stats_path, "drift_detection")