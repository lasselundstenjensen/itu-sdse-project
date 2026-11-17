
import numpy as np
import torch

from vial_classifier.config import THRESHOLD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = THRESHOLD


def count_classes(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def print_section_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


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


def compute_baseline_statistics(
    model,
    data_loader,
    device=device,
    feature_layer="fc1",
    threshold=threshold,
):
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
            for images, labels in data_loader:
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
