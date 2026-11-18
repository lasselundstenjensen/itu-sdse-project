from training_helpers import print_section_header
import torch
import numpy as np
import mlflow
from model_training import SimpleCNN, train_cnn_model
from ..config.constants import *
from preprocessing import split_dataset, create_dataloaders, load_image_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_section_header("CNN MODEL TRAINING")

torch.manual_seed(random_state)
np.random.seed(random_state)

if mlflow.active_run():
    mlflow.end_run()
    print("Ended previous active MLflow run")

run = mlflow.start_run(run_name="training")

# Log parameters to MLflow
mlflow.log_param("model_architecture", "SimpleCNN")
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("num_epochs", num_epochs)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("initial_threshold", threshold)
mlflow.log_param("num_classes", num_classes)
mlflow.log_param("image_size", "64x64")
mlflow.log_param("random_seed", random_state)

# Log model architecture details
model_params = sum(p.numel() for p in SimpleCNN().parameters())
trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
mlflow.log_param("total_parameters", model_params)
mlflow.log_param("trainable_parameters", trainable_params)


if __name__ == "__main__":
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