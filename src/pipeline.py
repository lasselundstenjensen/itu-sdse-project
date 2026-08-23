import sys
import torch
import mlflow
from torch.utils.data import DataLoader

from .data.fetch import fetch_and_prepare_data
from .data.preprocess import preprocess_data
from .data.features import create_features
from .models.train import train_models
from .models.registry import register_models
from .deployment.deploy import deploy_model, transition_to_production, set_model_alias
from .utils import print_section_header
from .config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME


def get_production_model():
    """
    Fetch the current Production model from MLflow registry.
    Returns (model_uri, version) or (None, None) if no Production model exists.
    """
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    for mv in model_versions:
        mv_dict = dict(mv)
        if mv_dict.get("current_stage") == "Production":
            version = mv_dict["version"]
            model_uri = f"models:/{MODEL_NAME}/{version}"
            return model_uri, version
    
    return None, None


def load_test_dataset_from_mlflow():
    """
    Load the test dataset from MLflow artifacts.
    Returns a DataLoader for the test dataset.
    """
    print_section_header("LOADING TEST DATASET FROM MLFLOW")
    
    # Fetch the latest MLflow run with the test dataset
    runs = mlflow.search_runs(
        filter_string="tags.mlflow.runName = 'full_pipeline'",
        max_results=1,
        order_by=["start_time DESC"]
    )
    
    if runs.empty:
        print("ERROR: No MLflow runs found with the test dataset.")
        print("Please run the training pipeline first: python -m src.pipeline")
        return None
    
    latest_run_id = runs.iloc[0].run_id
    print(f"Found latest pipeline run: {latest_run_id}")
    
    # Download the test dataset artifact
    test_dataset_path = mlflow.artifacts.download_artifacts(
        run_id=latest_run_id,
        artifact_path="data/test_dataset.pt"
    )
    print(f"Downloaded test dataset: {test_dataset_path}")
    
    # Load the dataset with allowed globals (to handle custom classes)
    test_dataset = torch.load(test_dataset_path, weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Recreated test DataLoader with {len(test_dataset)} samples")
    
    return test_loader


def evaluate_model_f1(model, test_loader):
    """
    Evaluate a model on the test dataset and return its F1-score.
    """
    from sklearn.metrics import f1_score
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # Binary classification
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    f1 = f1_score(all_labels, all_preds, average="binary")
    return f1


def compare_with_production(new_model_f1, production_model_uri, test_loader):
    """
    Compare new model's F1-score against current Production model.
    Returns True if new model is better or equal, False otherwise.
    """
    print_section_header("COMPARING WITH PRODUCTION MODEL")
    
    if production_model_uri is None:
        print("No current Production model found. New model will be promoted by default.")
        return True
    
    # Load and evaluate production model
    print(f"Loading Production model: {production_model_uri}")
    production_model = mlflow.pytorch.load_model(production_model_uri)
    production_f1 = evaluate_model_f1(production_model, test_loader)
    
    print(f"Current Production F1-score: {production_f1:.4f}")
    print(f"New model F1-score: {new_model_f1:.4f}")
    
    if new_model_f1 >= production_f1:
        print("New model is better or equal. Promoting to Production.")
        return True
    else:
        print("New model is worse. Keeping current Production model.")
        return False


def run_data_pipeline(data=None):
    print_section_header("IMAGE DATA PIPELINE")

    data = fetch_and_prepare_data()

    train_loader, val_loader, test_loader = preprocess_data(data)
    create_features()

    return train_loader, val_loader, test_loader


def run_model_pipeline(train_loader=None, val_loader=None, test_loader=None):
    print_section_header("CNN MODEL PIPELINE")

    models_result = train_models(train_loader, val_loader, test_loader)
    registry_result = register_models()

    if registry_result.get("model_details"):
        model_version = registry_result["model_details"].get("version", 1)
        
        # Always deploy to Staging
        deploy_result = deploy_model(
            model_version=model_version,
            transition_to_prod=False
        )
        registry_result["deploy_success"] = deploy_result

        if deploy_result:
            set_model_alias(model_version=model_version, alias="best")
            registry_result["model_labelled"] = True
            
            # Get the new model's F1-score from training results
            new_model_f1 = models_result.get("test_results", {}).get("f1_score", 0)
            
            # Get current Production model
            production_model_uri, _ = get_production_model()
            
            # Compare with Production (if exists) and promote if better
            if compare_with_production(new_model_f1, production_model_uri, test_loader):
                prod_result = transition_to_production(model_version=model_version)
                registry_result["production_deploy_success"] = prod_result
            else:
                registry_result["production_deploy_success"] = False

    return {
        "models": models_result,
        "registry": registry_result,
    }


def run_full_pipeline():
    print_section_header("FULL IMAGE CLASSIFICATION PIPELINE")
    print("Starting complete pipeline execution...\n")

    # Initialize MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment = mlflow.create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full_pipeline") as run:
        train_loader, val_loader, test_loader = run_data_pipeline()

        # Log the test dataset as an MLflow artifact
        print_section_header("LOGGING TEST DATASET TO MLFLOW")
        test_dataset = test_loader.dataset
        test_dataset_path = "test_dataset.pt"
        torch.save(test_dataset, test_dataset_path)
        mlflow.log_artifact(test_dataset_path, artifact_path="data")
        print(f"Logged test dataset to MLflow: {test_dataset_path}")

        model_results = run_model_pipeline(
            train_loader, val_loader, test_loader
        )

        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        test_samples = len(test_loader.dataset)

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
        print("\nSummary:")
        print(f"  Training samples: {train_samples}")
        print(f"  Validation samples: {val_samples}")
        print(f"  Test samples: {test_samples}")
        print(f"  CNN model trained: {model_results.get('models', {}).get('cnn_model') is not None}")
        print(f"  Model registered: {model_results.get('registry', {}).get('model_details') is not None}")
        print(f"  Deployment successful: {model_results.get('registry', {}).get('deploy_success', False)}")
        print(f"  Model labelled as best: {model_results.get('registry', {}).get('model_labelled', False)}")
        print(f"  Production deployment: {model_results.get('registry', {}).get('production_deploy_success', False)}")
        if model_results.get('models', {}).get('optimal_threshold'):
            print(f"  Optimal threshold: {model_results['models']['optimal_threshold']:.4f}")
        if model_results.get('models', {}).get('test_results'):
            test_f1 = model_results['models']['test_results'].get('f1_score', 0)
            print(f"  Test F1-score: {test_f1:.4f}")

        return {
            "data": {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
            },
            "models": model_results,
        }


if __name__ == "__main__":
    print("Running MLOps pipeline...")
    print("=" * 50)

    result = run_full_pipeline()
    print("\nPipeline executed successfully!")
