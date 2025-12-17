# General utility script to handle mlflow runs.

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Dict, Any
import time

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus



# Our script for handling all things MLflow


def set_experiment(experiment_name: str) -> None:
    '''
    Simple function for setting an MLflow experiment by name.

    Input:
        experiment_name     - Str input to name in mlflow
    '''
    mlflow.set_experiment(experiment_name)

@contextmanager
def start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    '''
    Context manager that starts an MLflow run and optionally handles any tags. 
    A context manager uses a Generator (that's the yield) and are used specifically with "with" calls
    Basically, when calling the start_run, the contextmanager will advance it until its first yield. (note that mlflow.start_run is mlflow's very own context manager)
    Conceptually, it's like a class! Read about it here: https://stackoverflow.com/questions/36559580/what-is-the-purpose-of-a-context-manager-in-python
    '''
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run

def log_params(params: Dict[str, Any]) -> None:
    '''
    Simple call to mlflow's log_params. We only log parameters that are not None.
    '''
    mlflow.log_params({k: v for k, v in params.items() if v is not None})

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    '''
    Simple call to mlflows log_metrics, here we should just feed it the Dict and the step we are at.
    '''
    mlflow.log_metrics(metrics, step=step)

def register_model(artifact_path: str, model_name: str) -> str:
    '''
    Returns model version/run_id
    '''
    result = mlflow.register_model(model_uri=f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
                                   name=model_name)
    return result.version

def transition_stage(model_name: str, model_version: int | str, stage: str, archive_existing: bool = True) -> None:
    '''
    Function for handling model version stages "Staging", "Production", "Archived", "None"
    '''
    client = MlflowClient()
    client.transition_model_version_stage(name=model_name, model_version=int(model_version), stage = stage, archive_existing_versions = archive_existing)


def wait_until_ready(model_name: str, model_version: int | str, retries: int = 10, sleep_seconds: float = 1.0, raise_on_fail: bool = True) -> bool:
    '''
    Expanded function from 02_model.py that polls MLflow until the model is READY or we hit a timeout.

    Input:
        model_name      - The str name of the model
        model_version   - The numbered version of the model. If str, converts to int
        retries         - Number of loops to try and see if the model is READY
        sleep_seconds   - The amount of time we are sleeping between retries.
        raise_on_fail   - If set to true, we raise rather than return False upon timeout or failue. You would want a raise rather than a False return, 
                        if we need the model to be ready to continue on to the next steps. If readiness is optional, or we have fallbacks, return False and handle it. 
    Output:
        bool            - True if ready

    '''
    client = MlflowClient()
    model_version = int(model_version)
    last_status = None

    # We loop for number of specified retries
    for _ in range(retries):
        try:
            model_version_details = client.get_model_version(name=model_name, version=model_version)
        except MlflowException as e:
            if raise_on_fail:
                raise
            return False
        

        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            return True
        
    ## THESE ARE SPECIAL CASES:
    ## Some helpful hints on how to deal with MLflow added here
    ## 
    #Some servers use explicit failure statuses; handle defensively
        if str(status) in ("ModelVersionStatus.FAILED", "ModelVersionStatus.FAILED_REGISTRATION"):
            if raise_on_fail:
                raise RuntimeError(
                    f"Registration failed for {model_name}:{model_version} (status={status})."
                )
            return False

        last_status = status
        time.sleep(sleep_seconds)

    # timeout
    if raise_on_fail:
        raise TimeoutError(
            f"Timed out waiting for {model_name}:{model_version} to be READY "
            f"(last status={ModelVersionStatus.to_string(last_status) if last_status else 'unknown'})."
        )
    return False