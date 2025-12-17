import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient

client = MlflowClient()


def wait_for_deployment(model_name, model_version, stage='Staging'):
    status = False
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name,version=model_version)
            )
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    return status

model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
model_status = True
if model_version_details['current_stage'] != 'Staging':
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,stage="Staging", 
        archive_existing_versions=True
    )
    model_status = wait_for_deployment(model_name, model_version, 'Staging')
else:
    print('Model already in staging')



def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)


def to_staging(model_name, model_version):
    client = MlflowClient()
    model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
    model_status = True
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,stage="Staging", 
            archive_existing_versions=True
        )
        return wait_for_deployment(model_name, model_version, 'Staging')
    else:
        print('Model already in staging')
        return True
