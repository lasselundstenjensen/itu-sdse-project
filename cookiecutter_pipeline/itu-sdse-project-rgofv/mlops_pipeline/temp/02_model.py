import pandas as pd
import os
import datetime
import json
import warnings
import numpy as np
from pprint import pprint

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    f1_score
)
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import time
import shutil

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

def create_dummy_cols(df, col):
    '''
    Create one-hot encoding columns in the data.

    Input:
        df = a pandas dataframe
        col = string specifying the column to work on

    Output:
        new_df = an updated pandas dataframe, with the column changed.

    '''
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


def load_and_prepare_data(data_gold_path, printing=False):
    data = pd.read_csv(data_gold_path)
    if printing:
        print(f"Training data length: {len(data)}")
        print(data.head(5))

    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]

    other_vars = data.drop(cat_cols, axis=1)

    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")
        if printing:
            print(f"Changed column {col} to float")

    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )

    if printing:
        print(y_train)

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, X_test, y_train, y_test, printing=False):
    model = XGBRFClassifier(random_state=42)
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }

    model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)

    model_grid.fit(X_train, y_train)

    best_model_xgboost_params = model_grid.best_params_
    if printing:
        print("Best xgboost params")
        pprint(best_model_xgboost_params)

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    if printing:
        print("Accuracy train", accuracy_score(y_pred_train, y_train))
        print("Accuracy test", accuracy_score(y_pred_test, y_test))

        conf_matrix = confusion_matrix(y_test, y_pred_test)
        print("Test actual/predicted\n")
        print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
        print("Classification report\n")
        print(classification_report(y_test, y_pred_test), '\n')

        conf_matrix = confusion_matrix(y_train, y_pred_train)
        print("Train actual/predicted\n")
        print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
        print("Classification report\n")
        print(classification_report(y_train, y_pred_train), '\n')

    xgboost_model = model_grid.best_estimator_
    xgboost_model_path = "./artifacts/lead_model_xgboost.json"
    xgboost_model.save_model(xgboost_model_path)

    model_results = {
        xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
    }

    return xgboost_model_path, model_results, xgboost_model


class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def train_logistic_regression_model(X_train, X_test, y_train, y_test, experiment_name, data_version, printing=False):
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()
        lr_model_path = "./artifacts/lead_model_lr.pkl"

        params = {
            'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            'penalty': ["none", "l1", "l2", "elasticnet"],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }
        model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_

        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", data_version)
        
        joblib.dump(value=best_model, filename=lr_model_path)
            
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model))

    model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

    best_model_lr_params = model_grid.best_params_

    if printing:
        print("Best lr params")
        pprint(best_model_lr_params)

        print("Accuracy train:", accuracy_score(y_pred_train, y_train))
        print("Accuracy test:", accuracy_score(y_pred_test, y_test))

        conf_matrix = confusion_matrix(y_test, y_pred_test)
        print("Test actual/predicted\n")
        print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
        print("Classification report\n")
        print(classification_report(y_test, y_pred_test), '\n')

        conf_matrix = confusion_matrix(y_train, y_pred_train)
        print("Train actual/predicted\n")
        print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
        print("Classification report\n")
        print(classification_report(y_train, y_pred_train), '\n')

        print(model_classification_report["weighted avg"]["f1-score"])

    return lr_model_path, model_classification_report, best_model


def save_artifacts(X_train, model_results, printing=False):
    column_list_path = './artifacts/columns_list.json'
    with open(column_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        if printing:
            pprint(columns)
        json.dump(columns, columns_file)

    if printing:
        print('Saved column list to ', column_list_path)

    model_results_path = "./artifacts/model_results.json"
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file)

def get_results(experiment_name, printing=False):
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    if printing:
        print(experiment_ids)
    
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    if printing:
        print(experiment_best)

    with open("./artifacts/model_results.json", "r") as f:
        model_results = json.load(f)
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    print(f"Model outputs: {results_df} \nBest model: {best_model}")
    return best_model, experiment_best

def save_best_model(best_model_path, xgboost_model, lr_model, printing=False):
    """
    Save the best performing model as model.pkl
    
    Args:
        best_model_path: Path to the best model (either .json or .pkl)
        xgboost_model: Trained XGBoost model object
        lr_model: Trained Logistic Regression model object
        printing: Whether to print status messages
    """
    output_path = "./artifacts/model.pkl"
    
    if "xgboost" in best_model_path:
        # Save XGBoost model as pickle
        joblib.dump(xgboost_model, output_path)
        if printing:
            print(f"Saved best model (XGBoost) to {output_path}")
    else:
        # Copy LR model (already in pickle format)
        shutil.copy(best_model_path, output_path)
        if printing:
            print(f"Saved best model (Logistic Regression) to {output_path}")
    
    return output_path

def get_model_version(model_name, experiment_best, artifact_path, client, printing=False):
    prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']
    prod_model_exists = len(prod_model)>0

    if prod_model_exists:
        prod_model_version = dict(prod_model[0])['version']
        prod_model_run_id = dict(prod_model[0])['run_id']

        print('Production model name: ', model_name)
        print('Production model version:', prod_model_version)
        print('Production model run id:', prod_model_run_id)
    else:
        print('No model in production')

    train_model_score = experiment_best["metrics.f1_score"]
    model_details = {}
    model_status = {}
    run_id = None

    if prod_model_exists:
        data, details = mlflow.get_run(prod_model_run_id)
        prod_model_score = data[1]["metrics.f1_score"]

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

        if train_model_score>prod_model_score:
            print("Registering new model")
            run_id = experiment_best["run_id"]
    else:
        print("No model in production")
        run_id = experiment_best["run_id"]

    print(f"Registered model: {run_id}")

    if run_id is not None:
        print(f'Best model found: {run_id}')

        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id,
            artifact_path=artifact_path
        )
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        wait_until_ready(model_details.name, model_details.version)
        model_details = dict(model_details)
        print(model_details)

    return run_id

def wait_for_deployment(model_name, model_version, client, stage='Staging'):
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

def set_model_to_staging(model_name, model_version, client):
    model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
    model_status = True
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,stage="Staging", 
            archive_existing_versions=True
        )
        model_status = wait_for_deployment(model_name, model_version, client, 'Staging')
    else:
        print('Model already in staging')
    return model_status


def main():
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    data_gold_path = "./artifacts/train_data_gold.csv"
    data_version = "00000"
    experiment_name = current_date
    artifact_path = "model"
    model_name = "lead_model"
    client = MlflowClient()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)

    mlflow.set_experiment(experiment_name)

    warnings.filterwarnings('ignore')
    pd.set_option('display.float_format', lambda x: "%.3f" % x)

    X_train, X_test, y_train, y_test = load_and_prepare_data(data_gold_path, printing=True)

    xgboost_model_path, xgb_report, xgboost_model = train_xgboost_model(X_train, X_test, y_train, y_test, printing=True)

    lr_model_path, lr_report, lr_model = train_logistic_regression_model(
        X_train, X_test, y_train, y_test, experiment_name, data_version, printing=True
    )
    
    model_results = {
        xgboost_model_path: list(xgb_report.values())[0],
        lr_model_path: lr_report
    }

    save_artifacts(X_train, model_results, printing=True)

    print("Model training completed.")

    best_model, experiment_best = get_results(experiment_name, printing=True)

    # Save the best performing model as model.pkl which is needed for the the model validator 
    best_model_pkl_path = save_best_model(best_model, xgboost_model, lr_model, printing=True)

    runid = get_model_version(model_name, experiment_best, artifact_path, client, printing=True)

    model_version = 1

    model_status = set_model_to_staging(model_name, model_version, client)

if __name__ == "__main__":
    main()