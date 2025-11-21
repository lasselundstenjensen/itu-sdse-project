
import pandas as pd
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint

from mlops_project.config import ARTIFACTS_DIR

def split_data(data: pd.DataFrame):
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y)
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
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
    return model_grid


def train_logistic_regression(X_train, y_train, experiment_name):
    
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()
        lr_model_path = "./artifacts/lead_model_lr.pkl"

        params = {
                'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                'penalty':  [None, "l1", "l2", "elasticnet"],
                'C' : [100, 10, 1.0, 0.1, 0.01]
        }
        model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_
        return model_grid
    

def train_models(data, experiment_name):
    """Main training pipeline"""
    X_train, X_test, y_train, y_test = split_data(data)
    xgb_model = train_xgboost(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train, experiment_name)
    return xgb_model, lr_model, X_test, y_test