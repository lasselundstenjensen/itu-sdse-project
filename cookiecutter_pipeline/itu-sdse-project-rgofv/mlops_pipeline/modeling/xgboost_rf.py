from pprint import pprint
from pathlib import Path
import joblib
from xgboost import XGBRFClassifier
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
    classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def train_xgboost_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, xgboost_model_path:Path, xgboost_json_path:Path, random_state:int = 42, printing: bool =False):
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
    # Here we save the Python object, including any scikit‑learn CV metadata
    joblib.dump(value=xgboost_model, filename=xgboost_model_path)
    # Here we save the native XGBoost model (Booster) in a portable format
    xgboost_model.save_model(xgboost_json_path)

    model_results = {
        str(xgboost_model_path): classification_report(y_train, y_pred_train, output_dict=True)
    }

    return xgboost_model_path, model_results, xgboost_model