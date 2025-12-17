from pprint import pprint
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
    classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.pipeline import Pipeline # Import Pipeline


def train_logreg_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, logreg_model_path:Path, random_state:int = 42, printing: bool =False):
    # Create a pipeline with imputation and Logistic Regression
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
        ('logistic_regression', LogisticRegression(random_state=random_state))
    ])

    # Define parameters for RandomizedSearchCV compatible with the pipeline and Logistic Regression
    params = {
        'logistic_regression__solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        'logistic_regression__penalty': ["l1", "l2", "elasticnet", None], # Use None for no penalty
        'logistic_regression__C': uniform(loc=0.01, scale=100), # Use uniform for C as it's a continuous parameter
    }

    # Filter out incompatible solver-penalty combinations for LogisticRegression
    # This ensures only valid combinations are used during randomized search
    compatible_params = []
    for solver in params['logistic_regression__solver']:
        for penalty in params['logistic_regression__penalty']:
            # 'liblinear' supports 'l1', 'l2', no elasticnet
            if solver == 'liblinear' and penalty in ['l1', 'l2', None]: 
                compatible_params.append({'logistic_regression__solver': [solver], 'logistic_regression__penalty': [penalty], 'logistic_regression__C': params['logistic_regression__C']})
            # These solvers support 'l2' and no penalty
            elif solver in ['newton-cg', 'lbfgs', 'sag'] and penalty in ['l2', None]: 
                compatible_params.append({'logistic_regression__solver': [solver], 'logistic_regression__penalty': [penalty], 'logistic_regression__C': params['logistic_regression__C']})
            # 'saga' is the most flexible
            elif solver == 'saga' and penalty in ['l1', 'l2', 'elasticnet', None]: 
                compatible_params.append({'logistic_regression__solver': [solver], 'logistic_regression__penalty': [penalty], 'logistic_regression__C': params['logistic_regression__C']})

    model_grid = RandomizedSearchCV(pipeline, param_distributions=compatible_params, verbose=3, n_iter=10, cv=3, random_state=random_state)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
        
    joblib.dump(value=best_model, filename=logreg_model_path)
    # The LogisticRegression model itself doesn't have .save_model like XGBoost
    # We will log the pickled model directly to MLflow in train.py

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

    return logreg_model_path, model_classification_report, best_model