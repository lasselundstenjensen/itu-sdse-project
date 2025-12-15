import datetime
from pathlib import Path

import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from xgboost import XGBRFClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAINING_GOLD_DATA_FILE = ARTIFACTS_DIR / "training_data_gold.csv"
XGBOOST_MODEL_FILE = ARTIFACTS_DIR / "xgboost_model.json"
LOGISTIC_REGRESSION_MODEL_FILE = ARTIFACTS_DIR / "logistic_regression_model.pkl"
X_TEST_FILE = ARTIFACTS_DIR / "X_test.csv"
Y_TEST_FILE = ARTIFACTS_DIR / "y_test.csv"


def encode_categoricals(df):
    df = df.copy()

    df = df.drop(["lead_id", "customer_code", "date_part"], axis=1)
    categorical = ["customer_group", "onboarding", "bin_source", "source"]

    for col in categorical:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df.astype("float64")


class LogisticRegressionAdapter(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_model):
        self._model = trained_model

    def predict(self, context, model_input):
        return self._model.predict_proba(model_input)[:, 1]




def train_single_model(
    model_cls,
    params,
    X_train,
    y_train,
    model_path: Path,
    autolog_module: str,
):
    getattr(mlflow, autolog_module).autolog(log_models=False)

    model = model_cls(random_state=42)

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=10,
        cv=3,
        verbose=2,
        random_state=42,
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_

    joblib.dump(best, model_path)
    mlflow.log_artifact(model_path)

    if isinstance(best, LogisticRegression):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LogisticRegressionAdapter(best),
        )


def main():
    data = pd.read_csv(TRAINING_GOLD_DATA_FILE)
    data = encode_categoricals(data)

    y = data["lead_indicator"]
    X = data.drop(columns=["lead_indicator"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    X_test.to_csv(X_TEST_FILE, index=False)
    y_test.to_csv(Y_TEST_FILE, index=False)

    experiment_name = datetime.datetime.now().strftime("%Y_%m_%d")
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    param_xgb = {
        "learning_rate": uniform(0.01, 0.3),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
    }

    param_lr = {
        "solver": ["lbfgs", "liblinear"],
        "C": [0.1, 1, 10],
    }

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="xgb"):
        train_single_model(
            XGBRFClassifier,
            param_xgb,
            X_train,
            y_train,
            XGBOOST_MODEL_FILE,
            "xgboost",
        )

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="lr"):
        train_single_model(
            LogisticRegression,
            param_lr,
            X_train,
            y_train,
            LOGISTIC_REGRESSION_MODEL_FILE,
            "sklearn",
        )


if __name__ == "__main__":
    main()
