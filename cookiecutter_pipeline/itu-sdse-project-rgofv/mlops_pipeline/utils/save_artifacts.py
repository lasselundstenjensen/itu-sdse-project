import json
from pprint import pprint
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

from mlops_pipeline.config import MODELS_DIR

def save_artifacts(X_train:pd.DataFrame, xgboost_classification_path:Path, lr_classification_path:Path, column_list_out:Path, model_results_path:Path, printing=False):
    
    make_artifacts_path = MODELS_DIR / 'artifacts'
    make_artifacts_path.mkdir(parents=True, exist_ok=True)


    with open(xgboost_classification_path, "r", encoding="utf-8") as f:
        xgb_metrics = json.load(f)

    with open(lr_classification_path, "r", encoding="utf-8") as f:
        lr_metrics = json.load(f)


    model_results = {
        "xg_boost": xgb_metrics,
        "log_reg": lr_metrics,
    }

    with open(column_list_out, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        if printing:
            pprint(columns)
        json.dump(columns, columns_file)

    if printing:
        print('Saved column list to ', column_list_out)

    #model_results_path = "./artifacts/model_results.json"
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file)