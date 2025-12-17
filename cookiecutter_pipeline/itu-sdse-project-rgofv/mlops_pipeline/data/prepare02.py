from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

from .io import write_any


def create_dummy_cols(df, col):
    '''
    Create one-hot encoding columns in the data.

    Input:
        df = a pandas dataframe
        col = string specifying the column to work on

    Output:
        new_df = an updated pandas dataframe, with the column changed.

    '''
    df_dummies = pd.get_dummies(df[col], prefix=col)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df





def load_and_prepare_data(data_gold_path: Path, out_dir: Path, scaler_path: Path, test_size: float = 0.15, random_state: int = 42, printing: bool=False):
    data = pd.read_csv(data_gold_path)
    if printing:
        print(f"[prepare.py] Training data length: {len(data)}")
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
    X = data.drop(["lead_indicator", "source_signup", "bin_source_group1", "onboarding_False", "customer_group_1"], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, test_size=test_size, stratify=y
    )


    if printing:
        print(y_train)


    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    #non_numeric_cols = X.columns.difference(numeric_cols)

    # Let's create our MinMaxScaler here:
    scaler = MinMaxScaler()
    scaler.fit(X_train[numeric_cols])

    # We copy all data from the original X, then transform with the fitted scaler.
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])

    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])


    joblib.dump(value=scaler, filename=scaler_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write FOUR separate files
    X_train_path = write_any(X_train_scaled, out_dir / "X_train.csv")
    y_train_path = write_any(y_train.to_frame(name="lead_indicator"), out_dir / "y_train.csv")
    X_test_path  = write_any(X_test_scaled,  out_dir / "X_test.csv")
    y_test_path  = write_any(y_test.to_frame(name="lead_indicator"),  out_dir / "y_test.csv")

    if printing:
        print(f"[prepare] Saved:")
        print(f"  - {X_train_path}")
        print(f"  - {y_train_path}")
        print(f"  - {X_test_path}")
        print(f"  - {y_test_path}")
        print(f"[prepare] Shapes: X_train={X_train_scaled.shape}, y_train={y_train.shape}, "
              f"X_test={X_test_scaled.shape}, y_test={y_test.shape}")


    return {
        "X_train": X_train_path,
        "y_train": y_train_path,
        "X_test": X_test_path,
        "y_test": y_test_path,
    }




   



