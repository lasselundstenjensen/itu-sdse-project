import pandas as pd

def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


def build_features(data: pd.DataFrame):
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    print("Columns in build_features:", data.columns.tolist())
    cat_cols = ["customer_group", "onboarding", "source"]
    cat_vars = data[cat_cols].copy()

    other_vars = data.drop(cat_cols, axis=1)
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")

    return data
