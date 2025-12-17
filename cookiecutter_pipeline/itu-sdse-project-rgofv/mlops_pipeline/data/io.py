# Not sure if we need functions that just reads and writes, but here they are. 
# Added more complex read and writes!

from pathlib import Path
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def write_any(df: pd.DataFrame, path: Path):
    '''
    Writes a DataFrame to disk as CSV or Parquet.
    If extension missing/unsupported, defaults to .parquet.
    Returns the final path written.
    '''

    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".csv":
        df.to_csv(path, index=False)
        return path
    else:
        # default to parquet
        if ext not in {".parquet", ".pq"}:
            path = path.with_suffix(".parquet")    
        df.to_parquet(path, index=False)


# We need some reader function for our exported X and y csv's: 
def load_X_y(X_path: Path, y_path: Path):
    """
    Load features and single-column labels from CSV.
    Assumes index is in column 0; adjust index_col as needed.
    """
    X = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)

    if y_df.shape[1] != 1:
        raise ValueError(f"{y_path} must have exactly 1 column, got {y_df.shape[1]}")

    y = y_df.iloc[:, 0]
    # Ensure binary labels are ints 0/1 if applicable
    if set(y.dropna().unique()) <= {0, 1}:
        y = y.astype(int)

    return X, y