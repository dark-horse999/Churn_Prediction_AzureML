# data_utils.py (with NaN handling improvements)
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV from a path. `path` can be:
     - local path (./data/train.csv)
     - path provided by AzureML job (mounted local path).
    """
    p = Path(path)
    # If it's a folder containing a single CSV, try to find it
    if p.is_dir():
        csvs = list(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in folder: {path}")
        print(f"[INFO] Loaded CSV file: {csvs[0]}")
        return pd.read_csv(csvs[0])
    # else assume file path
    print(f"[INFO] Loaded CSV file: {path}")
    return pd.read_csv(path)

def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and labels (y).
    - Drops 'CustomerID' if present
    - Drops rows where 'Churn' is missing
    """
    df = df.copy()

    if 'Churn' not in df.columns:
        raise KeyError("Target column 'Churn' not found in dataset!")

    # Drop rows with missing target
    missing_before = df['Churn'].isna().sum()
    if missing_before > 0:
        print(f"[WARNING] Found {missing_before} missing values in target 'Churn'. Dropping these rows.")
        df = df.dropna(subset=['Churn'])

    # Drop CustomerID if present
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Separate target and features
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    """
    Train/validation split with stratification.
    Ensures target has no NaNs before splitting.
    """
    if y.isna().any():
        raise ValueError("Target 'y' still contains NaN after preprocessing. Please check data.")
    
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )