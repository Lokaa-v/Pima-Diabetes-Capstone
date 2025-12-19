# src/imputation.py
from typing import List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

ZERO_NA_FEATURES_DEFAULT = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

ImputeStrategy = Literal["mean", "median", "knn"]

def replace_zeros_with_nan(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    df = df.copy()
    features = features or ZERO_NA_FEATURES_DEFAULT
    for col in features:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
    return df

def impute(
    df: pd.DataFrame,
    strategy: ImputeStrategy = "median",
    knn_neighbors: int = 5,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    df = df.copy()
    features = features or [c for c in df.columns if c != "Outcome"]

    if strategy in ("mean", "median"):
        for col in features:
            if col not in df.columns:
                continue
            if strategy == "mean":
                value = df[col].mean()
            else:
                value = df[col].median()
            df[col] = df[col].fillna(value)
        return df

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        # Keep the column order stable
        cols = df.columns.tolist()
        df[features] = imputer.fit_transform(df[features])
        df = df[cols]
        return df

    raise ValueError(f"Unknown strategy: {strategy}")