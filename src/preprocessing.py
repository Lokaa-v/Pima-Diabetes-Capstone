# src/preprocessing.py
from typing import Tuple, Literal, Optional, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ScalerName = Literal["standard", "minmax", "none"]
TransformName = Literal["none", "yeo-johnson", "box-cox", "log1p"]

def build_transformer(
    numeric_features: List[str],
    scaler: ScalerName = "standard",
    transform: TransformName = "none"
) -> ColumnTransformer:
    steps = []

    # Transform
    if transform == "yeo-johnson":
        steps.append(("power", PowerTransformer(method="yeo-johnson")))
    elif transform == "box-cox":
        # Box-Cox requires strictly positive data. Ensure by shifting if needed.
        # Prefer YJ in general; Box-Cox left as option for experimentation.
        steps.append(("power", PowerTransformer(method="box-cox", standardize=False)))
    elif transform == "log1p":
        steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))

    # Scale
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler == "none":
        pass

    num_pipeline = Pipeline(steps=steps) if steps else "passthrough"

    pre = ColumnTransformer(
        transformers=[("num", num_pipeline, numeric_features)],
        remainder="drop"
    )
    return pre

def split_xy(
    df: pd.DataFrame,
    target_col: str = "Outcome"
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

def build_preprocess_pipeline(
    df: pd.DataFrame,
    scaler: ScalerName = "standard",
    transform: TransformName = "none",
    target_col: str = "Outcome"
) -> Tuple[Pipeline, List[str]]:
    numeric_features = [c for c in df.columns if c != target_col]
    pre = build_transformer(numeric_features, scaler=scaler, transform=transform)
    pipe = Pipeline(steps=[("pre", pre)])
    return pipe, numeric_features