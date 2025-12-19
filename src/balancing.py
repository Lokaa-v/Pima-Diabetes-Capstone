# src/balancing.py
from typing import Literal, Tuple, Optional
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

BalanceMethod = Literal["none", "undersample", "oversample", "smote"]

def balance(
    X: pd.DataFrame,
    y: pd.Series,
    method: BalanceMethod = "none",
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    if method == "none":
        return X, y
    if method == "undersample":
        rus = RandomUnderSampler(random_state=random_state)
        Xb, yb = rus.fit_resample(X, y)
        return Xb, yb
    if method == "oversample":
        ros = RandomOverSampler(random_state=random_state)
        Xb, yb = ros.fit_resample(X, y)
        return Xb, yb
    if method == "smote":
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        Xb, yb = sm.fit_resample(X, y)
        return Xb, yb
    raise ValueError(f"Unknown balancing method: {method}")