# src/supervised_models.py
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def get_model(name: str, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    name = name.lower()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, **params)
    if name == "decision_tree":
        return DecisionTreeClassifier(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    if name == "svm":
        return SVC(probability=True, **params)
    if name == "knn":
        return KNeighborsClassifier(**params)
    raise ValueError(f"Unknown model: {name}")

def train_and_eval(
    model_name: str,
    preprocessor: Pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[Pipeline, Dict[str, float]]:
    model = get_model(model_name, params)
    pipe = Pipeline(steps=[("pre", preprocessor.named_steps["pre"]), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_val, y_proba)
        except Exception:
            metrics["roc_auc"] = float("nan")
    return pipe, metrics