# src/unsupervised_models.py
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def kmeans_cluster(
    preprocessor: Pipeline,
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(steps=[("pre", preprocessor.named_steps["pre"])])
    Xtr = pipe.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xtr)
    centers = km.cluster_centers_
    return labels, centers

def pca_project(
    preprocessor: Pipeline,
    X: pd.DataFrame,
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(steps=[("pre", preprocessor.named_steps["pre"])])
    Xtr = pipe.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(Xtr)
    return Xp, pca.explained_variance_ratio_