# src/visualization.py
from typing import List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

def histograms(df: pd.DataFrame, features: Optional[List[str]] = None, color: Optional[str] = None):
    features = features or [c for c in df.columns if c != "Outcome"]
    figs = []
    for col in features:
        figs.append(px.histogram(df, x=col, color=color, marginal="box", nbins=30, title=f"Distribution of {col}"))
    return figs

def skewness_table(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    features = features or [c for c in df.columns if c != "Outcome"]
    data = []
    for col in features:
        s = df[col].dropna()
        skew = s.skew()
        # Pearson’s second coefficient of skewness: 3(mean - median)/std
        symmetry = 3 * (s.mean() - s.median()) / (s.std() + 1e-9)
        data.append({"feature": col, "skewness": skew, "pearsons_symmetry": symmetry})
    return pd.DataFrame(data).sort_values("skewness", key=lambda x: x.abs(), ascending=False)

def correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None):
    features = features or [c for c in df.columns if c != "Outcome"]
    corr = df[features + (["Outcome"] if "Outcome" in df.columns else [])].corr()
    fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    return fig

def pairplot_sample(df: pd.DataFrame, sample: int = 500, color: Optional[str] = "Outcome"):
    d = df.sample(min(sample, len(df)), random_state=42) if len(df) > sample else df.copy()
    # Plotly doesn't have full seaborn-style pairplot; use scatter matrix
    fig = px.scatter_matrix(d, dimensions=[c for c in d.columns if c != "Outcome"], color=color)
    fig.update_traces(diagonal_visible=True)
    fig.update_layout(title="Scatter Matrix")
    return fig