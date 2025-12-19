# src/metrics.py
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff

def confusion_matrix_fig(y_true, y_pred, labels=(0, 1)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    z = cm.astype(int)
    x = [str(l) for l in labels]
    y = [str(l) for l in labels]
    fig = ff.create_annotated_heatmap(
        z=z, x=x, y=y, colorscale="Blues", showscale=True
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    return fig

def clf_report_table(y_true, y_pred) -> pd.DataFrame:
    rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(rpt).T.reset_index().rename(columns={"index": "metric"})