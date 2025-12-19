"""Streamlit app for the Pima Diabetes Capstone project (expanded modules).

Added interactive modules:
1. Introduction & Dataset Description
2. Module 1: Data Cleaning & Imputation
3. Module 2: Symmetry & Skewness Analysis
4. Module 3: Scaling & Transformations
5. Module 4: Univariate Visualization
6. Module 5: Dataset Balancing & Entropy
7. Module 6: Supervised Learning (compare models)
8. Module 7: Unsupervised Learning (silhouette + PCA)
9. Module 8: Performance Metrics (ROC/PR/Confusion)
10. Conclusion & exports
"""

from __future__ import annotations
import sys
from pathlib import Path
import typing as t
import io
import pickle

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, silhouette_score
from scipy.stats import entropy as scipy_entropy

# Ensure project root is importable so we can use src/*.py modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import preprocessing, imputation, balancing, supervised_models, unsupervised_models, visualization, metrics

DATA_PATH = ROOT / "data" / "pima_diabetes.csv"

st.set_page_config(page_title="Pima Diabetes Explorer", layout="wide")

@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data
def prepare_preprocessor(df: pd.DataFrame, scaler: str, transform: str):
    pipe, features = preprocessing.build_preprocess_pipeline(df, scaler=scaler, transform=transform)
    return pipe, features


def compute_entropy_from_counts(counts: pd.Series) -> float:
    probs = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


def transform_dataframe(pre, df, features):
    # pre is a Pipeline with named_steps['pre'] a ColumnTransformer
    transformer = pre.named_steps["pre"]
    arr = transformer.fit_transform(df[features])
    # If result is 1D for single feature, make 2D
    arr = np.atleast_2d(arr)
    cols = features[:arr.shape[1]] if arr.shape[1] == len(features) else [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def compare_models(df, settings, models_to_compare):
    results = []
    for m in models_to_compare:
        X_val, preds_out, model_out = run_supervised(df, **settings, model_name=m)
        metrics_dict = model_out["metrics"].copy()
        metrics_dict.update({"model": m})
        results.append({**metrics_dict, **{"model": m}})
    return pd.DataFrame(results)


def run_supervised(
    df: pd.DataFrame,
    impute_strategy: str,
    scaler: str,
    transform: str,
    balance_method: str,
    model_name: str,
    test_size: float,
    random_state: int
) -> tuple[pd.DataFrame, dict, t.Any]:
    # 1. clean zeros
    df2 = imputation.replace_zeros_with_nan(df)

    # 2. impute
    df2 = imputation.impute(df2, strategy=impute_strategy)

    # 3. split
    X, y = preprocessing.split_xy(df2)
    X_train, X_val, y_train, y_val = preprocessing.train_val_split(X, y, test_size=test_size, random_state=random_state)

    # 4. preprocess pipeline
    pre, features = prepare_preprocessor(df2, scaler=scaler, transform=transform)

    # 5. balance
    X_train_b, y_train_b = balancing.balance(X_train, y_train, method=balance_method, random_state=random_state)

    # 6. train and eval
    model_pipe, model_metrics = supervised_models.train_and_eval(model_name, pre, X_train_b, y_train_b, X_val, y_val)

    # predictions
    y_pred = model_pipe.predict(X_val)
    try:
        y_proba = model_pipe.predict_proba(X_val)[:, 1]
    except Exception:
        y_proba = None

    return X_val, {"y_val": y_val, "y_pred": y_pred, "y_proba": y_proba}, {"pipe": model_pipe, "metrics": model_metrics}


def single_sample_prediction(model_pipe, sample: dict[str, float], features: list[str]):
    df = pd.DataFrame([sample])[features]
    pred = model_pipe.predict(df)[0]
    try:
        proba = model_pipe.predict_proba(df)[0, 1]
    except Exception:
        proba = None
    return pred, proba


def main():
    st.title("🩺 Pima Diabetes — Module-driven Explorer")

    df = load_data()

    # Sidebar controls
    st.sidebar.header("Data & Preprocessing")
    st.sidebar.write("Load & preprocessing options")

    impute_strategy = st.sidebar.selectbox("Imputation Strategy", options=["median", "mean", "knn"], index=0)
    scaler = st.sidebar.selectbox("Scaler", options=["standard", "minmax", "none"], index=0)
    transform = st.sidebar.selectbox("Transform", options=["none", "yeo-johnson", "box-cox", "log1p"], index=0)
    balance_method = st.sidebar.selectbox("Balance Method", options=["none", "undersample", "oversample", "smote"], index=0)
    model_name = st.sidebar.selectbox("Default Model", options=["logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "svm", "knn"], index=0)
    test_size = st.sidebar.slider("Validation size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random seed", value=42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.header("Unsupervised")
    k_clusters = st.sidebar.slider("KMeans clusters", 2, 8, 3)
    pca_components = st.sidebar.slider("PCA components", 2, 4, 2)

    # Top-level module tabs
    tabs = st.tabs([
        "Introduction",
        "Module 1: Cleaning & Imputation",
        "Module 2: Skewness",
        "Module 3: Scaling/Transforms",
        "Module 4: Univariate Viz",
        "Module 5: Balancing & Entropy",
        "Module 6: Supervised",
        "Module 7: Unsupervised",
        "Module 8: Metrics",
        "Conclusion"
    ])

    # --- Introduction ---
    with tabs[0]:
        st.header("Introduction & Dataset")
        st.markdown("""
        **Dataset:** Pima Indians Diabetes Dataset
        - Target: `Outcome` (0 = no diabetes, 1 = diabetes)
        - Rows: {}
        - Features: {}
        """.format(len(df), ', '.join([c for c in df.columns if c != 'Outcome'])))
        st.write(df.head())
        st.info("Use the left sidebar to select preprocessing, balancing and modeling options.")

    # --- Module 1 ---
    with tabs[1]:
        st.header("Module 1 — Data Cleaning & Imputation")
        st.subheader("Missing values & zeros-as-missing")
        st.write("Zeros replaced by NaN for: Glucose, BloodPressure, SkinThickness, Insulin, BMI")
        zero_cols = imputation.ZERO_NA_FEATURES_DEFAULT
        zero_counts = {c: int((df[c] == 0).sum()) for c in zero_cols if c in df.columns}
        st.json(zero_counts)

        if st.button("Show missing before/after imputation"):
            df0 = df.copy()
            before = df0.isna().sum()
            df1 = imputation.replace_zeros_with_nan(df0)
            df2 = imputation.impute(df1, strategy=impute_strategy)
            after = df2.isna().sum()
            st.markdown("**Missing counts before**")
            st.write(before[before > 0])
            st.markdown("**Missing counts after**")
            st.write(after[after > 0])
            st.markdown("**Sample rows (before -> after)**")
            merged = pd.concat([df.head(5), df2.head(5)], keys=["raw", "imputed"])
            st.write(merged)

    # --- Module 2 ---
    with tabs[2]:
        st.header("Module 2 — Symmetry & Skewness")
        st.write(visualization.skewness_table(df))
        sel = st.multiselect("Show hist for", options=[c for c in df.columns if c != 'Outcome'], default=["Glucose", "BMI"])
        for s in sel:
            fig = px.histogram(df, x=s, color='Outcome', marginal='box', nbins=40, title=f"Distribution: {s}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Module 3 ---
    with tabs[3]:
        st.header("Module 3 — Scaling & Transformations")
        st.write("Compare raw vs transformed distributions for selected features")
        features = st.multiselect("Features to transform", options=[c for c in df.columns if c != 'Outcome'], default=["Glucose"])
        if features:
            # Build a transformer specifically for the selected features to avoid ColumnTransformer column mismatch
            local_transformer = preprocessing.build_transformer(features, scaler=scaler, transform=transform)
            arr = local_transformer.fit_transform(df)
            arr = np.atleast_2d(arr)
            cols_trans = features[:arr.shape[1]] if arr.shape[1] == len(features) else [f"f{i}" for i in range(arr.shape[1])]
            transformed = pd.DataFrame(arr, columns=cols_trans)

            # Show side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Raw")
                for f in features:
                    st.plotly_chart(px.histogram(df, x=f, nbins=40, title=f"Raw: {f}"), use_container_width=True)
            with col2:
                st.subheader(f"Transformed ({transform} + {scaler})")
                for f in transformed.columns:
                    st.plotly_chart(px.histogram(transformed, x=f, nbins=40, title=f"Transformed: {f}"), use_container_width=True)

    # --- Module 4 ---
    with tabs[4]:
        st.header("Module 4 — Univariate Visualization")
        st.write("Histograms and boxplots for all features")
        cols = [c for c in df.columns if c != 'Outcome']
        if st.button("Show histograms and boxplots"):
            for c in cols:
                st.plotly_chart(px.histogram(df, x=c, color='Outcome', nbins=40, title=f"Histogram: {c}"), use_container_width=True)
                st.plotly_chart(px.box(df, y=c, color=df['Outcome'].astype(str), title=f"Boxplot: {c}"), use_container_width=True)

    # --- Module 5 ---
    with tabs[5]:
        st.header("Module 5 — Dataset Balancing & Entropy")
        counts = df['Outcome'].value_counts()
        st.write("Class counts (raw):")
        st.write(counts)
        st.write(f"Entropy (bits): {compute_entropy_from_counts(counts):.4f}")

        st.write("Simulate balancing and report counts & entropy")
        bal_methods = ["none", "undersample", "oversample", "smote"]
        chosen = st.selectbox("Balancing method to preview", options=bal_methods, index=bal_methods.index(balance_method))
        X, y = preprocessing.split_xy(df)
        Xb, yb = balancing.balance(X, y, method=chosen, random_state=random_state)
        st.write(pd.Series(yb).value_counts())
        st.write(f"Entropy after {chosen}: {compute_entropy_from_counts(pd.Series(yb).value_counts()):.4f}")

    # --- Module 6 ---
    with tabs[6]:
        st.header("Module 6 — Supervised Learning")
        st.write("Train single model or compare several models")
        models = st.multiselect("Models to compare", options=["logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "svm", "knn"], default=[model_name])

        settings = dict(impute_strategy=impute_strategy, scaler=scaler, transform=transform, balance_method=balance_method, test_size=test_size, random_state=random_state)
        if st.button("Compare selected models"):
            with st.spinner("Training models..."):
                df_results = compare_models(df, settings, models)
            st.success("Done")
            st.dataframe(df_results.set_index('model'))
            # Pick best by f1
            best = df_results.sort_values('f1', ascending=False).iloc[0]
            st.write("Best model (by F1):", best.to_dict())
            st.session_state['model_comparison'] = df_results

        if st.button("Train default model"):
            st.info("Training default model — this may take a few seconds")
            X_val, preds_out, model_out = run_supervised(df, impute_strategy, scaler, transform, balance_method, model_name, test_size, random_state)
            st.success("Training complete")
            st.json(model_out["metrics"])
            st.session_state["last_model"] = model_out["pipe"]
            st.session_state["last_preds"] = preds_out
            st.session_state["features"] = list(X_val.columns)

    # --- Module 7 ---
    with tabs[7]:
        st.header("Module 7 — Unsupervised Learning")
        st.write("PCA projection and KMeans clustering with silhouette score")
        pre, feats = prepare_preprocessor(df, scaler=scaler, transform=transform)
        X = df[feats]
        Xtr = pre.named_steps['pre'].fit_transform(X)

        # PCA
        Xp, vr = unsupervised_models.pca_project(pre, X, n_components=pca_components)
        dfp = pd.DataFrame(Xp, columns=[f"pc{i+1}" for i in range(Xp.shape[1])])
        dfp['Outcome'] = df['Outcome'].values
        st.plotly_chart(px.scatter(dfp, x='pc1', y='pc2', color=dfp['Outcome'].astype(str), title='PCA projection'), use_container_width=True)

        # silhouette across k
        ks = list(range(2, min(9, len(df)-1)))
        sils = []
        for k in ks:
            labels, centers = unsupervised_models.kmeans_cluster(pre, X, n_clusters=k)
            try:
                s = silhouette_score(Xtr, labels)
            except Exception:
                s = float('nan')
            sils.append(s)
        fig = px.line(x=ks, y=sils, labels={'x': 'k', 'y': 'silhouette'}, title='Silhouette by k')
        st.plotly_chart(fig, use_container_width=True)

    # --- Module 8 ---
    with tabs[8]:
        st.header("Module 8 — Performance Metrics")
        if 'last_preds' in st.session_state:
            preds_out = st.session_state['last_preds']
            st.subheader('Confusion Matrix')
            st.plotly_chart(metrics.confusion_matrix_fig(preds_out['y_val'], preds_out['y_pred']), use_container_width=True)
            st.subheader('Classification Report')
            st.dataframe(metrics.clf_report_table(preds_out['y_val'], preds_out['y_pred']))

            if preds_out['y_proba'] is not None:
                fpr, tpr, _ = roc_curve(preds_out['y_val'], preds_out['y_proba'])
                roc_auc = auc(fpr, tpr)
                figroc = go.Figure()
                figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})'))
                figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line={'dash': 'dash'}, name='random'))
                figroc.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                st.plotly_chart(figroc, use_container_width=True)

                precision, recall, _ = precision_recall_curve(preds_out['y_val'], preds_out['y_proba'])
                ap = average_precision_score(preds_out['y_val'], preds_out['y_proba'])
                figpr = go.Figure()
                figpr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AP={ap:.3f})'))
                figpr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
                st.plotly_chart(figpr, use_container_width=True)
        else:
            st.info('Train a model (Module 6) to populate performance metrics.')

    # --- Conclusion ---
    with tabs[9]:
        st.header("Conclusion & Export")
        st.write("Summaries and export options")
        if 'model_comparison' in st.session_state:
            st.subheader('Model comparison table')
            st.dataframe(st.session_state['model_comparison'].set_index('model'))
            csv = st.session_state['model_comparison'].to_csv(index=False).encode('utf-8')
            st.download_button('Download comparison CSV', data=csv, file_name='model_comparison.csv')

        if 'last_model' in st.session_state:
            st.subheader('Last trained model')
            st.write(st.session_state['last_model'])
            model_bytes = pickle.dumps(st.session_state['last_model'])
            st.download_button('Download model (pickle)', data=model_bytes, file_name='last_model.pkl')

    # Single sample prediction panel (unchanged)
    st.sidebar.markdown("---")
    st.sidebar.header("Single prediction")
    if "features" in st.session_state:
        features = st.session_state["features"]
    else:
        features = [c for c in df.columns if c != "Outcome"]

    sample = {}
    for f in features:
        col = df[f]
        lo = float(max(col.min(), 0.0))
        hi = float(col.max())
        default = float(col.median())
        sample[f] = st.sidebar.slider(f, min_value=lo, max_value=hi, value=default)

    if st.sidebar.button("Predict sample"):
        if "last_model" not in st.session_state:
            st.sidebar.warning("No model trained yet. Train a model first.")
        else:
            pred, proba = single_sample_prediction(st.session_state["last_model"], sample, features)
            st.sidebar.success(f"Predicted class: {int(pred)}")
            if proba is not None:
                st.sidebar.info(f"Estimated probability of positive: {proba:.3f}")


if __name__ == "__main__":
    main()
