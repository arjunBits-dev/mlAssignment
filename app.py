"""ML Assignment 2 — Online Shoppers Purchasing Intention.

Streamlit app that evaluates five pre-trained classifiers on an uploaded
test CSV (all input features + 'Revenue' target column).

The models and the preprocessing pipeline are loaded from the model/
folder — nothing is retrained here.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ----------------------------- configuration -----------------------------

MODEL_DIR = Path(__file__).parent / "model"
TARGET_COL = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

# ----------------------------- cached loaders -----------------------------


@st.cache_resource
def load_preprocessor():
    """Load the fitted preprocessing pipeline (scaler + one-hot encoder)."""
    return joblib.load(MODEL_DIR / "preprocessor.joblib")


@st.cache_resource
def load_model(display_name: str):
    """Load one trained classifier from disk."""
    return joblib.load(MODEL_DIR / MODEL_FILES[display_name])


# ----------------------------- helpers -----------------------------


def coerce_target(series: pd.Series) -> pd.Series:
    """Accept Revenue as 0/1, True/False or 'TRUE'/'FALSE' strings."""
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        return (
            series.astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
        )
    return series.astype(int)


def validate_upload(df: pd.DataFrame, expected_features: list[str]) -> list[str]:
    """Return a list of problems with the uploaded CSV (empty list = OK)."""
    problems = []
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        problems.append(f"Missing feature columns: {', '.join(missing)}")
    if TARGET_COL not in df.columns:
        problems.append(f"Missing target column '{TARGET_COL}'")
    if len(df) == 0:
        problems.append("The file contains no rows")
    return problems


# ----------------------------- page layout -----------------------------

st.set_page_config(page_title="Online Shoppers — Model Evaluator", layout="wide")

st.title("🛒 Online Shoppers Purchasing Intention — Model Evaluator")
st.caption(
    "Upload a test CSV, pick one of the five pre-trained classifiers, and "
    "inspect its performance. Models were trained offline on the UCI "
    "Online Shoppers Purchasing Intention dataset and are only *loaded* here."
)

preprocessor = load_preprocessor()
expected_features = list(preprocessor.feature_names_in_)

with st.sidebar:
    st.header("1️⃣ Upload test data")
    uploaded = st.file_uploader(
        "CSV with all 17 input features + the 'Revenue' target column",
        type="csv",
    )
    st.header("2️⃣ Choose a model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()))

    with st.expander("Expected CSV columns"):
        st.code("\n".join(expected_features + [TARGET_COL]), language="text")

if uploaded is None:
    st.info("⬅️ Upload a CSV in the sidebar to begin (e.g. the repo's test_data.csv).")
    st.stop()

# ----------------------------- evaluation -----------------------------

data = pd.read_csv(uploaded)
issues = validate_upload(data, expected_features)
if issues:
    for issue in issues:
        st.error(issue)
    st.stop()

st.subheader("Uploaded data preview")
st.dataframe(data.head(10), use_container_width=True)
st.write(f"Rows: **{len(data)}** | Purchase rate: **{coerce_target(data[TARGET_COL]).mean():.2%}**")

# Transform raw features with the SAME pipeline used at training time
X_raw = data[expected_features].copy()
if "Weekend" in X_raw.columns and X_raw["Weekend"].dtype == bool:
    X_raw["Weekend"] = X_raw["Weekend"].astype(int)
y_true = coerce_target(data[TARGET_COL])

X_prepared = preprocessor.transform(X_raw)

model = load_model(model_name)
y_pred = model.predict(X_prepared)
y_prob = model.predict_proba(X_prepared)[:, 1]

# ----------------------------- metrics -----------------------------

st.subheader(f"📊 Evaluation metrics — {model_name}")

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_prob),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

cols = st.columns(6)
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, f"{value:.4f}")

# ----------------------------- confusion matrix + report -----------------------------

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No purchase", "Purchase"],
        zero_division=0,
        output_dict=True,
    )
    st.dataframe(
        pd.DataFrame(report).T.round(4),
        use_container_width=True,
    )

st.caption(
    "Metrics are computed on the uploaded data using the saved model — "
    "no retraining happens in this app."
)
