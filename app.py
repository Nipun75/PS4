"""
Streamlit web application for G-12 Smart Healthcare Patient Disease Risk Analytics.
Run: streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from train_model import load_data, train_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "metrics.json"

st.set_page_config(
    page_title="Smart Healthcare Disease Risk Analytics",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        train_model(save=True)
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_reference_data():
    X, _, feature_names = load_data()
    return X, feature_names


def format_metric(value):
    return f"{value * 100:.2f}%"


bundle = get_model()
pipeline = bundle["pipeline"]
feature_names = bundle["feature_names"]
class_names = bundle["class_names"]
X, _ = get_reference_data()

st.title("🩺 Smart Healthcare Patient Disease Risk Analytics")
st.caption(
    "G-12 | High-dimensional PCA + Decision Tree classification | "
    "Gini vs Entropy comparison"
)

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
else:
    _, metrics, _ = train_model(save=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Accuracy", format_metric(metrics["test_accuracy"]))
c2.metric("Precision", format_metric(metrics["test_precision"]))
c3.metric("Recall", format_metric(metrics["test_recall"]))
c4.metric("F1 Score", format_metric(metrics["test_f1"]))

st.divider()

with st.expander("Model & experiment details", expanded=True):
    st.write(
        f"**PCA:** {metrics['features_before_pca']} original features → "
        f"**{metrics['pca_components_selected']} principal components** "
        f"(best grid-search setting)"
    )
    st.write(f"**Best parameters:** {metrics['best_params']}")
    st.write(
        f"**Best CV F1:** {metrics['best_cv_f1']:.4f} | "
        f"**Gini best CV F1:** {metrics['gini_best_cv_f1']:.4f} | "
        f"**Entropy best CV F1:** {metrics['entropy_best_cv_f1']:.4f}"
    )

st.header("Patient Risk Prediction")
st.write(
    "Enter diagnostic measurements. Values are initialized to dataset medians "
    "so the demo is immediately usable."
)

defaults = X.median()
input_values = {}

left, right = st.columns(2)
for i, feature in enumerate(feature_names):
    target_col = left if i % 2 == 0 else right
    with target_col:
        input_values[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=float(defaults[feature]),
            format="%.6f",
            help=f"Reference median: {defaults[feature]:.6f}",
        )

if st.button("Analyze Disease Risk", type="primary", use_container_width=True):
    row = pd.DataFrame([[input_values[f] for f in feature_names]], columns=feature_names)
    prediction = int(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]
    risk_probability = float(probabilities[0])

    if prediction == 0:
        st.error(
            f"⚠️ Model classification: {class_names[prediction]}\n\n"
            f"Estimated malignant-class probability: {risk_probability * 100:.2f}%"
        )
    else:
        st.success(
            f"✅ Model classification: {class_names[prediction]}\n\n"
            f"Estimated malignant-class probability: {risk_probability * 100:.2f}%"
        )

    st.info(
        "This is an educational machine-learning demonstration, not a medical diagnosis. "
        "Clinical decisions must be made by qualified healthcare professionals."
    )

st.divider()
st.subheader("Experiment Summary")
st.markdown(
    """
- Problem: classify diagnostic measurements into malignant/benign classes.
- Preprocessing: standardization followed by PCA.
- Dimensionality reduction: PCA retains 90%, 95%, or 99% variance during grid search.
- Models: Decision Tree with both Gini and Entropy criteria.
- Validation: 5-fold GridSearchCV on the 70% training partition.
- Final evaluation: Accuracy, Precision, Recall, F1-score and confusion matrix on the untouched 30% test set.
- Deployment: serialized pipeline with joblib and served through Streamlit.
"""
)
