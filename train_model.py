"""
G-12: Smart Healthcare Patient Disease Risk Analytics
Training pipeline:
- 70/30 stratified train/test split
- Standardization
- PCA for high-dimensional feature reduction
- Decision Tree with Gini vs Entropy
- GridSearchCV hyperparameter tuning
- Test-set evaluation
- joblib model serialization
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "metrics.json"


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, list(data.feature_names)


def build_search():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("tree", DecisionTreeClassifier(random_state=42)),
    ])
    param_grid = {
        "pca__n_components": [0.90, 0.95, 0.99],
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [3, 5, 7, None],
        "tree__min_samples_split": [2, 5, 10],
    }
    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )


def train_model(save=True):
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    search = build_search()
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)

    metrics = {
        "dataset": "scikit-learn Breast Cancer Wisconsin (Diagnostic)",
        "samples": int(len(X)),
        "features_before_pca": int(X.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": 0.30,
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "pca_components_selected": int(search.best_estimator_.named_steps["pca"].n_components_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions)),
        "test_recall": float(recall_score(y_test, predictions)),
        "test_f1": float(f1_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "gini_best_cv_f1": float(max(
            r["mean_test_score"] for r in search.cv_results_ if str(r["param_tree__criterion"]) == "gini"
        )),
        "entropy_best_cv_f1": float(max(
            r["mean_test_score"] for r in search.cv_results_ if str(r["param_tree__criterion"]) == "entropy"
        )),
    }

    if save:
        joblib.dump(
            {
                "pipeline": search.best_estimator_,
                "feature_names": feature_names,
                "class_names": {0: "Malignant (higher disease risk)", 1: "Benign (lower disease risk)"},
            },
            MODEL_PATH,
        )
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return search, metrics, feature_names


if __name__ == "__main__":
    _, metrics, _ = train_model(save=True)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
