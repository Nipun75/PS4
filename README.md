# G-12 — Smart Healthcare Patient Disease Risk Analytics

An end-to-end Machine Learning mini-project for the practical requirement:
G-12: Smart Healthcare Patient Disease Risk Analytics.

The implementation covers high-dimensional PCA and a Gini-vs-Entropy Decision Tree comparison.

## Features

- 70/30 stratified train-test split
- StandardScaler preprocessing
- PCA dimensionality reduction
- GridSearchCV hyperparameter tuning
- Decision Tree comparison using Gini and Entropy
- Accuracy, Precision, Recall and F1 evaluation
- Confusion matrix
- Model serialization with joblib
- Interactive Streamlit prediction dashboard

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn.

It contains 569 samples and 30 numeric diagnostic features. Target classes:
- 0 — malignant (higher disease risk)
- 1 — benign (lower disease risk)

Using the built-in dataset keeps the project reproducible and avoids dependence on an external CSV URL.

## Project structure

PS4/
- app.py
- train_model.py
- requirements.txt
- README.md
- .gitignore
- test_training.py

Running the training script additionally creates model.joblib and metrics.json. These generated artifacts are intentionally ignored by Git.

## ML workflow

1. Load the dataset.
2. Split into 70% training and 30% testing data using stratification.
3. Standardize the 30 original features.
4. Apply PCA and tune retained variance: 90%, 95%, 99%.
5. Train Decision Trees with both gini and entropy.
6. Tune tree depth and minimum split size using 5-fold GridSearchCV.
7. Evaluate the best pipeline on the untouched test set.
8. Serialize the fitted preprocessing + PCA + classifier pipeline with joblib.
9. Load that serialized pipeline in Streamlit for real-time prediction.

## Run locally

python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py

Then open the Streamlit URL shown in the terminal.

## Reproducibility

The experiment uses fixed random seeds (42) for the train/test split, PCA and Decision Tree. The best model is selected using F1 score on the training partition with 5-fold cross-validation.

## Practical mapping

Requirement | Implementation
--- | ---
Domain-specific ML problem | Healthcare disease risk classification
70/30 split | train_test_split(..., test_size=0.30)
Feature scaling | StandardScaler
High-dimensional reduction | PCA
Decision Tree | DecisionTreeClassifier
Gini vs Entropy | Grid search over both criteria
Hyperparameter tuning | GridSearchCV
Evaluation | Accuracy, Precision, Recall, F1
Serialization | joblib
Web UI | Streamlit

## Important note

This is an academic ML project. Its predictions are not a medical diagnosis and must not be used for clinical decision-making.
