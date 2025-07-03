"""
modeling.py
Supervised modeling utilities for tweet sentiment analysis pipeline.
Requires pandas, numpy, scikit-learn, and imbalanced-learn to be installed in your environment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

MODELS = [
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("SVM", SVC(probability=True, random_state=42)),
    ("ANN", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)),
]

def train_and_evaluate_models(
    X: np.ndarray, y: np.ndarray, models: List[Tuple[str, object]] = MODELS, oversample: bool = True
) -> pd.DataFrame:
    """
    Train and evaluate classifiers using cross-validation and oversampling.
    Args:
        X: Feature matrix.
        y: Target array.
        models: List of (name, model) tuples.
        oversample: Whether to use RandomOverSampler for class imbalance.
    Returns:
        DataFrame with evaluation metrics for each model.
    """
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)
    results = []
    for name, model in models:
        y_pred = cross_val_predict(model, X, y, cv=10, method="predict_proba")[:, 1] if hasattr(model, "predict_proba") else cross_val_predict(model, X, y, cv=10)
        y_pred_binary = np.round(y_pred)
        acc = accuracy_score(y, y_pred_binary)
        roc_auc = roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else np.nan
        precision = precision_score(y, y_pred_binary, zero_division=0)
        recall = recall_score(y, y_pred_binary, zero_division=0)
        f1 = f1_score(y, y_pred_binary, zero_division=0)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "ROC AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "MSE": mse,
            "MAE": mae
        })
    return pd.DataFrame(results) 