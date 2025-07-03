#!/usr/bin/env python
# coding: utf-8

# ## Assignment: Cross Validation of Logistic Regression Model
# 
# Following assignment-1, we have a dataset for titanic passengers. Here is the first 5 rows of the dataset:

# In[116]:


import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import combinations
from typing import List, Tuple

def all_subsets(my_list: List[str]) -> List[List[str]]:
    """Return all non-empty subsets of a list."""
    subs = []
    for i in range(1, len(my_list) + 1):
        subs += combinations(my_list, i)
    return [list(i) for i in subs]

def preprocess_data(csv_path: str = 'titanic_cross_validation.csv') -> pd.DataFrame:
    """Read and preprocess the Titanic dataset."""
    data = pd.read_csv(csv_path)
    le = preprocessing.LabelEncoder()
    data['male'] = le.fit_transform(data['male'])
    data['Q'] = le.fit_transform(data['Q'])
    data['S'] = le.fit_transform(data['S'])
    return data

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    train_valid_XY, test_XY = train_test_split(data, test_size=0.2, random_state=0)
    train_XY, valid_XY = train_test_split(train_valid_XY, test_size=0.2, random_state=0)
    return train_XY, valid_XY, test_XY

def cross_validate(train_XY: pd.DataFrame, valid_XY: pd.DataFrame, features_subs: List[List[str]]) -> np.ndarray:
    """Fit logistic regression for each feature subset and return validation accuracies."""
    accuracy_cv = []
    for features in features_subs:
        train_X, train_Y = train_XY[features], train_XY['Survived']
        valid_X, valid_Y = valid_XY[features], valid_XY['Survived']
        logisticReg = linear_model.LogisticRegression()
        logisticReg.fit(train_X, train_Y)
        sub_accuracy_cv = logisticReg.score(valid_X, valid_Y)
        accuracy_cv.append(sub_accuracy_cv)
    return np.array(accuracy_cv)

def fit_best_model(train_valid_XY: pd.DataFrame, test_XY: pd.DataFrame, best_features: List[str]) -> Tuple[linear_model.LogisticRegression, pd.Series, pd.Series]:
    """Fit logistic regression with the best feature subset and return model and test data."""
    train_valid_X, train_valid_Y = train_valid_XY[best_features], train_valid_XY['Survived']
    test_X, test_Y = test_XY[best_features], test_XY['Survived']
    logisticReg = linear_model.LogisticRegression()
    logisticReg.fit(train_valid_X, train_valid_Y)
    return logisticReg, test_X, test_Y

def plot_roc_curve(test_Y: pd.Series, test_X: pd.DataFrame, model: linear_model.LogisticRegression) -> None:
    """Plot ROC curve for the optimal model."""
    fpr, tpr, thresholds = roc_curve(test_Y, model.predict_proba(test_X)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(dpi=100)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()

def main():
    data = preprocess_data()
    features = list(data.columns[1:])
    features_subs = all_subsets(features)
    train_XY, valid_XY, test_XY = split_data(data)
    accuracy_cv = cross_validate(train_XY, valid_XY, features_subs)
    best_idx = accuracy_cv.argmax()
    best_features = features_subs[best_idx]
    print(f'Best feature subset: {best_features}')
    print(f'Validation accuracy: {accuracy_cv[best_idx]:.4f}')
    train_valid_XY = pd.concat([train_XY, valid_XY])
    model, test_X, test_Y = fit_best_model(train_valid_XY, test_XY, best_features)
    test_accuracy = model.score(test_X, test_Y)
    print(f'Test accuracy with best feature subset: {test_accuracy:.4f}')
    plot_roc_curve(test_Y, test_X, model)

if __name__ == "__main__":
    main()

