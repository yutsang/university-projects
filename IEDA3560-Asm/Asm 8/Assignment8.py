#!/usr/bin/env python
# coding: utf-8

# # Assignment: Sentiment Classification by AdaBoost and Random Forest
# 
# This dataset was chosen to contain similar numbers of positive and negative reviews. Here is our objective: **for a given review, we want to predict whether its sentiment is positive or negetive**. In this assignment, you need to establish both AdaBoost model and Random Forest model to solve this problem. 
# 
# We have processed the dataset for you. Load file `sentiment_classification_processed_data.csv` to access the processed data. In the processed dataset, the column `sentiment` is our target label, with 1 means positive and -1 means negative. All the other columns except `sentiment` are our input features, and each column is corresponding to one important word.
# 
# You need to do the following tasks:
# 1. Split dataset into training set and testing set using an 80/20 split.
# 2. Generate a logistic regression model, fit the model by training set and calculate the accuracy on testing set.
# 3. Establish an AdaBoost model with the following setting: `n_estimators=5`, `random_state=1`. Calculate the accuracy on training set and test set.
# 4. Establish a Random Forest model with the following setting: `n_estimators=5`, `random_state=1`. Calculate the accuracy on training set and test set.
# 5. Do crossvalidation for AdaBoost. Generate 4 different AdaBoost models by setting `max_depth=2, 5, 10 and 20`. Fix `random_state=1` and `n_estimators=50` for these 4 models. Calculate the accuracy on training set and testing set for all these 4 models.
# 6. Do crossvalidation for Random Forest. Generate 4 Random Forest models by setting `n_estimators=5, 10, 50 and 100`. Fix `random_state=1` and `max_depth` to be default value for these 4 models. Calculate the accuracy on training set and testing set for all these 4 models.

# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple, Optional

def load_data(csv_path: str = 'sentiment_classification_processed_data.csv') -> Tuple[pd.DataFrame, pd.Series]:
    """Load and split features and label from the dataset."""
    data = pd.read_csv(csv_path)
    features = data.drop(columns=['sentiment'])
    label = data['sentiment']
    return features, label

def split_data(features: pd.DataFrame, label: pd.Series, test_size: float = 0.2, random_state: int = 0) -> Tuple:
    """Split data into training and testing sets."""
    return train_test_split(features, label, test_size=test_size, random_state=random_state)

def logistic_regression(train_X, train_Y, test_X, test_Y) -> None:
    """Train and evaluate a logistic regression model."""
    logistic = linear_model.LogisticRegression()
    logistic.fit(train_X, train_Y)
    print(f'Logistic Regression accuracy on the testing set: {logistic.score(test_X, test_Y):.2f}')

def adaboost_model(train_X, train_Y, test_X, test_Y, n_estimators: int = 5, max_depth: Optional[int] = None) -> None:
    """Train and evaluate an AdaBoost model."""
    base_estimator = DecisionTreeClassifier(max_depth=max_depth) if max_depth is not None else DecisionTreeClassifier()
    adaboost = AdaBoostClassifier(base_estimator, random_state=1, n_estimators=n_estimators)
    adaboost.fit(train_X, train_Y)
    print(f'AdaBoost (n_estimators={n_estimators}, max_depth={max_depth}) accuracy on training set: {adaboost.score(train_X, train_Y):.2f}')
    print(f'AdaBoost (n_estimators={n_estimators}, max_depth={max_depth}) accuracy on testing set: {adaboost.score(test_X, test_Y):.2f}')

def random_forest_model(train_X, train_Y, test_X, test_Y, n_estimators: int = 5, max_depth: Optional[int] = None) -> None:
    """Train and evaluate a Random Forest model."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=1, max_depth=max_depth)
    rf.fit(train_X, train_Y)
    print(f'Random Forest (n_estimators={n_estimators}, max_depth={max_depth}) accuracy on training set: {rf.score(train_X, train_Y):.2f}')
    print(f'Random Forest (n_estimators={n_estimators}, max_depth={max_depth}) accuracy on testing set: {rf.score(test_X, test_Y):.2f}')

def crossvalidate_adaboost(train_X, train_Y, test_X, test_Y):
    """Cross-validation for AdaBoost with different max_depth values."""
    for max_depth in [2, 5, 10, 20]:
        adaboost_model(train_X, train_Y, test_X, test_Y, n_estimators=50, max_depth=max_depth)

def crossvalidate_random_forest(train_X, train_Y, test_X, test_Y):
    """Cross-validation for Random Forest with different n_estimators values."""
    for n_estimators in [5, 10, 50, 100]:
        random_forest_model(train_X, train_Y, test_X, test_Y, n_estimators=n_estimators)

def main():
    features, label = load_data()
    train_X, test_X, train_Y, test_Y = split_data(features, label)
    print('--- Logistic Regression ---')
    logistic_regression(train_X, train_Y, test_X, test_Y)
    print('\n--- AdaBoost (n_estimators=5) ---')
    adaboost_model(train_X, train_Y, test_X, test_Y, n_estimators=5)
    print('\n--- Random Forest (n_estimators=5) ---')
    random_forest_model(train_X, train_Y, test_X, test_Y, n_estimators=5)
    print('\n--- Cross-validation: AdaBoost (n_estimators=50, varying max_depth) ---')
    crossvalidate_adaboost(train_X, train_Y, test_X, test_Y)
    print('\n--- Cross-validation: Random Forest (varying n_estimators) ---')
    crossvalidate_random_forest(train_X, train_Y, test_X, test_Y)

if __name__ == "__main__":
    main()

