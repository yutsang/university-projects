#!/usr/bin/env python
# coding: utf-8

# ## Assignment 10
# 
# In this assignment, you will get an on-hand experience of utilizing PCA as a dimensionality reduction tool to extract features. 
# 
# Specifically,
# 
# 1. Load the digits data from sklearn.
# 
# 2. Perform a PCA on the dataset **without** specifying `n_components` and which direction is the main principal component? Namely, along the direction, the variance of sample points is the largest.
# 
# 3. Compute the cumulative variance ratio of all the components. If we request that the PCA method should preserve at least 50% of the total variance, what is the minimum number of principal components?
# 
# 4. Choose the best number ($N$) of components by cross-validation. In order to achieve it,first you need to apply the PCA with different $N$ to transform the image data. Then, you are required to apply the logistic regression to do the classification with transformed data as $X$ and the corresponding labels as $Y$. Finally, you can do the cross validation for each $N$.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from typing import Tuple, List


# In[2]:


#1.
digits = load_digits()
data = digits.data
labels = digits.target


# In[3]:


#2.
pca = PCA().fit(data)
print(pca.explained_variance_)


# In[4]:


max(pca.explained_variance_)


# In[5]:


pca.components_[0]


# In[6]:


#3.
np.cumsum(pca.explained_variance_ratio_)


# In[7]:


#4
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
sns.set()


# In[8]:


pca = PCA()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=1)
n = data.shape[1]
for i in range(1, n+1):
    X_reduced_train = pca.fit_transform(scale(X_train))[:, :i] 

from sklearn import linear_model
logistic = linear_model.LogisticRegression(max_iter=5000)
logistic.fit(X_reduced_train, y_train)


# In[9]:


mse = []
# 10-fold cv
score = -cross_val_score(logistic, 
    np.ones((len(X_reduced_train), 1)), y_train,
    cv=10, scoring='neg_mean_squared_error')

mse.append(score.mean())

# Calculate MSE using cv for the 13 components, adding one at a time
for i in range(1, n+1):
    score = -cross_val_score(logistic, X_reduced_train[:, :i], y_train, cv=10, scoring='neg_mean_squared_error')
    mse.append(score.mean())

# Plot results
plt.figure(figsize=(10,6))
plt.plot(mse, marker='o')
plt.xlabel('Number of principal components')
plt.ylabel('MSE')


# In[10]:


a = []
for i in range(1, n+1):
    X_reduced_test = pca.transform(scale(X_test))[:, :i]
    logistic.fit(X_reduced_train[:, :i], y_train)
    pred = logistic. predict(X_reduced_test)
    a.append(mean_squared_error(y_test, pred))


# In[11]:


min(a)


# In[13]:


index = []
for i in range(len(a)):
    if a[i] == min(a): index.append(i)

index


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the digits dataset."""
    digits = load_digits()
    return digits.data, digits.target

def perform_pca(data: np.ndarray) -> PCA:
    """Fit PCA on the data and return the fitted PCA object."""
    pca = PCA().fit(data)
    print("Explained variance:", pca.explained_variance_)
    print("Max explained variance:", max(pca.explained_variance_))
    print("First principal component:", pca.components_[0])
    return pca

def cumulative_variance(pca: PCA) -> np.ndarray:
    """Compute and return the cumulative variance ratio."""
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print("Cumulative variance ratio:", cum_var)
    return cum_var

def min_components_for_variance(cum_var: np.ndarray, threshold: float = 0.5) -> int:
    """Return the minimum number of components to reach the given variance threshold."""
    n_components = np.argmax(cum_var >= threshold) + 1
    print(f"Minimum number of components to preserve {threshold*100}% variance: {n_components}")
    return n_components

def cross_validate_pca_logistic(data: np.ndarray, labels: np.ndarray) -> List[float]:
    """Cross-validate logistic regression for each number of principal components."""
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=1)
    n = data.shape[1]
    mse = []
    for i in range(1, n + 1):
        pca = PCA(n_components=i)
        X_reduced_train = pca.fit_transform(scale(X_train))
        logistic = LogisticRegression(max_iter=5000)
        # 10-fold cross-validation MSE
        score = -cross_val_score(logistic, X_reduced_train, y_train, cv=10, scoring='neg_mean_squared_error')
        mse.append(score.mean())
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n + 1), mse, marker='o')
    plt.xlabel('Number of principal components')
    plt.ylabel('Cross-validated MSE')
    plt.title('MSE vs Number of Principal Components')
    plt.show()
    return mse

def test_set_mse(data: np.ndarray, labels: np.ndarray) -> Tuple[List[float], int]:
    """Compute test set MSE for each number of principal components and return the list and the best index."""
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=1)
    n = data.shape[1]
    a = []
    for i in range(1, n + 1):
        pca = PCA(n_components=i)
        X_reduced_train = pca.fit_transform(scale(X_train))
        X_reduced_test = pca.transform(scale(X_test))
        logistic = LogisticRegression(max_iter=5000)
        logistic.fit(X_reduced_train, y_train)
        pred = logistic.predict(X_reduced_test)
        a.append(mean_squared_error(y_test, pred))
    min_mse = min(a)
    best_indices = [i + 1 for i, v in enumerate(a) if v == min_mse]
    print(f"Minimum test set MSE: {min_mse:.4f} at components: {best_indices}")
    return a, best_indices[0]

def main():
    data, labels = load_data()
    print('--- PCA Analysis ---')
    pca = perform_pca(data)
    cum_var = cumulative_variance(pca)
    min_components_for_variance(cum_var, threshold=0.5)
    print('\n--- Cross-validation for Logistic Regression with PCA ---')
    mse = cross_validate_pca_logistic(data, labels)
    print('\n--- Test Set MSE for Each Number of Components ---')
    a, best_n = test_set_mse(data, labels)
    print(f'Best number of components by test set MSE: {best_n}')

if __name__ == "__main__":
    main()

