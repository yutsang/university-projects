#!/usr/bin/env python
# coding: utf-8

# ## polynomial regression
# 
# assignment2.csv is the data for you to do analysis on. It is the data to predict cars' prices.
# Below is the guideline:
# 1. Read the data, call it data_original
# 2. Target = 'price'
# 3. Fix the features, e.g., features_numerical = \['A', 'B', 'C', 'D'\], features_category=\['E','F'\].
# 4. Preprocess the data
#  - add one-hot encoding for categorical to `data`
#  - add polynomial to `data`
# 5. Fit to model, `model.fit(data.drop[target], data[target])`
# 6. Predict the model by `data`
# 
# $^*$You may use the function `create_poly_feature()` defined as follows to generate the polynomials.

# In[357]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from typing import List, Optional

def create_poly_feature(df: pd.DataFrame, feature: str, degree: int) -> pd.DataFrame:
    """Create polynomial features for a given column up to a specified degree."""
    result = pd.DataFrame()
    if feature in df.columns:
        for power in range(2, degree + 1):
            name = f"{feature}_power_{power}"
            result[name] = df[feature].astype(float) ** power
        return result
    else:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

def onehot_encode(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """One-hot encode a categorical feature."""
    if feature in df.columns:
        return pd.get_dummies(df, columns=[feature])
    else:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

def preprocess_data(data_path: str = 'assignment2-1.csv') -> pd.DataFrame:
    """Read and preprocess the dataset, including one-hot encoding and polynomial features."""
    data_origin = pd.read_csv(data_path)
    cat_feature_make = onehot_encode(data_origin, 'make')
    cat_feature_mnd = onehot_encode(cat_feature_make, 'drive-wheels')
    poly_feature_length = create_poly_feature(data_origin, 'length', 2)
    poly_feature_height = create_poly_feature(data_origin, 'height', 3)
    data_processed = pd.concat([cat_feature_mnd, poly_feature_length, poly_feature_height], axis=1)
    return data_processed

def fit_and_evaluate(data_processed: pd.DataFrame) -> None:
    """Fit a linear regression model and evaluate it with RMSE and custom loss."""
    x = data_processed.drop('price', axis=1)
    y = data_processed['price']
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    pred = regr.predict(x)
    print("Model coefficients:")
    print(pd.DataFrame(regr.coef_, x.columns, columns=['Coeff']))
    print(f"Intercept: {regr.intercept_}")
    plt.scatter(y, pred, color="blue")
    m, b = np.polyfit(y, pred, 1)
    plt.plot(y, m * y + b, color="black")
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.show()
    plt.hist(y - pred)
    plt.title('Residuals Histogram')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.show()
    rmse = np.sqrt(metrics.mean_squared_error(y, pred))
    print("Root Mean Square Error (RMSE):", rmse)
    loss = np.sum(np.abs(y - pred))
    print("Custom loss (sum of absolute errors):", loss)

def main():
    print("There are 6 features: 'make', 'drive-wheels', 'length', 'width', 'height', and 'price'.")
    print("Categorical features in the dataset are 'make' and 'drive-wheels'.")
    data_processed = preprocess_data()
    fit_and_evaluate(data_processed)

if __name__ == "__main__":
    main()


# ### read the dataset assignment2.csv and answer the following question:
# - How many features in this dataset and what are their names?
# - Which are categorical features? List their names.

# In[358]:


import pandas as pd
data_origin = pd.read_csv('assignment2-1.csv')
target = data_origin['price']

# rwd( rear wheels drive)
# fwd( front wheels drive)
# make (company)
def onehot_encode(df, feature):
    result = pd.DataFrame()
    if feature in df.columns:
        # loop over the degrees:
        result = pd.get_dummies(df, columns=[feature])
        return result
    else:  
        return print("Please select a feature in this df!")

cat_feature_make = onehot_encode(data_origin, 'make')
cat_feature_mnd = onehot_encode(cat_feature_make, 'drive-wheels')
features_num = ['length', 'width', 'height']
features_cat = ['make', 'drive-wheels']
print("There are 6 features, their names are 'make', 'drive-wheels', 'length', 'width', 'height', and 'price'.")
print("Categorical features in the dataset are 'make'and 'drive-wheels' ")
data_origin


# ### add the following features to the original dataset:
# - degree of 2 polynomial of `length`$^*$
# - degree of 3 polynomial of `height`$^*$

# In[359]:


poly_feature_length = create_poly_feature(data_origin, 'length', 2)
poly_feature_height = create_poly_feature(data_origin, 'height', 3)


# ### consider the whole data set as training set and fit the model.

# In[360]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
target = ['price']
data_processed = pd.concat([cat_feature_mnd, poly_feature_length, poly_feature_height],axis = 1)
x = data_processed.drop('price',axis=1)
y = data_processed['price']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.9, random_state = 0)
regr = linear_model.LinearRegression()
#regr.fit(x_train, y_train)
regr.fit(x, y)
print(regr.coef_)
print(regr.intercept_)
pd = pd.DataFrame(regr.coef_, x.columns, columns = ['Coeff'])
#pred = regr.predict(x_test)
pred = regr.predict(x)
plt.scatter(y, pred, color="blue")

m, b = np.polyfit(y, pred, 1)
plt.plot(y, m*y + b, color="black")

plt.show()
print(pd)


# In[361]:


plt.hist(y - predictions)


# ### write down the formula of RMSE and compute its value of this model. 
# 
# - Hint: you can use $y_i$ as the value of i-th target and $\hat{y}_i$ as the i-th predicted target.

# The formula of RMSE:
# 
# RMSE = $\sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{(y_i -\hat{y}_i)^2}} $
# 

# In[362]:


import math
RMSE = np.sqrt(metrics.mean_squared_error(y, pred))
print("Root Mean Square Error: ", RMSE)


# ### compute the following user-defined metric of this model: 
# 
# - $loss = \sum_{i} |y_i -\hat{y}_i |$, where $y_i$ is the value of i-th target and $\hat{y}_i$ is the i-th predicted target.

# In[363]:


loss = np.sum(abs(np.subtract(y, pred)))
print("loss: ", loss)

