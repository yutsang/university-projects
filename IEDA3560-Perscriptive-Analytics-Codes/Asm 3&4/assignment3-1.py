#!/usr/bin/env python
# coding: utf-8

# # Assignment-3
# 
# TSANG YU 20705565

# `assignment3.csv`  is the data for you to do analysis on.  It is the data to predict cars' prices.

# ## best feature selection

# 1. The complete documentation on the dataset can be found here for your reference: https://archive.ics.uci.edu/ml/datasets/Automobile
# 2. Delete rows with missing data points (denoted with "?")
# 3. You should convert the data types of some features to the appropriate data types (i.e. float, integer, etc). 
# 4. You should be able to distinguish numerical from categorical features. 
# 5. target = \['price'\]
# 6. When doing "train_test_split", use random_state = 20220303
# 7. You need to use greedy algorithm on linear models to find a best feature subset which gives us the smallest mean squared error corresponding validation set. Finally, compute the root mean squared error (RMSE) of test set.

# Note: it will be much long for greedy algorithms to enumerate all combinations of features, so you only need to enumerate the cases when the number of features are less than or equal to 3.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from typing import List, Tuple


# In[2]:


file = pd.read_csv('assignment3-1.csv')
file


# In[3]:


file = file.replace('?', np.nan).dropna()
file


# In[4]:


file.dtypes


# In[5]:


type_dict = {'normalized-losses':int, 'bore':float , 'stroke':float,
             'compression-ratio':float, 'horsepower':int,'peak-rpm':int,
             'price':int }
file = file.astype(type_dict)
file.dtypes


# In[6]:


file.columns


# In[7]:


features_numerical = ['normalized-losses','wheel-base', 'length', 'width', 'height', 'curb-weight',
                      'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                        'city-mpg', 'highway-mpg']
features_categorical = ['symboling','make', 'fuel-type', 'aspiration','num-of-doors', 'body-style', 
                        'drive-wheels', 'engine-location','engine-type','num-of-cylinders', 'fuel-system']

features = features_categorical + features_numerical

target = ['price']

file_df = file[features_numerical + features_categorical + target]

file_df


# In[8]:


#pip install sklearn


# In[9]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(file_df, test_size = 0.2, random_state = 20220303) 
train = train.reset_index(drop = True)
test = test.reset_index(drop = True)


# In[10]:


from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# initialize a list to save features
greedy_select = []
# and a numpy array to save their model MSE
MSE_greedy_algo = np.array([])

def onehot_encoder(df, feature): 
    result = pd.DataFrame()
    if feature in df.columns:
        # The following line is important, refer to Assignment 2
        result = pd.get_dummies(df, columns=[feature])
        return result 
    else:
        return print("Please select a feature in this df!")
    
    
def kfold_cv(data, target, n):
    # We need a vector to record mse from k-fold 
    MSE = np.array([])
    # KFold is a build-in function in Scikit-learn
    #    it can help us cut data into n pieces
    # (compare with simple cross validation)
    kf = KFold(n_splits = n, shuffle = True, random_state=20220303)
    
    for train_index, validation_index in kf.split(data): 
        # obtain the train and validation part
        train, valid = data.loc[train_index,:], data.loc[validation_index,:]
        
        # extract X and Y to be fit in a model
        X_train = train.drop(target, axis = 1)
        Y_train = train[target]
        X_valid = valid.drop(target, axis = 1)
        Y_valid = valid[target]
        
        # build linear regression model
        model = linear_model.LinearRegression() 
        
        # fit model using training data
        model.fit(X_train,Y_train)
        
        # predict using validation data
        Y_valid_fit = model.predict(X_valid)
        
        # Calculate MSE
        MSE_temp = mean_squared_error(Y_valid_fit, Y_valid) 
        # Add MSE to the list
        MSE = np.append(MSE, MSE_temp)
    return MSE.mean()

for i in range(len(features)):
    MSE = np.array([])
    features_left = list(set(features) - set(greedy_select))
    
    for new in features_left:
        features_new = greedy_select + [new]
        train_sub = train[features_new + target]
        
        # get all categorical features in sub
        categorical_sub = list(set(features_new) & set(features_categorical))
        
        # if there really are categorical features, # we need to do onthot encoding.
        if len(categorical_sub) != 0:
            for i in categorical_sub:
                # Again, this line is important. Refer to Assignment 2 
                train_sub = onehot_encoder(train_sub, i)
        
        # CrossValidation, compute the mse and save it into MSE_sub
        MSE_sub = kfold_cv(train_sub, 'price', 5)
        MSE = np.append(MSE, MSE_sub)

    # pick the features that gives the smallest MSE
    # and add it into our features list
    # meanwhile, save the corresponding MSE
    greedy_select += [features_left[MSE.argmin()]] 
    MSE_greedy_algo = np.append(MSE_greedy_algo, MSE.min())


# In[11]:


MSE_greedy_algo.argmin()


# In[12]:


features_greedy = greedy_select[:(MSE_greedy_algo.argmin()+1)]
features_greedy


# In[13]:


file_greedy = file_df[features_greedy + target]
file_greedy


# In[14]:


categorical_cv = list(set(features_greedy) & set(features_categorical))

if len(categorical_cv) != 0: 
    for i in categorical_cv:
        file_greedy  = onehot_encoder(file_greedy , i)
        
file_greedy


# In[15]:


train_greedy, test_greedy = train_test_split(file_greedy, test_size = 0.2, random_state = 20220303)

model_greedy = linear_model.LinearRegression()

X_greedy = train_greedy.drop(target, axis = 1)
Y_greedy = train_greedy[target]

model_greedy.fit(X_greedy, Y_greedy)

X_test_greedy = test_greedy.drop(target, axis = 1)
Y_test_greedy = test_greedy[target]

Y_test_greedy_fit = model_greedy.predict(X_test_greedy)
mean_squared_error(Y_test_greedy_fit, Y_test_greedy)


# In[16]:


np.sqrt(mean_squared_error(Y_test_greedy_fit, Y_test_greedy))


# ## Lasso regression

# 8. Use LASSO regression combined with $k$-fold to find a best feature subset which gives us the smallest square root of mean square error corresponding validation set. Use k = 10. Then compute the  square root of mean square error of test set.
# 9. lambda_list = np.array(\[0.001,0.01,0.1,1,10,100,1000,10000\])

# In[17]:


lambda_list = np.array([0.001,0.01,0.1,1,10,100,1000,10000])
MSE_lasso = np.array([])


# In[18]:


for i in features_categorical:
    file_df = onehot_encoder(file_df, i)
    
train, test = train_test_split(file_df, test_size=0.2, random_state=20220303)
train = train.reset_index(drop = True)
test = test.reset_index(drop = True)

def kfold_lasso(data, target, n, lamb):
    # We need a vector to record mse from k-fold
    MSE = np.array([]) 
    
    # KFold is a build-in function in Scikit-learn
    #    it can help us cut data into n pieces 
    #    (compare with simple cross validation)
    kf = KFold(n_splits = n)
    
    for train_index, validation_index in kf.split(data):

        # obtain the train and validation part
        train, valid = data.loc[train_index,:], data.loc[validation_index,:]
        
        # extract X and Y to be fit in a Ridge regression model
        X_train = train.drop(target, axis = 1)
        Y_train = train[target]        
        X_valid = valid.drop(target, axis = 1)
        Y_valid = valid[target]

        # build Ridge regression model
        model = linear_model.Lasso(alpha = lamb)
        
        # fit model using training data
        model.fit(X_train,Y_train)
        
        # predict using validation data
        Y_valid_fit = model.predict(X_valid)
        
        # Calculate MSE
        MSE_temp = mean_squared_error(Y_valid_fit, Y_valid)
        
        # Add MSE to the list
        MSE = np.append(MSE, MSE_temp)
        
    return MSE.mean()

for l in lambda_list:
    MSE_lasso = np.append(MSE_lasso, kfold_lasso(train, 'price', 10, 1))


# In[19]:


MSE_lasso


# In[20]:


lambda_selected = lambda_list[MSE_lasso.argmin()]
lambda_selected


# In[21]:


model_lasso = linear_model.Lasso(alpha=lambda_selected)

X_train = train.drop(target, axis = 1)
Y_train = train[target]
model_lasso.fit(X_train, Y_train)

X_test = test.drop(target, axis = 1)
Y_test = test[target]

Y_pred = model_lasso.predict(X_test)
MSE_lasso = mean_squared_error(Y_pred, Y_test)

MSE_lasso


# In[22]:


np.sqrt(MSE_lasso)


# In[ ]:


def load_and_clean_data(csv_path: str = 'assignment3-1.csv') -> pd.DataFrame:
    """Load the dataset, replace '?' with NaN, drop missing, and convert types."""
    file = pd.read_csv(csv_path)
    file = file.replace('?', np.nan).dropna()
    type_dict = {'normalized-losses': int, 'bore': float, 'stroke': float,
                 'compression-ratio': float, 'horsepower': int, 'peak-rpm': int, 'price': int}
    file = file.astype(type_dict)
    return file

def onehot_encoder(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """One-hot encode a categorical feature."""
    if feature in df.columns:
        return pd.get_dummies(df, columns=[feature])
    else:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

def kfold_cv(data: pd.DataFrame, target: str, n: int) -> float:
    """Perform k-fold cross-validation and return mean MSE."""
    MSE = np.array([])
    kf = KFold(n_splits=n, shuffle=True, random_state=20220303)
    for train_index, validation_index in kf.split(data):
        train, valid = data.loc[train_index, :], data.loc[validation_index, :]
        X_train = train.drop(target, axis=1)
        Y_train = train[target]
        X_valid = valid.drop(target, axis=1)
        Y_valid = valid[target]
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        Y_valid_fit = model.predict(X_valid)
        MSE_temp = mean_squared_error(Y_valid_fit, Y_valid)
        MSE = np.append(MSE, MSE_temp)
    return MSE.mean()

def greedy_feature_selection(train: pd.DataFrame, features: List[str], features_categorical: List[str], target: str) -> Tuple[List[str], np.ndarray]:
    """Greedy algorithm for feature selection (up to 3 features)."""
    greedy_select = []
    MSE_greedy_algo = np.array([])
    for _ in range(min(3, len(features))):
        MSE = np.array([])
        features_left = list(set(features) - set(greedy_select))
        for new in features_left:
            features_new = greedy_select + [new]
            train_sub = train[features_new + [target]]
            categorical_sub = list(set(features_new) & set(features_categorical))
            if len(categorical_sub) != 0:
                for cat in categorical_sub:
                    train_sub = onehot_encoder(train_sub, cat)
            MSE_sub = kfold_cv(train_sub, target, 5)
            MSE = np.append(MSE, MSE_sub)
        greedy_select += [features_left[MSE.argmin()]]
        MSE_greedy_algo = np.append(MSE_greedy_algo, MSE.min())
    return greedy_select[:(MSE_greedy_algo.argmin() + 1)], MSE_greedy_algo

def fit_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, features: List[str], features_categorical: List[str], target: str) -> None:
    """Fit model on selected features and evaluate on test set."""
    train_sub = train[features + [target]]
    test_sub = test[features + [target]]
    for cat in list(set(features) & set(features_categorical)):
        train_sub = onehot_encoder(train_sub, cat)
        test_sub = onehot_encoder(test_sub, cat)
    X_train, y_train = train_sub.drop(target, axis=1), train_sub[target]
    X_test, y_test = test_sub.drop(target, axis=1), test_sub[target]
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE with selected features: {rmse:.4f}")
    print(f"Selected features: {features}")

def main():
    file = load_and_clean_data()
    features_numerical = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
                          'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                          'city-mpg', 'highway-mpg']
    features_categorical = ['symboling', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
                            'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
    features = features_categorical + features_numerical
    target = 'price'
    file_df = file[features_numerical + features_categorical + [target]]
    train, test = train_test_split(file_df, test_size=0.2, random_state=20220303)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    selected_features, mse_greedy = greedy_feature_selection(train, features, features_categorical, target)
    fit_and_evaluate(train, test, selected_features, features_categorical, target)

if __name__ == "__main__":
    main()

