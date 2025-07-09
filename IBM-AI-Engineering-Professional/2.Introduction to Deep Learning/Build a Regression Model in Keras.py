#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width = 400> </a>
# 
# <h1 align=center><font size = 5>Final Project : Regression Model in Keras</font></h1>
# 

# ## Guide to reviewer
# 

# This notebook contains 4 parts A, B, C, and D which contain codes corresponding to the parts asked.
# 

# <h2>Regression Models with Keras</h2>
# 
# I'll be working upon the concrete data mentioned in the problem.

# Let's start by importing the <em>pandas</em> and the Numpy libraries.
# 

# In[1]:


import pandas as pd
import numpy as np


# <strong>The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:</strong>
# 
# <strong>1. Cement</strong>
# 
# <strong>2. Blast Furnace Slag</strong>
# 
# <strong>3. Fly Ash</strong>
# 
# <strong>4. Water</strong>
# 
# <strong>5. Superplasticizer</strong>
# 
# <strong>6. Coarse Aggregate</strong>
# 
# <strong>7. Fine Aggregate</strong>
# 

# Let's download the data and read it into a <em>pandas</em> dataframe.
# 

# In[2]:


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()


# So the first concrete sample has 540 cubic meter of cement, 0 cubic meter of blast furnace slag, 0 cubic meter of fly ash, 162 cubic meter of water, 2.5 cubic meter of superplaticizer, 1040 cubic meter of coarse aggregate, 676 cubic meter of fine aggregate. Such a concrete mix which is 28 days old, has a compressive strength of 79.99 MPa. 
# 

# #### Let's check how many data points we have.
# 

# In[3]:


concrete_data.shape


# So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.
# 

# Let's check the dataset for any missing values.
# 

# In[4]:


concrete_data.describe()


# In[5]:


concrete_data.isnull().sum()


# The data looks very clean and is ready to be used to build our model.
# 

# # PART : A
# 
# ### 1. Building baseline model :
#     1.1. 1 hidden layer with 10 nodes
#     1.2. ReLU
#     1.3. Adam
#     1.4. MSE
# 
# ### 2. Train Test Split.
# ### 3. Training in 50 epochs.
# ### 4. Evaluatin on test data, finding MSE
# ### 5. Repeating 2-4 50 times and creating list of 50 MSEs.
# ### 6. Finding meand and SD of MSEs.

# In[10]:


import keras
from keras.models import Sequential
from keras.layers import Dense

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[11]:


from sklearn.model_selection import train_test_split
# splitting data
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2)


# In[12]:


# predictors
predictors.head()


# In[13]:


# target
target.head()


# In[14]:


# number of predictor columns
n_cols = predictors.shape[1] # number of predictors


# In[15]:


# Buidling, Fitting and Evaluating model
from sklearn.metrics import mean_squared_error
MSEs = []

# build the model
model = regression_model()

for index in range(50) :
    
    # fit the model
    res = model.fit(X_train, y_train, epochs=50, verbose=0)
    
    # predicting and calculating MSE
    y_pred = model.predict(X_test)
    MSEs.append(mean_squared_error(y_test, y_pred))

print(MSEs)    


# In[16]:


# finding mean and standard deviation of MSEs.
# importing statistics
import statistics

mean_mse = statistics.mean(MSEs)
stdev = statistics.stdev(MSEs)

print('Mean : %.2f' % mean_mse)
print('Standard Deviation : %.2f' % stdev)


#  

# # PART : B
# 
# ## Part A with Normalized data

# In[17]:


get_ipython().run_cell_magic('time', '', '# Normalizing data\npredictors_norm = (predictors - predictors.mean()) / predictors.std()\nn_cols = predictors_norm.shape[1] # number of predictors\n\n# train_test_split\nX_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size = 0.2)\n\n# Buidling, Fitting and Evaluating model\nfrom sklearn.metrics import mean_squared_error\nMSEsB = []\n\n# build the model\nmodelB = regression_model()\n\nfor index in range(50) :\n    \n    # fit the model\n    modelB.fit(X_train, y_train, epochs=50, verbose=0)\n    \n    # predicting and calculating MSE\n    y_pred = modelB.predict(X_test)\n    MSEsB.append(mean_squared_error(y_test, y_pred))\n\n    \nprint(MSEsB)\n')


# In[18]:


# finding mean and standard deviation of MSEs.
# importing statistics
import statistics

mean_mseB = statistics.mean(MSEsB)
stdevB = statistics.stdev(MSEsB)

print('Mean : %.2f' % mean_mseB)
print('Standard Deviation : %.2f' % stdevB)


# In[19]:


# Difference of means of A and B model
print('Difference of means of Model A and Model B is {}', abs(mean_mseB - mean_mse))


# # Part : C
# 
# ### Part B with 100 epochs

# In[20]:


get_ipython().run_cell_magic('time', '', '# Normalizing data\npredictors_norm = (predictors - predictors.mean()) / predictors.std()\nn_cols = predictors_norm.shape[1] # number of predictors\n\n# train_test_split\nX_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size = 0.2)\n\n# Buidling, Fitting and Evaluating model\nfrom sklearn.metrics import mean_squared_error\nMSEsC = []\n\n# build the model\nmodelC = regression_model()\n\nfor index in range(50) :\n    \n    # fit the model\n    modelC.fit(X_train, y_train, epochs=100, verbose=0)\n    \n    # predicting and calculating MSE\n    y_pred = modelC.predict(X_test)\n    MSEsC.append(mean_squared_error(y_test, y_pred))\n\n    \nprint(MSEsC)\n')


# In[21]:


# finding mean and standard deviation of MSEs.
# importing statistics
import statistics

mean_mseC = statistics.mean(MSEsC)
stdevC = statistics.stdev(MSEsC)

print('Mean : %.2f' % mean_mseC)
print('Standard Deviation : %.2f' % stdevC)

# Difference of means of A and B model
print('Difference of means of Model B and Model C is {}', abs(mean_mseB - mean_mseC))


# # Part : D
# ### Part D is Part B with Three hidden layers, each of 10 nodes and ReLU activation function.

# In[22]:


# define regression model
def regression_model_D():
    # create model
    modelD = Sequential()
    modelD.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    modelD.add(Dense(10, activation='relu')) # HL1
    modelD.add(Dense(10, activation='relu')) # HL2
    modelD.add(Dense(10, activation='relu')) # HL3
    modelD.add(Dense(1))
    
    # compile model
    modelD.compile(optimizer='adam', loss='mean_squared_error')
    return modelD


# In[23]:


get_ipython().run_cell_magic('time', '', '# Normalizing data\npredictors_norm = (predictors - predictors.mean()) / predictors.std()\nn_cols = predictors_norm.shape[1] # number of predictors\n\n# train_test_split\nX_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size = 0.2)\n\n# Buidling, Fitting and Evaluating model\nfrom sklearn.metrics import mean_squared_error\nMSEsD = []\n\n# build the model\nmodelD = regression_model_D()\n\nfor index in range(50) :\n    \n    # fit the model\n    modelD.fit(X_train, y_train, epochs=50, verbose=0)\n    \n    # predicting and calculating MSE\n    y_pred = modelD.predict(X_test)\n    MSEsD.append(mean_squared_error(y_test, y_pred))\n\n    \nprint(MSEsD)\n')


# In[24]:


# finding mean and standard deviation of MSEs.
# importing statistics
import statistics

mean_mseD = statistics.mean(MSEsD)
stdevD = statistics.stdev(MSEsD)

print('Mean : %.2f' % mean_mseD)
print('Standard Deviation : %.2f' % stdevD)

# Difference of means of B and D model
print('Difference of means of Model B and Model D is {}', abs(mean_mseB - mean_mseD))


# # End of Notebook
