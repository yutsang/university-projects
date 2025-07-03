#!/usr/bin/env python
# coding: utf-8

# Assignment-5
# 
# Your main task for this assignment is to combine Linear Regression, Greedy Algorithm and PnL discussed in lecture to predict your trading results. You will be predicting Bitcoin price this time. You should refer to the ipynb file for this tutorial to get the structure. 
# 
# This time the prediction will be slightly different from the Lab. Every prediction should be a result from a model that is trained based on the previous 400 days. Example: if you are predicting 1-1-2019, the model should be trained using the previous 400 days, and so on.
# 
# 
# The instruction to the assignment are as follow:
# 
# 1. Read BTC price data
# 2. Only keep `close` column for analysis.
# 3. Create features that correspond to the time interval in which we want to predict
# 4. Drop all rows with missing value.
# 5. Set `fix_history_length` equal to 400, `fix_test_length` equal to 100
# 6. Use Greedy Algorithm instead of All Subset Selection
# 7. Find 50 different samples of Return using the given seed format
# 8. Use a histogram to summarize your result. Make sure to adjust the histogram settings to make it look nice
# 
# The total running time should be about 7 hours. Do not shut down your kernel during the computation.

# In[27]:


#install packages
#pip3 install pandas


# In[28]:


import pandas as pd
import numpy as np


# In[29]:


pd.options.display.max_rows, pd.options.display.max_columns = 10, 25
BTC_price_origin = pd.read_csv('BTC-USD.csv')
BTC_price_origin


# In[30]:


BTC_price_origin.index = pd.to_datetime(BTC_price_origin['Date'].values, format='%Y-%m-%d')
stock = BTC_price_origin[['Close']].copy()
print(stock)
print(stock.index)


# In[31]:


stock['Return'] = stock['Close'].diff()/stock['Close']
stock


# In[32]:


predictor_variables = ['1D','3D','1W','2W','3W','1M','6W','2M','3M','4M','5M','6M','9M','1Y']

target = ['Return']


# In[33]:


#calculate the predictor to corresponding no of days
vardict = { '1D': 1, '3D': 3,
            '1W': 5, '2W': 10, '3W': 15, '6W': 30,
            '1M': 20, '2M': 40, '3M': 60, '4M': 80, '5M': 100, '6M': 120, '9M': 180,
            '1Y': 250}

for var in predictor_variables:
    name = 'Return_' + var
    stock[name] = stock.Return.shift(vardict[var])

stock = stock.dropna()
stock


# In[34]:


#pip install itertools


# In[35]:


from itertools import combinations

def all_subsets(my_list):
    subs = []
    for i in range(1, len(my_list) + 1):
        subs += combinations(my_list, i)
    subset_List = []
    for i in subs:
        subset_List += [list(i)]
    return subset_List

features = stock.columns.values.tolist()[2:]
target = ['Return']

features_subs = all_subsets(features)


# In[36]:


fix_history_length = 400
fix_test_length = 100
sample_size = 50
randomRange = len(stock) - fix_history_length - fix_test_length
randomRange


# In[37]:


def greedy_algo(train_valid, target, proportion):
    
    # initialize a list to save features
    greedy_select = []
    
    profit_greedy_algo = np.array([])
    for i in range(len(features)):
        profit = np.array([])
        features_left = list(set(features) - set(greedy_select))

        for new in features_left:
            features_new = greedy_select + [new]
            train_valid_sub = train_valid[features_new + target]

            # CrossValidation, compute the profit and save it into profit_sub
            profit_sub = PnL(train_valid_sub, target, proportion)
            profit = np.append(profit, profit_sub)

        # pick the features that gives the largest profit
        # and add it into our features list
        # meanwhile, save the corresponding profit
        greedy_select += [features_left[profit.argmax()]]
        profit_greedy_algo = np.append(profit_greedy_algo, profit.max())
        
    return greedy_select[:(profit_greedy_algo.argmax()+1)]


# In[38]:


from sklearn.model_selection import train_test_split
def computation(df):
    for i in range(1, len(df)):
        if df.iloc[i, 1] >= 0:
            df.iloc[i, 2] = df.iloc[i-1, 2] * (1 + df.iloc[i, 0])
        else:
            df.iloc[i, 2] = df.iloc[i-1, 2] * (1 - df.iloc[i, 0])
    return df

def PnL(data, target, proportion):
    train_sub, valid_sub = train_test_split(data, test_size = proportion, random_state = 0)

    # create a linear model
    X_train = train_sub.drop(target, axis = 1)
    Y_train = train_sub[target]
    X_valid = valid_sub.drop(target, axis = 1)
    Y_valid = valid_sub[target]

    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_valid_fit = model.predict(X_valid)

    # a data frame for computing and saving long_short value
    long_short_df = pd.DataFrame({'Return': Y_valid.iloc[:,0].values,
                                  'Predicted Return': Y_valid_fit.reshape(1,-1)[0].tolist(),
                                  'Long-short value': np.zeros(len(valid_sub))},
                                 index = valid_sub.index)

    cols = ['Return', 'Predicted Return', 'Long-short value']
    long_short_df = long_short_df[cols]

    # give an initial point
    initial = pd.DataFrame(np.array([0,0,1]).reshape(-1,3),
                       columns = long_short_df.columns)

    # combine df and initial point
    long_short_df = pd.concat([initial, long_short_df]) 
    
    # compute long_short value
    long_short_df_final = computation(long_short_df)
    
    # return final long_short value of this period
    return long_short_df_final.iloc[-1,2]


# In[39]:


get_ipython().run_cell_magic('time', '', "from sklearn import linear_model\nprofit_final = np.array([])\n\n# Think about why we need two loops here:\n# The answer is that this time, we need to change the training data for each day, each sample\n# In total, we have 50 samples\nfor j in range(50):\n    \n    np.random.seed(j)\n    #print('j=', j)\n    #return_max = 0\n    random_Num = np.random.randint(randomRange)\n    BeginTime = random_Num\n    timestamp = fix_history_length + random_Num\n    EndTime = timestamp + fix_test_length\n    y_pred = []\n\n    for i in range(fix_test_length):  \n    #for i in range(10): \n\n        train_valid = stock.iloc[BeginTime+i:timestamp+i, : ]\n        test = stock.iloc[timestamp:EndTime, : ]\n\n        feature_update = greedy_algo(train_valid, target, 0.2)\n        #print('i=', i)\n        #print(feature_update)\n\n        X_cv = train_valid[feature_update]\n        Y_cv = train_valid[target]\n\n        X_test_cv = test[feature_update]\n        Y_test_cv = test[target]\n\n        model_cv = linear_model.LinearRegression()\n        model_cv.fit(X_cv, Y_cv)\n\n        Y_test_cv_fit = model_cv.predict(X_test_cv)\n        y_pred = np.append(y_pred, Y_test_cv_fit[0])\n\n    long_short_df = pd.DataFrame({'Return': Y_test_cv.values.reshape(1,-1)[0].tolist(),\n                                    'Predicted Return': y_pred.reshape(1,-1)[0].tolist(),\n                                    'Long-short value': np.zeros(len(Y_test_cv))},\n                                    index = Y_test_cv.index)\n\n    cols = ['Return', 'Predicted Return', 'Long-short value']\n    long_short_df = long_short_df[cols]\n\n    initial = pd.DataFrame(np.array([0,0,1]).reshape(-1,3),\n                               columns = long_short_df.columns)\n        \n    long_short_df = pd.concat([initial, long_short_df])\n    long_short_df_final = computation(long_short_df)\n        \n    #if (long_short_df_final.iloc[-1,2] > return_max):\n    #        return_max = long_short_df_final.iloc[-1,2]\n        #feature_update\n        #print(return_max)\n        #print(long_short_df)\n        #print('long_short_df_final: ', long_short_df_final)\n    \n    profit_final = np.append(profit_final,long_short_df_final.iloc[-1,2])\n    #profit_final = np.append(profit_final,return_max)\n\n#print(profit_final)\n#print('return_max: ', return_max)\n")


# In[40]:


profit_final


# In[47]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,5))
bins = np.linspace(0.3, 2.4, 15)
a = plt.hist(profit_final, bins, histtype='bar', rwidth=0.8)
for i in range(len(bins)-1):
    plt.text(a[1][i]+0.037,a[0][i]+0.2,int(a[0][i]))
plt.title("histogram of 50 Returns for cryptocurrency--'Bitcoin'")

# you should change the coordinates of the plot to make it look nice
plt.text(0.3, 5, ("Avg Return: {0:.2f}%".format((profit_final.mean()-1) * 100)))
plt.text(0.3, 5.3, ("Std dev: {0:.2f}%".format(profit_final.std() * 100)))
plt.text(0.3, 5.6, "320 training points")
plt.text(0.3, 5.9, "80 validation points")
plt.text(0.3, 6.2, "100 test points")

