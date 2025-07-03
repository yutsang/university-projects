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

# In[ ]:


import ...


# In[ ]:


BTC_price_origin = ...


# In[ ]:





# In[ ]:


predictor_variables = ['1D','3D','1W','2W','3W','1M','6W','2M','3M','4M','5M','6M','9M','1Y']

target = ['Return']


# In[ ]:


fix_history_length = ...
fix_test_length = ...
sample_size = 50
randomRange = ...


# In[ ]:


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


# In[ ]:


profit_final = np.array([])

# Think about why we need two loops here:
# The answer is that this time, we need to change the training data for each day, each sample
# In total, we have 50 samples
for j in range(sample_size):
    
    np.random.seed(j)

    for i in range(fix_test_length):  
        
        ...

        feature_update = greedy_algo(...)

        ...
        
    profit_final = np.append(profit_final, ...)


# In[ ]:


fig = plt.figure(figsize=(8,5))
bins = np.linspace(0.3, 1.7, 15)
a = plt.hist(profit_final, bins, histtype='bar', rwidth=0.9)
for i in range(len(bins)-1):
    plt.text(a[1][i]+0.037,a[0][i]+0.2,int(a[0][i]))
plt.title("histogram of 50 Returns for cryptocurrency--'Bitcoin'")

# you should change the coordinates of the plot to make it look nice
plt.text(0.3, 18, ("Avg Return: {0:.2f}%".format((profit_final.mean()-1) * 100)))
plt.text(0.3, 17, ("Std dev: {0:.2f}%".format(profit_final.std() * 100)))
plt.text(0.3, 16, "320 training points")
plt.text(0.3, 15, "80 validation points")
plt.text(0.3, 14, "100 test points")

