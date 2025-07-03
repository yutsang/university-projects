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

# In[15]:


#install packages
#pip3 install pandas


# In[16]:


import pandas as pd
import numpy as np
from itertools import combinations
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Dict

def read_btc_data(csv_path: str = 'BTC-USD.csv') -> pd.DataFrame:
    """Read BTC price data and return DataFrame with Date as index and Close column."""
    btc_price = pd.read_csv(csv_path)
    btc_price.index = pd.to_datetime(btc_price['Date'].values, format='%Y-%m-%d')
    stock = btc_price[['Close']].copy()
    return stock

def add_return_features(stock: pd.DataFrame, predictor_variables: List[str], vardict: Dict[str, int]) -> pd.DataFrame:
    """Add return features to the stock DataFrame based on predictor_variables and vardict."""
    stock['Return'] = stock['Close'].diff() / stock['Close']
    for var in predictor_variables:
        name = 'Return_' + var
        stock[name] = stock['Return'].shift(vardict[var])
    stock = stock.dropna()
    return stock

def greedy_algo(train_valid: pd.DataFrame, target: List[str], proportion: float, features: List[str]) -> List[str]:
    """Greedy algorithm for feature selection based on trading profit."""
    greedy_select = []
    profit_greedy_algo = np.array([])
    for _ in range(len(features)):
        profit = np.array([])
        features_left = list(set(features) - set(greedy_select))
        for new in features_left:
            features_new = greedy_select + [new]
            train_valid_sub = train_valid[features_new + target]
            profit_sub = PnL(train_valid_sub, target, proportion)
            profit = np.append(profit, profit_sub)
        greedy_select += [features_left[profit.argmax()]]
        profit_greedy_algo = np.append(profit_greedy_algo, profit.max())
    return greedy_select[:(profit_greedy_algo.argmax() + 1)]

def computation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute long-short value for trading simulation."""
    for i in range(1, len(df)):
        if df.iloc[i, 1] >= 0:
            df.iloc[i, 2] = df.iloc[i-1, 2] * (1 + df.iloc[i, 0])
        else:
            df.iloc[i, 2] = df.iloc[i-1, 2] * (1 - df.iloc[i, 0])
    return df

def PnL(data: pd.DataFrame, target: List[str], proportion: float) -> float:
    """Calculate profit and loss using linear regression and trading simulation."""
    train_sub, valid_sub = train_test_split(data, test_size=proportion, random_state=0)
    X_train = train_sub.drop(target, axis=1)
    Y_train = train_sub[target]
    X_valid = valid_sub.drop(target, axis=1)
    Y_valid = valid_sub[target]
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_valid_fit = model.predict(X_valid)
    long_short_df = pd.DataFrame({
        'Return': Y_valid.iloc[:, 0].values,
        'Predicted Return': Y_valid_fit.reshape(1, -1)[0].tolist(),
        'Long-short value': np.zeros(len(valid_sub))
    }, index=valid_sub.index)
    cols = ['Return', 'Predicted Return', 'Long-short value']
    long_short_df = long_short_df[cols]
    initial = pd.DataFrame(np.array([0, 0, 1]).reshape(-1, 3), columns=long_short_df.columns)
    long_short_df = pd.concat([initial, long_short_df])
    long_short_df_final = computation(long_short_df)
    return long_short_df_final.iloc[-1, 2]

def run_simulation():
    predictor_variables = ['1D', '3D', '1W', '2W', '3W', '1M', '6W', '2M', '3M', '4M', '5M', '6M', '9M', '1Y']
    vardict = {'1D': 1, '3D': 3, '1W': 5, '2W': 10, '3W': 15, '6W': 30, '1M': 20, '2M': 40, '3M': 60, '4M': 80, '5M': 100, '6M': 120, '9M': 180, '1Y': 250}
    fix_history_length = 400
    fix_test_length = 100
    sample_size = 50
    target = ['Return']
    stock = read_btc_data()
    stock = add_return_features(stock, predictor_variables, vardict)
    features = stock.columns.values.tolist()[2:]
    random_range = len(stock) - fix_history_length - fix_test_length
    np.random.seed(42)
    random_starts = np.random.choice(random_range, sample_size, replace=False)
    returns = []
    for start in random_starts:
        train_valid = stock.iloc[start:start + fix_history_length]
        test = stock.iloc[start + fix_history_length:start + fix_history_length + fix_test_length]
        selected_features = greedy_algo(train_valid, target, 0.2, features)
        train_valid_sub = train_valid[selected_features + target]
        test_sub = test[selected_features + target]
        X_train = train_valid_sub.drop(target, axis=1)
        Y_train = train_valid_sub[target]
        X_test = test_sub.drop(target, axis=1)
        Y_test = test_sub[target]
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        Y_test_fit = model.predict(X_test)
        long_short_df = pd.DataFrame({
            'Return': Y_test.iloc[:, 0].values,
            'Predicted Return': Y_test_fit.reshape(1, -1)[0].tolist(),
            'Long-short value': np.zeros(len(test_sub))
        }, index=test_sub.index)
        cols = ['Return', 'Predicted Return', 'Long-short value']
        long_short_df = long_short_df[cols]
        initial = pd.DataFrame(np.array([0, 0, 1]).reshape(-1, 3), columns=long_short_df.columns)
        long_short_df = pd.concat([initial, long_short_df])
        long_short_df_final = computation(long_short_df)
        returns.append(long_short_df_final.iloc[-1, 2])
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Trading Returns (50 samples)')
    plt.xlabel('Final Long-Short Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    print(f"Mean return: {np.mean(returns):.4f}, Median return: {np.median(returns):.4f}")

if __name__ == "__main__":
    run_simulation()

