#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import joblib

warnings.filterwarnings('ignore')

raw_merge = pd.read_csv("raw_merge_osm_weather_hk_all.csv", index_col = 0)

#for result recording
columns = ['Method', 'MAE', 'MSE', 'R-squared', 'MAPE']
result_df = pd.DataFrame(columns=columns)


# In[2]:


import matplotlib.pyplot as plt

# convert planStartTime to datetime format
raw_merge['planStartTime'] = pd.to_datetime(raw_merge['planStartTime'])

# calculate Difference and Reference
intermediate = raw_merge.assign(
    Difference=raw_merge["ActDurn-nonNull"] / raw_merge["Time"]
).query("0.3 <= Difference <= 10") \
 .assign(
    Weekday=lambda x: x["planStartTime"].dt.dayofweek,
    NoOfLocation=lambda x: x["Route"].apply(lambda x: len(x.split(','))/2)
).drop(["index"], axis=1).reset_index(drop=True)

# Set Pandas options to show all columns
pd.set_option('display.max_columns', None)
#intermediate = pd.read_csv('intermediate.csv', index_col=0)
#df[column] = pd.to_datetime(df[column], errors = 'coerce', format='%Y-%m-%d %H:%M:%S')

intermediate['planStartDateTime'] = pd.to_datetime(intermediate['planStartTime'], 
            errors = 'coerce', format='%Y-%m-%d %H:%M:%S')
intermediate['planStartYear'] = intermediate['planStartDateTime'].dt.year
intermediate['planStartMonth'] = intermediate['planStartDateTime'].dt.month
intermediate['planStartDay'] = intermediate['planStartDateTime'].dt.day
intermediate['planStartHour'] = intermediate['planStartDateTime'].dt.hour

intermediate['actStartDateTime'] = pd.to_datetime(intermediate['actStartTime'], 
            errors = 'coerce', format='%Y-%m-%d %H:%M:%S')
intermediate['actStartYear'] = intermediate['actStartDateTime'].dt.year
intermediate['actStartMonth'] = intermediate['actStartDateTime'].dt.month
intermediate['actStartDay'] = intermediate['actStartDateTime'].dt.day
intermediate['actStartHour'] = intermediate['actStartDateTime'].dt.hour

from sklearn.neighbors import LocalOutlierFactor

#Outlier Detection Q1 - 1.5*IQR to Q3 + 1.5*IQR

shape_before = intermediate.shape

Method = 'IQR'

# Define the column(s) to remove outliers from
columns_to_clean = ['Difference']

if Method == 'IQR':

    # Define the threshold for outlier detection (1.5 is a common value)
    threshold = 1

    # Loop over the columns and remove outliers using the IQR method
    for col in columns_to_clean:
        Q1 = intermediate[col].quantile(0.25)
        Q3 = intermediate[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold*IQR
        upper_bound = Q3 + threshold*IQR
        intermediate = intermediate[(intermediate[col] >= lower_bound) & (intermediate[col] <= upper_bound)]

elif Method == 'ModifiedZ':
    
    # Define the threshold for outlier detection
    threshold = 3.5

    # Loop over the columns and remove outliers using the modified Z-score method
    for col in columns_to_clean:
        median = intermediate[col].median()
        mad = intermediate[col].mad()
        modified_z = 0.6745 * (intermediate[col] - median) / mad
        intermediate = intermediate[abs(modified_z) <= threshold]

elif Method == 'LOF':
    # Define the number of neighbors for the LOF algorithm
    n_neighbors = 20

    # Define the contamination parameter for outlier detection
    contamination = 0.01

    # Create an instance of the LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    # Loop over the columns and remove outliers using the LOF method
    for col in columns_to_clean:
        # Fit the LOF algorithm to the column
        lof.fit(intermediate[[col]])
    
        # Predict the outlier scores for the column
        outlier_scores = lof.negative_outlier_factor_
    
        # Identify the outliers using the outlier scores
        outliers = np.where(outlier_scores < -2)[0]
    
        # Remove the outliers from the intermediate dataframe
        intermediate = intermediate.drop(index=intermediate.index[outliers])

print("Shape before and after:", shape_before, intermediate.shape)
intermediate.head()


# In[75]:


#Machine Learning Models

#RandomForest

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def train_rf(df, target_variable, append_results=False, print_result=False, have_model=False, col_name=None, return_mape=False):
    """
    Trains a random forest regressor model with TruncatedSVD on the input dataframe.
    Returns the evaluation metrics in a dictionary.
    """
    if have_model:
        print_result = True
    
    # Define variables to drop from the dataframe
    drop_variables = [col for col in df.columns if col.startswith('pred_')]
    system_variables = ['MiscTime', 'planDurn', 'actDurn', 'Route', 'Maps', 'planStartTime', 'actStartTime', 'Difference']
    
    # Append target_variable to drop_variables
    drop_variables.append(target_variable) 
    drop_variables = drop_variables + system_variables
    
    X_train_processed = None
    X_test_processed = None

    if not have_model:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.drop(drop_variables, axis=1), df[target_variable], test_size=0.2, random_state=42)

        # Preprocess the data using ColumnTransformer
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Define the hyperparameter distribution for RandomizedSearchCV
        param_distribution = {
            'n_estimators': randint(50, 200),
            'max_depth': [5, 10, 20, None],
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'max_features': uniform(0.1, 0.9)
        }

        # Create a random forest regressor estimator
        rf = RandomForestRegressor(random_state=42)

        # Create a randomized search object with 5-fold cross-validation and 100 iterations
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distribution,
            cv=5,
            n_iter=50,  # Reduce the number of iterations to 50
            n_jobs=-1,
            scoring='neg_mean_absolute_error',
            random_state=42
        )
        # Fit the randomized search object to the training data
        random_search.fit(X_train_processed, y_train)

        # Train a random forest model with the best hyperparameters and the reduced data
        rf_best = RandomForestRegressor(
            n_estimators=random_search.best_params_['n_estimators'],
            max_depth=random_search.best_params_['max_depth'],
            min_samples_split=random_search.best_params_['min_samples_split'],
            min_samples_leaf=random_search.best_params_['min_samples_leaf'],
            max_features=random_search.best_params_['max_features'],
            random_state=42
        )
        rf_best.fit(X_train_processed, y_train)

        # Save the preprocessor and model together
        joblib.dump((preprocessor, rf_best), 'preprocessor_and_model.joblib')

        # Check the number of features in the preprocessed new data
        #print('Number of features in preprocessed new data - Train:', X_test_processed.shape[1])
    else:
        preprocessor, rf_best = joblib.load('preprocessor_and_model.joblib')
        X_test, y_test = df.drop(drop_variables, axis=1), df[target_variable]
        X_test_processed = preprocessor.transform(X_test)
        #print('Number of features in preprocessed new data - Test:', X_test_processed.shape[1])

    # Make predictions on the test data
    y_pred = rf_best.predict(X_test_processed)
    if print_result: print(y_pred)

    # Evaluate the predictions using different metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Print the results
    #if print_result:
        #print(f'Mean Absolute Error: {mae:.2f}')
        #print(f'Mean Squared Error: {mse:.2f}')
        #print(f'Coefficient of Determination (R^2): {r2:.2f}')

    # Append the metrics to the results dataframe
    if append_results and col_name is not None:
        df[col_name] = pd.DataFrame(y_pred, columns=[col_name])
        if return_mape:
            return df, mape
        return df
    else:
        return {'Method': 'RandomForest', 'MAE': mae, 'MSE': mse, 'R-squared': r2, 'MAPE':mape}


# In[77]:


#random forest

from tqdm import tqdm  # Import the tqdm library for progress visualization

def train_and_predict(df, target_variable, min_rows=100):
    predictions_df = pd.DataFrame(columns=['Date', 'Prediction', 'Performance_Ratio'])

    for date in df['planStartTime'].unique():
        train_data = df[df['planStartTime'] < date]

        if len(train_data) >= min_rows:
            print(date)
            model_results, model_mape = train_rf(train_data, target_variable, append_results=True, col_name='pred_result', return_mape=True)
            current_rows_results, current_rows_mape = train_rf(df[df['planStartTime'] == date], target_variable, print_result=True, have_model=True, append_results=True, col_name='pred_result', return_mape=True)

            performance_ratio = current_rows_mape / model_mape

            for _, row in current_rows_results.iterrows():
                predictions_df = predictions_df.append({'Date': date,
                                                        'Prediction': row['pred_result'],
                                                        'Performance_Ratio': performance_ratio},
                                                       ignore_index=True)
        else:
            predictions_df = predictions_df.append({'Date': date, 'Prediction': 0, 'Performance_Ratio': 0}, ignore_index=True)

    return predictions_df


predictions_df = train_and_predict(df=intermediate, target_variable='ActDurn-nonNull')
print(predictions_df)


# In[3]:


import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

def train_xgb(df, target_variable, print_result=False, have_model=False, col_name=None, append_results=False, return_mape=False):
    """
    Trains an XGBoost regressor model with TruncatedSVD on the input dataframe.
    Returns the evaluation metrics in a dictionary.
    """
    
    # Define variables to drop from the dataframe
    drop_variables = [col for col in df.columns if col.startswith('pred_')]
    system_variables = ['MiscTime', 'planDurn', 'actDurn', 'Route', 'Maps', 'planStartTime', 'actStartTime', 'Difference']
    
    # Append target_variable to drop_variables
    drop_variables.append(target_variable) 
    drop_variables = drop_variables + system_variables
    
    X_train_processed = None
    X_test_processed = None

    if not have_model:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.drop(drop_variables, axis=1), df[target_variable], test_size=0.2, random_state=42)

        # Preprocess the data using ColumnTransformer
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Define the hyperparameters for XGBRegressor
        params = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist'  # Use CUDA for training
        }    

        # Create an XGBRegressor estimator
        xgb_reg = xgb.XGBRegressor(**params)
        #print(f"XGBoost Model: {xgb_reg}")

        # Use cross-validation to fit the XGBRegressor to the training data
        scores = cross_val_score(xgb_reg, X_train_processed, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        #print(f"Cross-validation scores: {scores}")
        #print(f"Mean MAE: {np.mean(scores):.2f} (std: {np.std(scores):.2f})")
        xgb_reg.fit(X_train_processed, y_train)
        y_pred = xgb_reg.predict(X_test_processed)

        # Save the preprocessor and model together
        joblib.dump((preprocessor, xgb_reg), 'preprocessor_and_model.joblib')
        #print("Saved preprocessor and model.")

        # Check the number of features in the preprocessed new data
        #print('Number of features in preprocessed new data - Train:', X_test_processed.shape[1])
    else:
        preprocessor, xgb_reg = joblib.load('preprocessor_and_model.joblib')
        X_test, y_test = df.drop(drop_variables, axis=1), df[target_variable]
        X_test_processed = preprocessor.transform(X_test)
        #print("X_test_processed:\n", X_test_processed)
        y_pred = xgb_reg.predict(X_test_processed)
        if print_result: print(y_pred)
        #print(f"Predictions(xgb): {y_pred}")

        # Print feature importances
        feature_importances = xgb_reg.feature_importances_
        #print("Feature importances:\n", feature_importances)

    # Evaluate the predictions using different metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Print the results
    #if print_result:
        #print(f'Mean Absolute Error: {mae:.2f}')
        #print(f'Mean Squared Error: {mse:.2f}')
        #print(f'Coefficient of Determination (R^2): {r2:.2f}')
        #print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    # Append the metrics to the results dataframe
    if append_results and col_name is not None:
        df[col_name] = pd.DataFrame(y_pred, columns=[col_name])
        return df, y_test
    else:
        return y_pred, y_test



# In[5]:


#xgboost
import pandas as pd
import numpy as np
from tqdm import tqdm

def train_and_predict(df, target_variable, min_rows=100):
    predictions_df = pd.DataFrame(columns=['Date', 'Prediction', 'Performance_Ratio'])

    for date in df['planStartTime'].unique():
        train_data = df[df['planStartTime'] < date]

        if len(train_data) >= min_rows:
            print(date)
            model_results, model_actual = train_xgb(train_data, target_variable, append_results=True, col_name='pred_result')
            model_mape = np.mean(np.abs((model_actual - model_results['pred_result']) / model_actual)) * 100
            current_rows_results, current_rows_actual = train_xgb(df[df['planStartTime'] == date], target_variable, have_model=True, append_results=True, col_name='pred_result', print_result=True)
            current_rows_mape = np.mean(np.abs((current_rows_actual - current_rows_results['pred_result']) / current_rows_actual)) * 100

            performance_ratio = current_rows_mape / model_mape

            for _, row in current_rows_results.iterrows():
                predictions_df = predictions_df.append({'Date': date,
                                                        'Prediction': row['pred_result'],
                                                        'Performance_Ratio': performance_ratio},
                                                       ignore_index=True)
        else:
            predictions_df = predictions_df.append({'Date': date, 'Prediction': 0, 'Performance_Ratio': 0}, ignore_index=True)

    return predictions_df

predictions_df = train_and_predict(df=intermediate, target_variable='ActDurn-nonNull')
print(predictions_df)


# In[19]:


import category_encoders as ce

def train_nn(df, target_variable, append_results=False, print_result=False, have_model=False, col_name='pred_result'):
    # Import the category_encoders package

    if have_model: print_result = True

    drop_variables = [col for col in intermediate.columns if col.startswith('pred_')]
    system_variables = ['planStartTime', 'Difference']

    # Append target_variable to drop_variables
    drop_variables.append(target_variable) 
    drop_variables = drop_variables + system_variables
    
    # Encode the categorical columns using binary encoding
    for col in df.columns:
        if df[col].dtype == 'object':
            encoder = ce.BinaryEncoder(cols=[col])
            df = encoder.fit_transform(df)

    # Convert datetime64 columns to string data type
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    for date_col in date_cols:
        df[date_col] = pd.to_datetime(df[date_col])
        df[date_col] = df[date_col].astype('int64')

    # Split the data into training and testing sets
    X = df.drop(drop_variables, axis=1)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if have_model == False:

        # Train the neural network
        mlp = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        # Save the model to a file
        joblib.dump((scaler, mlp), 'nn_model.joblib')

        # Make predictions on the test set
        y_pred = mlp.predict(X_test_scaled)
    
    else:
        scaler, mlp = joblib.load('nn_model.joblib')
        X_test, y_test = df.drop(drop_variables, axis=1), df[target_variable]
        X_test_scaled = scaler.transform(X_test)
        #print("X_test_processed:\n", X_test_processed)
        y_pred = mlp.predict(X_test_scaled)
        if print_result: print(y_pred)
        #print(f"Predictions(xgb): {y_pred}")

    # Append the prediction results to the DataFrame
    pred_df = pd.DataFrame({'pred_nn': y_pred}, index=X_test.index)
    #df['pred_nn'] = pred_df['pred_nn']

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE
    mape = (abs((y_test - y_pred) / y_test)).mean() * 100

        # Append the metrics to the results dataframe
    if append_results and col_name is not None:
        df[col_name] = pd.DataFrame(y_pred, columns=[col_name])
        return df, y_test
    else:
        return y_pred, y_test



# In[20]:


#nn
import pandas as pd
import numpy as np
from tqdm import tqdm

def train_and_predict(df, target_variable, min_rows=100):
    predictions_df = pd.DataFrame(columns=['Date', 'Prediction', 'Performance_Ratio'])

    for date in df['planStartTime'].unique():
        train_data = df[df['planStartTime'] < date]

        if len(train_data) >= min_rows:
            print(date)
            model_results, model_actual = train_nn(train_data, target_variable, append_results=True, col_name='pred_result')
            model_mape = np.mean(np.abs((model_actual - model_results['pred_result']) / model_actual)) * 100
            current_rows_results, current_rows_actual = train_nn(df[df['planStartTime'] == date], target_variable, have_model=True, append_results=True, col_name='pred_result', print_result=True)
            current_rows_mape = np.mean(np.abs((current_rows_actual - current_rows_results['pred_result']) / current_rows_actual)) * 100

            performance_ratio = current_rows_mape / model_mape

            for _, row in current_rows_results.iterrows():
                predictions_df = predictions_df.append({'Date': date,
                                                        'Prediction': row['pred_result'],
                                                        'Performance_Ratio': performance_ratio},
                                                       ignore_index=True)
        else:
            predictions_df = predictions_df.append({'Date': date, 'Prediction': 0, 'Performance_Ratio': 0}, ignore_index=True)

    return predictions_df

predictions_df = train_and_predict(df=intermediate, target_variable='ActDurn-nonNull')
print(predictions_df)


# In[ ]:


predictions_df.to_csv('xg_result_v2.csv')


# In[18]:


#=========================
from scipy import stats
import pandas as pd

z_value = pd.read_csv('intermediate_all.csv')

shape = z_value.shape[0]
print(shape)

# Filter the data using actual values
actual_count = ((z_value['Difference'] >= 0.3) & (z_value['Difference'] <= 10)).sum()

# Calculate the z-score for the Difference column
mean = z_value['Difference'].mean()
std_dev = z_value['Difference'].std()
z_value['z_score'] = (z_value['Difference'] - mean) / std_dev

# Filter the data using z-score
z_score_count = ((z_value['z_score'] >= 0.3 / std_dev) & (z_value['z_score'] <= 10 / std_dev)).sum()

print("Number of data points within the range using actual values:", actual_count, round(actual_count/shape*100, 2))
print("Number of data points within the range using z-score method:", z_score_count, round((z_score_count/shape)*100, 2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




