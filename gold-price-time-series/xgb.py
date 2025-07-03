import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load and clean the dataset
file_path = 'gold_dec24(GC=F)_1wk.csv'
df = pd.read_csv(file_path, skiprows=3, names=['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], header=None)

# Parse dates and set as index
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)

# Handle missing values in the 'Close' column
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')

# Create sequences for supervised learning
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 4):  # Ensure enough data for y
        X.append(data[i:i + seq_length])  # Input sequence of length seq_length
        y.append(data[i + seq_length:i + seq_length + 4])  # Next 4 steps as output
    return np.array(X), np.array(y)

seq_length = 12  # Use 12 weeks of data to predict the next values
data = df['Close'].values
X, y = create_sequences(data, seq_length)

# Reshape data for XGBoost (XGBoost expects 2D input)
X_reshaped = X.reshape(X.shape[0], -1)  # Flatten each sequence

# Define the model with best parameters found
best_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.6,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=5,
    n_estimators=100,
    subsample=0.6
)

# Fit the model on all available data
best_model.fit(X_reshaped, y)

# Generate predictions on validation set (last sequence)
val_preds = best_model.predict(X_reshaped[-4:]).flatten()  # Predicting last sequence

# Calculate MAPE for evaluation (using last actual values)
actuals = y[-4:].flatten()  # Last actual values for comparison

mape_final = mean_absolute_percentage_error(actuals, val_preds) * 100

print(f"Final Evaluation MAPE (Last-4 Points): {mape_final:.2f}%")

# Plotting results with moving averages and predictions highlighted
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_12'] = df['Close'].rolling(window=12).mean()

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Full Dataset', color='green')
plt.plot(df.index[-len(val_preds):], actuals, label='Actual Prices', color='blue')
plt.plot(df.index[-len(val_preds):], val_preds, label='Predicted Prices (XGBoost)', color='orange', linestyle='--')  # Indicate XGBoost here
plt.title('Gold Price Trend with Predictions by XGBoost')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
