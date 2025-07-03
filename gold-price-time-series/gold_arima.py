import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Assuming df is already loaded and preprocessed with 'Close' column available

# Check stationarity using ADF test
result = adfuller(df['Close'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] > 0.05:
    print("Time series is not stationary. Differencing is required.")
    df['Close_diff'] = df['Close'].diff().dropna()
else:
    print("Time series is stationary.")

# Split into train/test sets (80% train, 20% test)
train_size = int(len(df) * 0.8)
train_data, test_data = df['Close'][:train_size], df['Close'][train_size:]

# Fit ARIMA model with selected parameters (use auto_arima for optimal parameters)
auto_model = auto_arima(train_data, seasonal=False, trace=True,
                        error_action='ignore', suppress_warnings=True,
                        stepwise=True)
optimal_order = auto_model.order  # Get best (p,d,q)
model = ARIMA(train_data, order=optimal_order)
arima_model = model.fit()

# Multi-step forecasting (4 steps)
forecast_steps = 4
forecast = arima_model.forecast(steps=forecast_steps)

# Evaluate predictions on test data (first 4 points)
mse = mean_squared_error(test_data[:forecast_steps].values, forecast.values)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# Plot actual vs predicted values for 4 steps
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data.values, label='Actual Prices', color='blue')
plt.plot(test_data.index[:forecast_steps], forecast.values, label='Predicted Prices (4 Steps)', color='orange', linestyle='--')
plt.title('Gold Prices: Actual vs Predicted (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()