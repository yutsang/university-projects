#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Model names and their corresponding MAPE values
models = [
    'ARIMA',
    'HMM (1-step)',
    'HMM (4-step)',
    'iGARCH (1-step)',
    'LSTM',
    'Transformer (10 days)',
    'XGBoost'
]

mape_values = [
    2.45,
    0.87,
    0.78,
    0.74,
    12.47,
    38.17,
    0.69
]

# Sort models and MAPE values based on model names
sorted_indices = np.argsort(models)
models_sorted = np.array(models)[sorted_indices]
mape_values_sorted = np.array(mape_values)[sorted_indices]

# Creating the vertical bar chart
plt.figure(figsize=(10, 6))
plt.bar(models_sorted, mape_values_sorted, color='skyblue')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.title('Comparison of MAPE for Various Forecasting Models')
plt.ylim(0, max(mape_values_sorted) + 10)

# Display MAPE values on the bars
for index, value in enumerate(mape_values_sorted):
    plt.text(index, value + 1, f'{value:.2f}%', ha='center')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

