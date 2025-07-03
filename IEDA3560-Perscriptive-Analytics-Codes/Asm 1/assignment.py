#!/usr/bin/env python
# coding: utf-8

# # Assignment 1

# ### Problem 1
# 
# Solve the following linear system using Numpy and round the answer to 2 decimals:
# 
# 
# 3a + 7b + 5c = 1
# 
# 2 + 2b + 3c = 8
# 
# 5a + 6c = 4

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Problem 1: Linear System
# ----------------------
def solve_linear_system() -> np.ndarray:
    # Coefficient matrix
    a = np.array([[3, 7, 5], [0, 2, 3], [5, 0, 6]])
    # Constants vector
    b = np.array([1, 6, 4])
    x = np.linalg.solve(a, b)
    return np.around(x, decimals=2)

# ----------------------
# Problem 2: 4th Largest Value
# ----------------------
def fourth_largest_value(seed: int = 42) -> float:
    np.random.seed(seed)
    array = np.random.rand(20)
    array.sort()
    return array[-4]

# ----------------------
# Problem 3: DataFrame Manipulation
# ----------------------
def process_assignment_csv(csv_path: str = "assignment.csv") -> pd.DataFrame:
    assignment = pd.read_csv(csv_path, sep=',')
    # Transform 'Begin Time' and 'End Time' to datetime
    assignment['Begin Time'] = pd.to_datetime(assignment['Begin Time'])
    assignment['End Time'] = pd.to_datetime(assignment['End Time'])
    # Add 'Processing Time' column
    assignment['Processing Time'] = assignment['End Time'] - assignment['Begin Time']
    return assignment

def print_process_time_stats(df: pd.DataFrame) -> None:
    mean_time = df['Processing Time'].mean()
    median_time = df['Processing Time'].median()
    print(f"The mean of Process Time is {mean_time}")
    print(f"The median of Process Time is {median_time}")

# ----------------------
# Problem 4: Basic Line Plot
# ----------------------
def plot_basic_lines() -> None:
    x = np.arange(0.0, 1.0, 0.01)
    plt.plot(x, x, label="x^1")
    plt.plot(x, x**2, label="x^2")
    plt.plot(x, x**3, label="x^3")
    plt.plot(x, x**4, label="x^4")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.legend()
    plt.title('Basic Line Plot')
    plt.show()

# ----------------------
# Problem 5: Histogram
# ----------------------
def plot_histogram(seed: int = 19680801) -> None:
    np.random.seed(seed)
    n_bins = 10
    x = np.random.randn(1000, 3)
    colors = ['red', 'tan', 'lime']
    plt.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
    plt.legend(prop={'size': 10})
    plt.title('bars with legend')
    plt.show()

if __name__ == "__main__":
    # Problem 1
    print("Problem 1 Solution:", solve_linear_system())
    # Problem 2
    print("Problem 2 Solution: The 4th largest value is", fourth_largest_value())
    # Problem 3
    df = process_assignment_csv()
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head(5))
    print("\nSorted by Order Type:")
    print(df.sort_values(by=['Order Type']))
    print("\nRows where Order Type is 'storage':")
    print(df.loc[df["Order Type"] == 'storage'])
    print_process_time_stats(df)
    # Problem 4
    plot_basic_lines()
    # Problem 5
    plot_histogram()

