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
a = np.array([[3, 7, 5], [0, 2, 3], [5, 0, 6]])
b = np.array([1, 6, 4])
x = np.linalg.solve(a, b)
np.around(x, decimals=2)


# ### Problem 2
# 
# Find the 4th largest value in the following array:

# In[10]:


import numpy as np

np.random.seed(42)
array = np.random.rand(20)
print(array)


# *Hint: look for functions to sort arrays in Numpy documentation!*

# In[13]:


array.sort()
#print(array)
print ('The 4th largest value in the array is ' + str(array[-4]))


# ### Problem 3
# 
# Manipulate the Pandas dataframe following the instructions.

# In[1]:


import pandas as pd


# Read a `csv` file which is called `assignment.csv`.

# In[2]:


assignment = pd.read_csv(r"assignment.csv", sep=',')


# View the first 5 rows in this dataframe.

# In[3]:


assignment.head(5)


# Sort the full dataframe by `Order Type`.

# In[4]:


assignment.sort_values(by=['Order Type'])


# Select all rows whose `Order Type` is `storage`.

# In[5]:


assignment.loc[assignment["Order Type"] == 'storage']


# Transform `Begin Time` and `End Time` to date format.

# In[6]:


assignment['Begin Time'] = pd.to_datetime(assignment['Begin Time'] )


# In[7]:


assignment['End Time'] = pd.to_datetime(assignment['End Time'] )


# Group the dataframe by `Order ID` and  `Order Type`, and count the size.

# In[8]:


assignment.groupby(['Order ID'] and ['Order Type']).count()


# Use `Processing Time`  =  `End Time` - `Begin Time`,  add a new column called `Process Time`.

# In[9]:


assignment['Processing Time'] = assignment['End Time'] - assignment['Begin Time']


# Compute the mean and median of `Process Time`.

# In[10]:


print ('The mean of Process Time is ' + str(assignment['Processing Time'].mean()))
print ('The median of Process Time is ' + str(assignment['Processing Time'].median()))


# ### Problem 4
# 
# Use Matplotlib to draw a **basic line plot**.
# 
# - use `np.arange(0.0, 1.0, 0.01)` as x-values
# - use `x ** n` for `n=1, 2, 3, 4` as 4 group y-values
# - plot four lines together in one graph with legend

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

# Draw the line plot
x = np.arange(0.0, 1.0, 0.01)
y1 = x**1
y2 = x**2
y3 = x**3
y4 = x**4

plt.plot(x, y1, label = "line 1")
plt.plot(x, y2, label = "line 2")
plt.plot(x, y3, label = "line 3")
plt.plot(x, y4, label = "line 4")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend()

plt.show()


# ### Problem 5
# 
# Suppose your data is `np.random.randn(1000, 3)` where the numpy seed is 19680801 (`np.random.seed(19680801)`)
# 
# You need to provide the **histogram** with 10 bins based on your data.
# 
# Let's color them via `'red', 'tan', 'lime'`.
# 
# Also set your graph title as `bars with legend`.

# In[12]:


np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 3)
colors = ['red', 'tan', 'lime']

# Draw the bar chart
plt.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
plt.legend(prop={'size': 10})
plt.show()

