#!/usr/bin/env python
# coding: utf-8

# # Decision tree regression

# ## Importing librairies

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing dataset

# In[15]:


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values


# In[16]:


dataset.head()


# In[17]:


print(X)
print(Y)


# ## Training the Decision Tree Regression on the whole dataset

# In[18]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) 
regressor.fit(X, Y)


# ## Predicting a new result

# In[19]:


regressor.predict([[6.5]])


# ## Visualizing the Decision Tree Regression result on a higher resolution

# In[23]:


# arange for creating a range of values from min value of X to max value of X  with a difference of 0.01 between two consecutive values
X_grid = np.arange(min(X), max(X), 0.01)
# reshape for reshaping the data into a len(X_grid)*1 array, i.e. to make a column out of the X_grid values
X_grid = X_grid.reshape((len(X_grid), 1)) 
# scatter plot for original data
plt.scatter(X, Y, color = 'red')
# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') 
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')


# In[24]:


from sklearn.tree import export_graphviz 
export_graphviz(regressor, out_file ='tree.dot',
               feature_names =['Production Cost'])

