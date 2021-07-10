#!/usr/bin/env python
# coding: utf-8

# # Support Vector Regression

# ## Importing librairies

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing dataset

# In[17]:


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


dataset.head()


# In[21]:


Y = Y.reshape(len(Y),1)


# In[22]:


print(Y)


# ## Feature scaling

# In[23]:


#Standartscaler expects a 2D input
from sklearn.preprocessing import StandardScaler
ssx = StandardScaler()
ssy = StandardScaler()
X = ssx.fit_transform(X)
Y = ssy.fit_transform(Y)


# In[24]:


print(X)


# In[25]:


print(Y)


# ## Training the Support Vector Regression on the whole set

# In[26]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)


# ## Predcting a new result

# In[33]:


ssy.inverse_transform(regressor.predict(ssx.transform([[6.5]])))
# we need to inverse the feature scaling to have the right result


# ## Visualising the SVR results 

# In[44]:


plt.scatter(ssx.inverse_transform(X),ssy.inverse_transform(Y), color = 'red')
plt.plot(ssx.inverse_transform(X), ssy.inverse_transform(regressor.predict((X))), color = 'green')
# X in the predict method is correct because it's already scaled
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')


# ## Visualising the SVR results with smoother curve and higher resolution

# In[53]:


X_grid = np.arange(min(ssx.inverse_transform(X)), max(ssx.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ssx.inverse_transform(X),ssy.inverse_transform(Y), color = 'red')
plt.plot(X_grid, ssy.inverse_transform(regressor.predict((ssx.transform(X_grid)))), color = 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# ## Converting to script

# In[54]:


get_ipython().system('jupyter nbconvert --to script SVR.ipynb')

