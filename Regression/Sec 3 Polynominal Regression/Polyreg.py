#!/usr/bin/env python
# coding: utf-8

# # Polynominal Regression model

# ## Importing librairies  

# In[84]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing dataset

# In[85]:


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values


# In[86]:


print(X)


# In[87]:


print(Y)


# In[88]:


dataset.head()


# ## Splitting the Dataset into training set and test set

# In[89]:


#we wont be splitting the dataset cause the dataset itself is so few


# ## Training the Linear Regression model on the training set

# In[90]:


from sklearn.linear_model import LinearRegression
regressorLR = LinearRegression()
regressorLR.fit(X,Y)


# ## Training the Polynomial Regression model on the training set

# In[107]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
Xpoly = poly_reg.fit_transform(X)
regressorPL = LinearRegression().fit(Xpoly,Y)


# ## Visualizing the Linear Regression model results

# In[92]:


plt.scatter(X,Y, color = 'red')
#regressoLR.predict(X): predicted Y values
plt.plot(X, regressorLR.predict(X), color = 'green')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Visualizing the Polynomial Regression model results

# In[108]:


plt.scatter(X,Y, color = 'red')
plt.plot(X, regressorPL.predict(Xpoly), color = 'green')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Visualizing the Polynomial Regression model results with higher resolution and smoother curve

# In[109]:


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y, color = 'red')
plt.plot(X_grid, regressorPL.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# ## Predicting a new result with the Linear Regression model

# In[104]:


regressorLR.predict([[6.5]])


# ## Predicting a new result with the Polynomial Regression model

# In[111]:


regressorPL.predict(poly_reg.fit_transform([[6.5]]))


# In[112]:


get_ipython().system("jupyter nbconvert --to script 'Poly reg.ipynb'")

