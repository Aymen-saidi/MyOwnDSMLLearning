#!/usr/bin/env python
# coding: utf-8

# # Polynomial Regression

# ## Importing the libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[3]:


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[4]:


dataset.head()


# ## Splitting the dataset into the Training set and Test set

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Training the Polynomial Regression model on the Training set

# In[6]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)


# ## Predicting the Test set results

# In[7]:


y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[8]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

