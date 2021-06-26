#!/usr/bin/env python
# coding: utf-8

# # Multiple Regression model

# ## Importing librairies

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing dataset

# In[6]:


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# In[7]:


print(X)
print(Y)


# ## Encoding categorical data

# In[8]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[9]:


print(X)


# ## Splitting the Dataset into training set and test set
# 

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)


# In[11]:


print(X_train)


# In[12]:


print(X_test)


# In[13]:


print(Y_train) 


# In[14]:


print(Y_test)    


# ## Training the Multiple Linear Regression model on the training set

# In[15]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# ## Predicting the Test set results

# In[16]:


Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#Concatenate both predicted result and real one into one matrice vertically put
#reshape used to help show the result vertically
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))


# ### Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')

# In[17]:


print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))


# Therefore, our model predicts that the profit of a Californian startup which spent 160000 in R&D, 130000 in Administration and 300000 in Marketing is $ 181566,92.
# 
# **Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array. Simply put:
# 
# $1, 0, 0, 160000, 130000, 300000 \rightarrow \textrm{scalars}$
# 
# $[1, 0, 0, 160000, 130000, 300000] \rightarrow \textrm{1D array}$
# 
# $[[1, 0, 0, 160000, 130000, 300000]] \rightarrow \textrm{2D array}$
# 
# **Important note 2:** Notice also that the "California" state was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the second row of the matrix of features X, "California" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, not the last three ones, because the dummy variables are always created in the first columns.
# 

# ### Getting the final linear regression equation with the values of the coefficients

# In[18]:


print(regressor.coef_)
print(regressor.intercept_)


# Therefore, the equation of our multiple linear regression model is:
# 
# $$\textrm{Profit} = 86.6 \times \textrm{Dummy State 1} - 873 \times \textrm{Dummy State 2} + 786 \times \textrm{Dummy State 3} + 0.773 \times \textrm{R&D Spend} + 0.0329 \times \textrm{Administration} + 0.0366 \times \textrm{Marketing Spend} + 42467.53$$
# 
# **Important Note:** To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.

# In[30]:


get_ipython().system('jupyter nbconvert MLR.ipynb --to .ipy')

