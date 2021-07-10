#!/usr/bin/env python
# coding: utf-8

# ## Improrting libraries
# 

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing dataset

# In[14]:


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# In[15]:


print(X)
print(Y)


# In[16]:


dataset.head()


# # Splitting the Dataset into training set and test set

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)


# In[18]:


print(X_train)


# In[19]:


print(X_test)


# In[20]:


print(Y_train) 


# In[21]:


print(Y_test)    


# # Applying the SLR model on the training set 

# In[35]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show() 


# # Predicting the Test set results
# 
# 

# In[30]:


Y_pred = regressor.predict(X_test)


# # Visualizing the training set results 

# In[31]:


plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# # Visualizing the test set results 

# In[33]:


plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

