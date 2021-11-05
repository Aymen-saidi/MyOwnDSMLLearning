#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[7]:


dataset = pd.read_csv(r'C:\Users\Aymen\Documents\GitHub\MyOwnDSMLLearning\Clustering\sec1 Kmeans\Mall_Customers.csv')
X = dataset.iloc[:, 3:].values


# In[6]:


dataset.describe()


# In[8]:


dataset.head(10)


# In[9]:


print(X)


# ## Using the dendrogram to find the optimal number of clusters

# In[9]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# ## Training the K-Means model on the dataset

# In[10]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# ## Visualising the clusters

# In[11]:


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[16]:


get_ipython().system('ipython nbconvert Hierarchical clustering.ipynb --to script')


# In[15]:


get_ipython().system('jupyter nbconvert Hierarchical clustering.ipynb --to python')

