#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import yfinance as yf


# In[4]:


stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']


# In[6]:


data = {}


# In[8]:


dataframes = []


# In[12]:


for stock in stocks:
    df = yf.download(stock, start='2015-01-01', end='2023-01-01')
    df.rename(columns={'Adj Close': stock}, inplace=True)
    dataframes.append(df)


# In[14]:


price_data = pd.concat(dataframes, axis=1)


# In[16]:


returns = price_data.pct_change().dropna()


# In[18]:


scaler = StandardScaler()


# In[20]:


returns_scaled = scaler.fit_transform(returns)


# In[22]:


pca = PCA()


# In[24]:


returns_pca = pca.fit_transform(returns_scaled)


# In[26]:


plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.show()


# In[28]:


n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1


# In[30]:


print(f"Number of components explaining 95% variance: {n_components}")


# In[32]:


pca = PCA(n_components=n_components)


# In[34]:


returns_pca = pca.fit_transform(returns_scaled)


# In[78]:


dbscan = DBSCAN(eps=0.7,min_samples=2)


# In[80]:


labels = dbscan.fit_predict(returns_pca)


# In[82]:


plt.figure(figsize=(10, 6))
unique_labels = set(labels)

for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    plt.scatter(returns_pca[class_member_mask, 0], returns_pca[class_member_mask, 1],
                label=f'Cluster {k}', alpha=0.6)

plt.title('DBSCAN Clustering of Assets')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()


# In[84]:


print("DBSCAN Labels:", np.unique(labels))


# In[88]:


if 'Cluster' not in returns.columns:
    returns['Cluster'] = labels


# In[90]:


if 'Cluster' in returns.columns and not returns['Cluster'].isnull().all():
    cluster_summary = returns.groupby('Cluster').mean()
    print("Cluster Summary:")
    print(cluster_summary)
else:
    print("No valid clusters found. Check DBSCAN parameters or input data.")


# In[92]:


print(returns.head())


# In[96]:


if 'Cluster' in returns.columns:
    cluster_summary = returns.groupby('Cluster').mean()
    print("Cluster Summary:")
    print(cluster_summary)
else:
    print("Error")


# In[102]:


print('Cluster Summary')
print()
print(cluster_summary)


# In[104]:


returns.to_csv('portfolio_clusters.csv')

