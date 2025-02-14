#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Data loading and cleaning

pd = pd.read_csv('WineQT.csv')


# In[4]:


pd


# In[5]:


# Checking the shape of the data
pd.shape


# In[6]:


# Checking the head

pd.head()


# In[7]:


# Checking for missing values

pd.isnull().sum()


# In[9]:


# Checking the data information
pd.info()


# In[8]:


## Performing EDA using statistical version

pd.describe()


# In[12]:


# Visualization of the number of values of each Quality

sns.catplot(kind= 'count',data =pd, x= 'quality')


# In[13]:


# visualization of value that is related to the quality using density vs quality

plt.figure(figsize =(5,5))
sns.barplot(data = pd, x = 'quality', y = 'density')


# In[14]:


# visualization of value that is related to the quality using Acidity vs quality

plt.figure(figsize =(5,5))
sns.barplot(data = pd, x = 'quality', y = 'volatile acidity')


# In[15]:


# visualization of value that is related to the quality using Acidity vs quality

plt.figure(figsize =(5,5))
sns.barplot(data = pd, x = c, y = 'citric acid')


# In[ ]:


## Find the correlations btw the columns
plt.figure(figsize = (10,10)
sns.heatmap


# In[16]:


## Find the correlations btw the columns

pd.corr(numeric_only = True)


# In[20]:


plt.figure(figsize = (8,8))
sns.heatmap(pd.corr(numeric_only = True), annot = True)
plt.title("Correlation Matrix")
plt.show()


# In[25]:


# To preprocessing the data
# dropping the quality column

train = pd[pd['quality'] < 7]
test = pd[pd['quality'] < 7]


# In[26]:


from sklearn.model_selection import train_test_split


# In[28]:


x = pd.drop('quality', axis =1)
y = pd['quality'] 


# In[29]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33, random_state =42)


# In[30]:


from sklearn.preprocessing import StandardScaler


# In[31]:


scaler = StandardScaler()


# In[32]:


pd_scaled = scaler.fit_transform(pd)


# In[38]:


x_train_scaled = scaler.fit_transform(x_train)


# In[34]:


x_test_scaled = scaler.transform(x_test)


# In[35]:


# Model building

from sklearn.linear_model import LogisticRegression


# In[36]:


classification = LogisticRegression()


# In[39]:


classification.fit(x_train_scaled, y_train)


# In[41]:


classification.predict(x_test_scaled)


# In[42]:


# Model building by RandomForest

from sklearn.ensemble import RandomForestClassifier


# In[43]:


model = RandomForestClassifier()


# In[44]:


model.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[45]:


x_test_prediction = model.predict(x_test)


# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


test_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[49]:


print('accuracy:', test_data_accuracy)


# In[ ]:




