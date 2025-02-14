#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Data loading and cleaning

df_ca = pd.read_csv('CAvideos.csv')


# In[3]:


df_ca


# In[4]:


# Checking the details of the data

df_ca.head(10)


# In[5]:


# Checking the shape

df_ca.shape


# In[6]:


# Checking the datatype

df_ca.info()


# In[7]:


# Checking for duplicate values

df_ca.duplicated().sum()


# In[8]:


# Checking for null values

df_ca.isnull().sum()


# In[9]:


# There is a column that contains many null values that can affect our result hence, it will be dropped.
# The trending_date will also be dropped, for we dont need it


df_ca.drop(columns = ['description','trending_date'], inplace = True)


# In[10]:


df_ca


# In[11]:


df_ca.isnull().sum()


# # Observation
# 
# * There is missing value of 1296 in the column "description" that will not allow us to have accurate data result,
#   I will drop the columns without have any effect on our result.
# * The trending date dtype is in object, so it will be changed to date dtype
# * Thereafter some columns will be renamed for proper identification

# In[12]:


df_ca.columns


# In[13]:


# Conversion of trending date to date dtype

df_ca['publish_time'] = pd.to_datetime(df_ca['publish_time'], format = "mixed")


# In[14]:


# Renaming the columns

df_ca.rename(columns ={'comments_disabled':'comments', 'ratings_disabled':'rating','video_error_or_removed':'video_error'})


# In[15]:


# Columns tha
df_ca.head()


# In[ ]:





# In[16]:


# the outlier in the numerical columns

for i in df_ca.select_dtypes(include="number").columns:
    sns.boxplot(data = df_ca, x=i)
    plt.show()


# In[18]:


#  dislikes

plt.boxplot(df_ca.dislikes)
Q2 = df_ca.dislikes.quantile(0.50)
Q3 = df_ca.dislikes.quantile(0.75)
IQR = Q3 - Q2
df_ca = df_ca[(df_ca.dislikes >= Q2 - 1.5*IQR) & (df_ca.dislikes <= Q2+ 1.5*IQR)]


# In[19]:


#  category_id

plt.boxplot(df_ca.category_id)
Q2 = df_ca.category_id.quantile(0.50)
Q3 = df_ca.category_id.quantile(0.75)
IQR = Q3 - Q2
df_ca = df_ca[(df_ca.category_id >= Q2 - 1.5*IQR) & (df_ca.category_id <= Q2+ 1.5*IQR)]


# In[20]:


# likes

plt.boxplot(df_ca.likes)
Q2 = df_ca.likes.quantile(0.50)
Q3 = df_ca.likes.quantile(0.75)
IQR = Q3 - Q2
df_ca = df_ca[(df_ca.likes >= Q2 - 1.5*IQR) & (df_ca.likes <= Q2+ 1.5*IQR)]


# In[21]:


# views	

plt.boxplot(df_ca.views)
Q2 = df_ca.views.quantile(0.50)
Q3 = df_ca.views.quantile(0.75)
IQR = Q3 - Q2
df_ca = df_ca[(df_ca.views >= Q2 - 1.5*IQR) & (df_ca.views <= Q2+ 1.5*IQR)]


# In[ ]:




