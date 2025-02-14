#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading and cleaning the data

pd_nyc = pd.read_csv('AB_NYC_2019.csv')


# In[3]:


pd_nyc


# In[4]:


# Checking the details of the data


pd_nyc.head(10)


# In[5]:


# Checking the shape of the data

pd_nyc.shape


# In[6]:


# Checking the datatype 

pd_nyc.info()


# In[7]:


# Checking for duplicate values

pd_nyc.duplicated().sum()


# In[8]:


# Checking for null values

pd_nyc.isnull().sum()


# # Observation
# The two columns ('last_review' and 'reviews_per_month') have too many missing values of over 10000 rows per each, 
# instead of dropping the missing values that will affect the result, I will rather drop the 2 columns without have any effect on my result.

# In[9]:


# Dropping the two columns with many null values

pd_new = pd_nyc.drop(columns = ['last_review','reviews_per_month'])


# In[10]:


pd_new


# In[11]:


# Checking for the remaining null values

pd_new.isnull().sum()


# In[12]:


# dropping the missing values

pd_new.dropna(inplace = True)


# In[13]:


pd_new.isnull().sum()


# In[14]:


# Checking the shape of the data before and after dealing with the missing values

print(pd_nyc.shape)
print(pd_new.shape)


# In[15]:


pd_new.head()


# In[24]:


# sepating the numerical from categorical columns

pd_new_categ = pd_new.select_dtypes(include="object")
pd_new_numb = pd_new.select_dtypes(include="number")


# In[25]:


pd_new_categ


# In[26]:


pd_new_numb


# In[27]:


# the outlier in the numerical columns

for i in pd_new_numb.columns:
    sns.boxplot(data = pd_new_numb, x=i)
    plt.show()

