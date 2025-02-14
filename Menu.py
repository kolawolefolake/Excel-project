#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading and cleaning data

df= pd.read_csv('menu.csv')
df


# In[3]:


# Checking the Data head

df.head()


# In[4]:


# Checking the columns

df.columns


# In[5]:


# Checking the Data type 

df.info()


# In[6]:


# checking for missing values

df.isnull().sum()


# In[7]:


# Checking for duplicate values

df.duplicated().sum()


# In[8]:


# Checking the shape of the data

df.shape


# Observations
# 
# * The data is clean
# * There is no missing value as well no duplicate value

# In[9]:


df.describe()


# In[10]:


#Correlation Coefficient (check numerical columns that correlates with themselves)

df.corr(numeric_only = True)


# In[11]:


plt.figure(figsize = (10,12))
sns.heatmap(df.corr(numeric_only = True), annot = True)
plt.title("Correlation Matrix")
plt.show()


# OBSERVATIONS
# 
# * It is shown that  has a strong positive correlation with Calories 
# *    
#    , sodium and protein.
#   l Fat (% Daily Value)Saturated Fat,Saturated Fat (% Daily Value), sodium and protein.
# * Protein has a strong positive correlation with Iron (% daily value)Dietry and sugar has negative corrlations wth each otherTotal Fat (%and sugar has negative correlation with each other

# In[ ]:





# In[12]:


# Total no of Category and visualisation

df['Category'].value_counts().reset_index()


# In[13]:


plt.figure(figsize = (8,5))
sns.countplot(x=df['Category'])
plt.title("Total No by Category")
plt.show()


# In[14]:


Calory_category = df['Calories'].groupby(by = df['Category']).sum().sort_values(ascending = False).reset_index()
Calory_category


# In[15]:


plt.figure(figsize = (10,5))
sns.barplot(data = Calory_category, x = 'Category', y = 'Calories' )
plt.title("Category By Calory")
plt.show()


# In[17]:


Avg_Calory_Category = df['Calories'].groupby(by = df['Category']).mean().sort_values(ascending = False).reset_index()
Avg_Calory_Category


# In[18]:


plt.figure(figsize = (8,5))
sns.barplot(data = Avg_Calory_Category, x = 'Category', y = 'Calories' )
plt.title("Category By Calory")
plt.show()


# In[19]:


df['Item'].value_counts().sum()


# # Recommendations

# * It is observed that the patronage on Coffee & Tea and Breakfast are very high in consumption compare with other 
#   category items, therefore, it is recommended to continuing make it available.
# * Chicken & Fish is another consistence daily intake therefore the serving per calories should be maintained.
# * Beverages in other hand should be looked into to improve the making of it in the aspect of serving per calories to meet up with 
#   the taste of the consumer to improve the consumption

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




