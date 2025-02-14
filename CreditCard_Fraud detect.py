#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Data loading 

df=pd.read_csv('creditcard.csv')
df


# In[3]:


# Checking the head

df.head(5)


# In[4]:


# Checking the shape of the data
df.shape


# In[5]:


# Checking the data information
df.info()


# In[6]:


# Checking for missing values

df.isnull().sum()


# In[7]:


# deleting the duplicate
df.drop_duplicates(inplace = True)


# In[8]:


df['Class'].value_counts()


# In[9]:


fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]


# In[10]:


# Visualization of the rate of the class

plt.figure(figsize = (5,4))
sns.countplot(x=df['Class']) 
plt.xlabel("Class:0 ='normal', 1='fraud'")
plt.ylabel('Rate')
plt.title('Class Rate')
plt.show()


# In[11]:


print (fraud.shape, normal.shape)


# In[12]:


# To know amount of fraud

fraud.Amount.describe()


# In[13]:


# To know amount of normal transanction

normal_amt=normal.Amount.describe()
normal_amt


# In[14]:


#distribuition of the normal transaction amount

plt.figure(figsize = (5,3))
sns.histplot(normal_amt, bins =30)
plt.title("Distributon of Normal Transaction Amount")
plt.xlabel("Amount")
plt.ylabel("Frequency")


# In[15]:


#  Transaction Period

sns.scatterplot(data = df, x= 'Time', y='Amount')
plt.title("Period of the Transaction")
plt.ylabel("Total amount ")
plt.show


# In[16]:


# Model Building 

x = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[17]:


x.head()


# In[18]:


y.head()


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33, random_state =42)


# In[21]:


# Model building by RandomForest

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train, y_train)


# In[23]:


# model performance

y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)


# In[ ]:




