#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd


# In[2]:


# Data loading 

df = pd.read_csv('Twitter_Data.csv')
df


# In[3]:


# Checking the shape of the data

df.shape


# In[4]:


# Checking the data head

df.head()


# In[5]:


# Checking the data information

df.info()


# In[6]:


# Checking for missing values

df.isnull().sum()


# In[7]:


# Dealing with missing values

df[df['clean_text'].isna()]


# In[8]:


df[df['category'].isna()]


# In[9]:


# Deleting the NaN values so that it will not affect our result and as such values are very few

df = df.drop([130448,155642,155698,155770,158693,159442,160559,148,158694,159443,160560], axis =0).reset_index(drop=True)


# In[10]:


df[df['clean_text'].isna()]


# In[11]:


df[df['category'].isna()]


# In[12]:


df.isna().sum()


# In[13]:


# 1 ------> positive 
# 0 ------> Neutral 
# -1 -----> negative

df['category'].value_counts()


# In[14]:


# Separating data and label

x = df['clean_text'].values
y = df['category'].values


# In[15]:


# importing libraries for model building

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
all_stopwords=stopwords.words('english')


# In[16]:


print(df.shape)


# In[17]:


corpus =[]

for i in range(0,162969):
    clean_text = re.sub("[^a-zA-Z]"," ",df['clean_text'][i])
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [ps.stem(word) for word in clean_text if not word in all_stopwords]
    clean_text =' '.join(clean_text)
    corpus.append(clean_text)


# In[18]:


corpus


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features = 1420)


# In[20]:


x = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values


# In[21]:


# Splitting the data into train data and test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =0.20, random_state = 0)


# In[22]:


from sklearn.naive_bayes import GaussianNB


classifier = GaussianNB()
classifier.fit(x_train, y_train)


# In[23]:


# model performance

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




