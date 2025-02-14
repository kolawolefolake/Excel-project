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

data = pd.read_csv('apps.csv')


# In[3]:


# Checking the head

data.head(5)


# In[4]:


# Checking the shape of the data
data.shape


# In[5]:


# Checking the data information

data.info()


# In[6]:


# Dropping some columns, for we dont need them

data.drop(columns = ['Unnamed: 0','Genres','Current Ver','Android Ver'], axis =1, inplace =True)


# In[7]:


data.head()


# In[8]:


# Checking for missing values

data.isnull().sum()


# In[9]:


# Dealing with missing values. 
# Rating and Size has missing values of 1463 and 1227 respectively so i will apply the mean value

data['Rating'] = data['Rating'].fillna(data['Rating'].mean())
data['Size'] = data['Size'].fillna(data['Size'].mean())


# In[10]:


# Changing the Date from object dtype to date dtype

data['Last Updated'] = pd.to_datetime(data['Last Updated'])


# In[11]:


data.isnull().sum()


# In[12]:


data.info()


# In[13]:


# Checking for duplicate values

data.duplicated().sum()


# # Performing EDA

# In[14]:


data.describe()


# In[15]:


data.describe(include ="object")


# In[16]:


# Distribution of the data
df= data.drop('Reviews', axis = 1)


# In[17]:


for i in df.select_dtypes(include="number").columns:
    sns.histplot(data = df, x=i)
    plt.show()


# In[18]:


# Visualization of the relationship by scatter plot

for i in ['Reviews','Size','Price']:
    sns.scatterplot(data = data, x=i, y='Rating')
    plt.show()


# In[19]:


data['Category'].value_counts()


# In[20]:


plt.figure(figsize = (10,3))
sns.countplot(x=df['Category'])
plt.title("count of Categories")
plt.show()


# In[21]:


# Type by percentage and visualized

data['Type'].value_counts()


# In[22]:


print((data.groupby('Type')['Type'].count()/data['Type'].count())*100)


# In[23]:


((data.groupby('Type')['Type'].count()/data['Type'].count())*100).plot.pie()


# # Sentiment Analysis on user_review

# In[24]:


# loading the review data for review sentiment analysis

df= pd.read_csv('user_reviews.csv')
df


# In[25]:


# Checking the data head

df.head()


# In[26]:


# Checking the data shape

df.shape


# In[27]:


# Checking the data info

df.info()


# In[28]:


# Dealing with missing values

df[df['Translated_Review'].isna()]


# In[29]:


df[df['Translated_Review'].isna()]


# In[30]:


df[df['Translated_Review'].isna()]


# In[31]:


df[df['Translated_Review'].isna()]


# # Observation
# 
# The missing value in this dataset is the half of the dataset in which, if we are to carry out any Analysis with the messy data we can not get the accurate result, in the cause of large volume of the missing value, we are going to seek an approval of using the available data and to delete the null values records to perform the data Analysis and the sentiment Analysis. The available data will also be used to build the model on Sentiment Analysis, so that the result will be used to predict the subsequent incoming data.

# In[32]:


df.dropna(inplace = True)


# In[33]:


# Checking the null values

df.isnull().sum()


# In[34]:


# checking the shape after dropping null values. This is the data we ll be using for the rest of the analysis

print(df.shape)


# In[35]:


df.head()


# In[36]:


# Renaming some columns

df.rename(columns ={'Translated_Review':'review','Sentiment':'remarks'}, inplace=True)


# In[37]:


df.head()


# In[38]:


df.head()


# In[39]:


df['remarks'].value_counts()o


# In[40]:


# 0 ---------> Nuetral
# 1----------> Positive
# -1----------> Negative


df['remarks'][df.iloc[:,2]=="Neutral"] = 0
df['remarks'][df.iloc[:,2]=="Positive"] = 1
df['remarks'][df.iloc[:,2]=="Negative"] = -1


# In[41]:


df.head()


# In[42]:


df['remarks'].dtype


# In[43]:


# converting the remarks dtype from object to float dtype


df['remarks'] = df['remarks'].replace(',', '.').astype(float)


# In[44]:


# importing libraries for model building

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import RegexpStemmer
st = RegexpStemmer('ing$|s$|e$|able$', min=4)
ps = PorterStemmer()
all_stopwords=stopwords.words('english')


# In[45]:


from nltk.tokenize import sent_tokenize, word_tokenize
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

sentence=df['review']


# In[46]:


sentence


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features = 1420)


# In[48]:


# Separating the data and label

x = cv.fit_transform(sentence).toarray()
y = df.iloc[:,2].values


# In[49]:


# Splitting the data into train data and test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =0.20, random_state = 0)


# In[50]:


from sklearn.naive_bayes import GaussianNB


classifier = GaussianNB()
classifier.fit(x_train, y_train)


# In[51]:


# model performance

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




