#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libararies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading and cleaning data

df=pd.read_csv('customer_shopping_data.csv')
df


# In[3]:


# Checking the Data head

df.head()


# In[4]:


# Checking the Data type 

df.info()


# In[5]:


# Checking the columns

df.columns


# In[6]:


# Checking the shape of the data

df.shape


# In[7]:


# checking for missing values

df.isnull().sum()


# In[8]:


# Checking for duplicate values

df.duplicated().sum()


# In[9]:


# Checking the uniqueness of the columns

df.nunique()


# In[10]:


df.describe()


# ## OBSERVATIONS

# * The data is clean
# * There is no missing value as well no duplicate value
# * The average price is approximately 689.26
# * The age of the customer ranges from 18 years to 69 years.
# * Most customers are around the age 56
# * Most Items cost 5250.
# * Most quatity purchased by the customer is 5.
#   
# 

# In[11]:


#Correlation Coefficient (check numerical columns that correlates with themselves)

df.corr(numeric_only = True)


# In[12]:


sns.heatmap(df.corr(numeric_only = True), annot = True)
plt.title("Correlation Matrix")
plt.show()


# Observation
# 
# It is observed that there is positive correlation between price and quantity.

# # Time series application

# In[13]:


# Renaming the invoice_ date into Date

df = df.rename(columns = {'invoice_date' : 'Date'})


# In[14]:


df


# In[15]:


# convert 'Date' to datetime and set as index

df['Date'] = pd.to_datetime(df['Date'], format = "mixed")
df.set_index('Date', inplace = True)
df


# In[16]:


df


# In[17]:


# select the customer shopping for 2021

df.loc['2021']


# In[18]:


# Checking for 7 -day rolling average of the price 

df['7_day_rolling'] = df['price'].rolling(window = 7).mean()
df['7_day_rolling']


# In[19]:


# Checking for 30 -day rolling average of the price

df['30_day_rolling'] = df['price'].rolling(window = 30).mean()
df['30_day_rolling']


# In[20]:


# Extract time variable

df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df


# In[21]:


# To plot the Monthly Price

df.resample('ME').sum()['price'].plot()
plt.show()


# In[22]:


# To plot the Monthly Price

df.resample('YE').sum()['price']


# In[23]:


df.resample('YE').sum()['price'].plot()


# # Customer and Product ANALYSIS

# In[24]:


#PRICE BY CATEGORY

Total_Category = df['price'].groupby(by = df['category']).sum().sort_values(ascending = False).reset_index()
Total_Category


# In[25]:


# Visualization using barchart
plt.figure(figsize = (10, 5))
sns.barplot(data = Total_Category, x = "category", y = "price")
plt.title("Price By Category")
plt.xlabel("Category")
plt.ylabel("Price")
plt.show()


# In[26]:


# Category by Quantity and visualization

Quantity_Categ = df['quantity'].groupby(by= df['category']).sum().sort_values(ascending = False).reset_index()
Quantity_Categ

plt.figure(figsize = (7,10))
sns.barplot(data = Quantity_Categ, x = "category", y = "quantity")
plt.title(" Category By Quantity")
plt.show()


# In[27]:


# Total quantity by payment method and visualization

qty_by_pay = df['quantity'].groupby(by = df['payment_method']).sum().reset_index()
qty_by_pay


# In[28]:


plt.figure(figsize = (5,3))
sns.barplot(data = qty_by_pay, x = "payment_method", y = "quantity")
plt.title(" Total Quantity by Payment Method")
plt.show()


# In[29]:


# Total price by payment method
Total_price_pay = df['price'].groupby(by = df['payment_method']).sum().sort_values(ascending = False).reset_index()
Total_price_pay


# In[30]:


plt.figure(figsize = (5,3))
sns.barplot(data = Total_price_pay, x = "payment_method", y = "price")
plt.title(" Total Price by Payment Method")
plt.show()


# In[31]:


## Age rate by Gender and visualisation

age_rate_gend = df['age'].groupby(by = df['gender']).sum()

# percentage by gender
round(age_rate_gend/age_rate_gend.sum(),1)


# In[32]:


plt.pie(age_rate_gend, labels = [ 'Female (0.6%)','Male (0.4%)'], radius = 0.8)
plt.show()


# # Recommendations

# * with the critical inspection of the category by quantity, the female clothing is on high side therefore the availability, quality standard
#   and the legacy should be maintained
# * The book and Technology section needs improvement, and a kind of gifting or advert should be introduced to create widely awareness to the public
# * The services of the debit cards should be looked into and make it more robust to encourage the debit card holder for wide range patronage.
# * The age should also be put into consideration especially the younger age, a kind of questionaire techniques should be carried out to know the area of 
#   interest among younger ones 
#   
