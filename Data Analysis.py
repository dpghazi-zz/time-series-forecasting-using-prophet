#!/usr/bin/env python
# coding: utf-8

# # Predict Future Avocado Prices Using Facebook Prophet 

# ### Prepared by Donald Ghazi

# ## Project Overview 

# - In this project, I will predict the future prices of avocados using Facebook Prophet.

# ## What is Facebook Prophet?
# - Tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth.
# - Source: https://facebook.github.io/prophet/

# # Import Libraries and Dataset 

# - I first installed fbprophet package as follows: pip install fbprophet
# - Source: https://github.com/facebook/prophet

# In[1]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet


# In[2]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('avocado.csv')


# 
# - Date: The date of the observation
# - AveragePrice: the average price of a single avocado
# - type: conventional or organic
# - year: the year
# - Region: the city or region of the observation
# - Total Volume: Total number of avocados sold
# - 4046: Total number of avocados with PLU 4046 sold
# - 4225: Total number of avocados with PLU 4225 sold
# - 4770: Total number of avocados with PLU 4770 sold

# In[3]:


# view the head of the training dataset
avocado_df.head()


# In[4]:


# view the last elements in the training dataset
avocado_df.tail(10)


# In[5]:


avocado_df.describe()


# In[6]:


avocado_df.info()


# In[7]:


avocado_df.isnull().sum()


# # Explore Dataset

# In[8]:


avocado_df = avocado_df.sort_values('Date')


# In[9]:


# plot date and average price
plt.figure(figsize = (10,10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])


# In[10]:


# plot distribution of the average price
plt.figure(figsize = (10, 6))
sns.distplot(avocado_df['AveragePrice'], color = 'b')


# In[11]:


# plot a violin plot of the average price vs. avocado type
sns.violinplot(y = 'AveragePrice', x ='type', data =avocado_df)


# In[12]:


# bar chart to indicate the number of regions 

sns.set(font_scale=0.7) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)


# In[13]:


# bar chart to indicate the count in every year
sns.set(font_scale=1.5) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[14]:


# plot the avocado prices vs. regions for conventional avocados
conventional = sns.catplot('AveragePrice', 'region', data = avocado_df[avocado_df['type']=='conventional'],hue = 'year', height = 20)


# In[15]:


# plot the avocado prices vs. regions for organic avocados
conventional = sns.catplot('AveragePrice', 'region', data = avocado_df[avocado_df['type']=='organic'],hue = 'year', height = 20)


# # Prepare the Data Before Applying Facebook Prophet Tool 

# In[16]:


avocado_df


# In[17]:


avocado_prophet_df = avocado_df[['Date','AveragePrice']] 


# In[18]:


avocado_prophet_df


# In[19]:


avocado_prophet_df = avocado_prophet_df.rename(columns = {'Date': 'ds', 'AveragePrice': 'y'})


# In[20]:


avocado_prophet_df 


# # Develop Model and Make Predictions 

# In[21]:


m = Prophet()
m.fit(avocado_prophet_df)


# In[22]:


# Forcasting into the future
future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)


# In[23]:


forecast


# In[24]:


figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Price ')


# In[25]:


figure2 = m.plot_components(forecast)


# # Develop Model and Make Predictions of West Region 

# In[26]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('avocado.csv')


# In[27]:


# Select specific region
avocado_df_sample = avocado_df[avocado_df['region']=='West']


# In[28]:


avocado_df_sample = avocado_df_sample.sort_values('Date')


# In[29]:


plt.plot(avocado_df_sample['Date'],avocado_df_sample['AveragePrice'])


# In[30]:


avocado_df_sample = avocado_df_sample.rename(columns = {'Date':'ds','AveragePrice':'y'})


# In[31]:


m = Prophet()
m.fit(avocado_df_sample)
# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[32]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[33]:


figure3 = m.plot_components(forecast)

