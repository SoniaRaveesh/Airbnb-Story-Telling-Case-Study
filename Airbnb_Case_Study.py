#!/usr/bin/env python
# coding: utf-8

# # Storytelling Case Study: Airbnb, NYC

# ## Problem Statement

# For the past few months, Airbnb has seen a major decline in revenue. Now that the restrictions have started lifting and people have started to travel more, Airbnb wants to make sure that it is fully prepared for this change.
# 
#  
# 
# The different leaders at Airbnb want to understand some important insights based on various attributes in the dataset so as to increase the revenue such as -
# 
# - Which type of hosts to acquire more and where?
# 
# The categorisation of customers based on their preferences.
# 
# - What are the neighbourhoods they need to target?
# 
# - What is the pricing ranges preferred by customers?
# 
# The various kinds of properties that exist w.r.t. customer preferences.
# 
# Adjustments in the existing properties to make it more customer-oriented.
# 
# - What are the most popular localities and properties in New York currently?
# 
# - How to get unpopular properties more traction? and so on...

# # Reading and Understanding the Data
# 

# #### Importing Libraries

# In[2]:


# Imporitng the required packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


# Import warnings 

import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Read the csv file using pandas

airbnb_df = pd.read_csv("AB_NYC_2019.csv")


# In[5]:


# Check first few rows

airbnb_df.head()


# id	name	host_id	host_name	neighbourhood_group	neighbourhood	latitude	longitude	room_type	price	minimum_nights	number_of_reviews	last_review	reviews_per_month	calculated_host_listings_count	availability_365
# 0	2539	Clean & quiet apt home by the park	2787	John	Brooklyn	Kensington	40.64749	-73.97237	Private room	149	1	9	19-10-2018	0.21	6	365
# 1	2595	Skylit Midtown Castle	2845	Jennifer	Manhattan	Midtown	40.75362	-73.98377	Entire home/apt	225	1	45	21-05-2019	0.38	2	355
# 2	3647	THE VILLAGE OF HARLEM....NEW YORK !	4632	Elisabeth	Manhattan	Harlem	40.80902	-73.94190	Private room	150	3	0	NaN	NaN	1	365
# 3	3831	Cozy Entire Floor of Brownstone	4869	LisaRoxanne	Brooklyn	Clinton Hill	40.68514	-73.95976	Entire home/apt	89	1	270	05-07-2019	4.64	1	194
# 4	5022	Entire Apt: Spacious Studio/Loft by central park	7192	Laura	Manhattan	East Harlem	40.79851	-73.94399	Entire home/apt	80	10	9	19-11-2018	0.10	1	0

# In[6]:


# Check the shape of dataset

airbnb_df.shape

#(48895, 16)


# #### The dataset has 48895 rows and 16 columns

# In[7]:


#Checking info & datatypes 

airbnb_df.info()


# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 48895 entries, 0 to 48894
# Data columns (total 16 columns):
#  #   Column                          Non-Null Count  Dtype  
# ---  ------                          --------------  -----  
#  0   id                              48895 non-null  int64  
#  1   name                            48879 non-null  object 
#  2   host_id                         48895 non-null  int64  
#  3   host_name                       48874 non-null  object 
#  4   neighbourhood_group             48895 non-null  object 
#  5   neighbourhood                   48895 non-null  object 
#  6   latitude                        48895 non-null  float64
#  7   longitude                       48895 non-null  float64
#  8   room_type                       48895 non-null  object 
#  9   price                           48895 non-null  int64  
#  10  minimum_nights                  48895 non-null  int64  
#  11  number_of_reviews               48895 non-null  int64  
#  12  last_review                     38843 non-null  object 
#  13  reviews_per_month               38843 non-null  float64
#  14  calculated_host_listings_count  48895 non-null  int64  
#  15  availability_365                48895 non-null  int64  
# dtypes: float64(3), int64(7), object(6)
# memory usage: 6.0+ MB

# #### As we can observe that there are various type of numerical as well as categorical variables present in the dataframe. And also we can see that some columns have missing values, lets evaluate that first.

# In[8]:


# Checking for the null values in the columns : 

airbnb_df.isna().sum()


# id                                    0
# name                                 16
# host_id                               0
# host_name                            21
# neighbourhood_group                   0
# neighbourhood                         0
# latitude                              0
# longitude                             0
# room_type                             0
# price                                 0
# minimum_nights                        0
# number_of_reviews                     0
# last_review                       10052
# reviews_per_month                 10052
# calculated_host_listings_count        0
# availability_365                      0
# dtype: int64

# #### As we can observe that out of 16 columns , 4 have missing values.Lets evaluate each column one by one for missing values and try to impute instead of dropping the columns right away.

# # Data Cleaning
# 

# As we have observed that the missing data does not require a lot of extra consideration. Further observations can be made based on the characteristics of our dataset: the columns "name" and "host_name" are unimportant and irrelevant to our data analysis, while the columns "last_review" and "review_per_month" require very straightforward handling. To clarify, the term "last_review" refers to the date; if there are no reviews for the listing, the data will be absent obviously. Since this column is unnecessary and unimportant in our situation, it is not necessary to append those values but we can extract year from the date for ease of understanding.For the "review_per_month" column, we can simply add it with 0 for missing data. 

# In[9]:


# lets replace the nulls in review_per_month with a 0 for ease of understanding :

airbnb_df.fillna({'reviews_per_month':0},inplace= True)

# checking for null values in review_per_month column

airbnb_df['reviews_per_month'].isna().sum()

# no nulls present


# In[10]:


# lets extract year from last_review and dropping the last review column, for nulls lets keep it as it is.

airbnb_df['Year']=pd.DatetimeIndex(airbnb_df['last_review']).year

airbnb_df = airbnb_df.drop(['last_review'],axis=1)

airbnb_df.head(5)


# In[11]:


# lets drop the following columns : id, name, host_name - as they are not only irrelevant but also for ethical reasons.

airbnb_df.drop(['id','name','host_name'], axis=1, inplace=True)

# checking first 3 rows

airbnb_df.head(3)


# In[12]:


# Creating a copy dataframe for reference analytics to be utilized later : 

df_2 = airbnb_df.copy(deep=False)


# ## Checking Unique Values
# 

# In[ ]:


# checking categorical variables


# In[13]:


airbnb_df.room_type.unique()

#['Private room', 'Entire home/apt', 'Shared room']

len(airbnb_df.room_type.unique())

#3


# In[14]:


airbnb_df.neighbourhood_group.unique()

#['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']


# In[15]:


len(airbnb_df.neighbourhood.unique())

#221


# In[16]:


# lets check for top hosts, who have the most listings on Airbnb platform 

top_host=airbnb_df.host_id.value_counts().head(10)
top_host


# In[42]:


# lets save the dataframe  to excel

airbnb_df.to_excel('Airbnb_1.xlsx')


# In[ ]:





# 219517861    327
# 
# 107434423    232
# 
# 30283594     121
# 
# 137358866    103
# 
# 16098958      96
# 
# 12243051      96
# 
# 61391963      91
# 
# 22541573      87
# 
# 200380610     65
# 
# 7503643       52
# 
# top host_id - 219517861

# In[17]:


# lets do binning for countinous variables

#Creating bins for the Price column : 

price_ranges = ['low','medium','high','very high']
price_ranges

#['low', 'medium', 'high', 'very high']


# In[18]:


df_2['price_range'] = pd.cut(x=df_2['price'],bins = [1,99,999,4999,10000],labels= price_ranges)


# In[19]:


# checking data

df_2.head()


# In[20]:


# Creating bins for the minimum_nights column : 

min_nights_range = ['Less than a week','1-2 Weeks','2-4 Weeks','1-3 Months','3 Months - 1 Year','More than a Year']

min_nights_range

#['Less than a week','1-2 Weeks','2-4 Weeks','1-3 Months','3 Months - 1 Year','More than a Year']


# In[21]:


df_2['min_nights_range'] = pd.cut(x=df_2['minimum_nights'],bins = [0,8,15,31,91,365,400],labels = min_nights_range)


# In[22]:


# checking data

df_2.head()


# In[23]:


# Creating bins for the no_of_reviews column :

no_of_reviews_range = ['Very Low','Low','Average','High','Very High']
no_of_reviews_range

#['Very Low', 'Low', 'Average', 'High', 'Very High']


# In[24]:


df_2['no_of_reviews_range'] = pd.cut(x=df_2['number_of_reviews'],bins = [1,100,200,300,400,1000],labels = no_of_reviews_range)


# In[25]:


# checking data

df_2.head()


# In[26]:


# Creating bins for year column : 

year_range = ['Old','Infrequent','Recent']
year_range

#['Old', 'Infrequent', 'Recent']


# In[27]:


df_2['year_range'] = pd.cut(x=airbnb_df['Year'],bins = [2011,2015,2017,2019],labels = year_range)


# In[28]:


# checking data

df_2.head()


# In[29]:


# Creating bins for Reviews per month column : 

review_per_month_range = ['Low','Medium','High']
review_per_month_range


# In[30]:


df_2['reviews_month_range'] = pd.cut(x=df_2['reviews_per_month'],bins = [0,5,10,100],labels = review_per_month_range)


# In[31]:


# checking data

df_2.head()


# In[32]:


# Creating bins for calculated host listings count column : 

host_listing_range = ['Low','Medium','High']
host_listing_range


# In[33]:


df_2['host_listing_range'] = pd.cut(x=airbnb_df['calculated_host_listings_count'],bins = [0,50,100,1000],labels = host_listing_range)


# In[34]:


# checking data

df_2.head()


# In[35]:


# Creating bins for the availability 365 column : 

availability_365_range = ['0-60 days','60-120 days','120-180 days','180-240 days','240-300 days','300-370 days']
availability_365_range


# In[36]:


df_2['availability_365_range'] = pd.cut(x=df_2['availability_365'],bins = [0,60,120,180,240,300,370],labels=availability_365_range)


# In[37]:


# checking data

df_2.head()


# In[38]:


# checking columns 

df_2.columns

#['host_id', 'neighbourhood_group', 'neighbourhood', 'latitude','longitude', 'room_type', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month','calculated_host_listings_count', 'availability_365', 'Year','price_range', 'min_nights_range', 'no_of_reviews_range', 'year_range','reviews_month_range', 'host_listing_range', 'availability_365_range'],


# In[39]:


# lets drop the extra columns as we have created the bins for required columns

df_2.drop(['calculated_host_listings_count','Year','number_of_reviews','minimum_nights','reviews_per_month','availability_365'],axis = 1 ,  inplace = True)


# In[40]:


# checking head 

df_2.head()


# In[41]:


# lets convert the dataframe df_2 to excel

df_2.to_excel('Airbnb.xlsx')


# In[ ]:





# In[ ]:




