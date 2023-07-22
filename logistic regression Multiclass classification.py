#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Iris.csv')


# In[3]:


df


# In[5]:


df['Species'].unique()


# In[8]:


df['Species'].replace({'Iris-setosa':'1','Iris-versicolor':'2','Iris-virginica':'3'},inplace=True)


# In[9]:


df


# In[11]:


x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values


# In[12]:


y=df[['Species']].values


# In[13]:


from sklearn.model_selection import train_test_split


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],df['Species'],test_size=0.2)


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


lr=LogisticRegression()


# In[21]:


lr.fit(x_train,y_train)


# In[22]:


lr.predict(x_test)


# In[23]:


x_test


# In[24]:


lr.score(x_test,y_test)


# In[26]:


sns.pairplot(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']],hue='Species')


# In[ ]:




