#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('gender_submission.csv')


# In[3]:


df


# In[4]:


x=df.iloc[:,1:2].values


# In[5]:


x


# In[6]:


y=df.iloc[:,0].values


# In[7]:


y


# In[8]:


plt.scatter(x,y)


# In[11]:


from sklearn.model_selection import train_test_split

# Your code using train_test_split


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(df[['PassengerId']],df['Survived'],test_size=0.2)


# In[14]:


len(x_train)


# In[15]:


len(x_test)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


lr=LogisticRegression()


# In[18]:


lr.fit(x_train,y_train)


# In[19]:


lr.predict(x_test)


# In[ ]:




