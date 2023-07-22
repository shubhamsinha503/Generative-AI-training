#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Position_Salaries.csv')


# In[3]:


df


# In[7]:


x=df.iloc[:,1:2].values


# In[5]:


x


# In[8]:


x


# In[9]:


y=df.iloc[:,2].values


# In[10]:


y


# In[11]:


plt.scatter(x,y)


# In[13]:


sns.lmplot(x='Level',y='Salary',data=df)


# In[16]:


from sklearn import linear_model


# In[19]:


reg=linear_model.LinearRegression()


# In[20]:


reg.fit(x,y)


# In[21]:


reg.predict([[6.5]])


# In[22]:


from sklearn.preprocessing import PolynomialFeatures


# In[23]:


poly=PolynomialFeatures(degree=2)


# In[24]:


x_poly=poly.fit_transform(x)


# In[25]:


reg2=linear_model.LinearRegression()


# In[30]:


reg2.fit(x_poly,y)


# In[31]:


reg2.predict(poly.fit_transform([[6.5]]))


# In[ ]:




