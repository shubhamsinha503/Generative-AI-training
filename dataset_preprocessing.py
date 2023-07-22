#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd


# In[70]:


data=pd.read_csv('Salary_Data.csv')


# In[71]:


x=data[['Age','Years of Experience',]].values


# In[72]:


#x=data[['Age','Gender','Years of Experience',]]


# In[73]:


y=data[['Salary']].values


# In[74]:


z=data[['Gender']]


# In[75]:


z


# In[76]:


x


# In[77]:


y


# In[78]:


from sklearn import preprocessing


# In[79]:


#onehotencoder= preprocessing.OneHotEncoder()


# In[80]:


#z=onehotencoder.fit_transform(data.Gender.values.reshape(-1, 1))


# In[81]:


encoder = preprocessing.LabelEncoder()


# In[82]:


gender_encoded = encoder.fit_transform(z)


# In[83]:


encoded_z=pd.DataFrame({'Gender':gender_encoded})


# In[ ]:





# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


# In[86]:


encoded_z_train,encoded_z_test=train_test_split(encoded_z,test_size=0.4,random_state=0)


# In[87]:


x_train


# In[88]:


y_train


# In[89]:


encoded_z_train


# In[90]:


x_test


# In[91]:


y_test


# In[92]:


encoded_z_test


# In[93]:


from sklearn.preprocessing import StandardScaler


# In[94]:


sc=StandardScaler()


# In[95]:


x_train=sc.fit_transform(x_train)


# In[96]:


x_test=sc.fit_transform(x_test)


# In[97]:


x_train


# In[98]:


x_test


# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[3]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# In[4]:


ax.scatter(x, y,, c=color, cmap='viridis')


# In[ ]:




