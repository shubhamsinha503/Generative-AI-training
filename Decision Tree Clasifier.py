#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('diabetes.csv')


# In[2]:


df


# In[4]:


feature_cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x=df[feature_cols]
y=df.Outcome


# In[5]:


x


# In[6]:


y


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[10]:


classifier=DecisionTreeClassifier(criterion='gini')


# In[11]:


classifier.fit(x_train,y_train)


# In[12]:


classifier.predict(x_test)


# In[13]:


x_test


# In[15]:


from sklearn import tree
tree.plot_tree(classifier)


# In[16]:


classifier=DecisionTreeClassifier(criterion='entropy')


# In[18]:


classifier.fit(x_train,y_train)


# In[19]:


tree.plot_tree(classifier)


# In[ ]:




