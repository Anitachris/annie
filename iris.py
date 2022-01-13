#!/usr/bin/env python
# coding: utf-8

# # LOADING THE DATA SET

# In[1]:


import pandas as pd
df=pd.read_excel(r'C:\Users\shine\Downloads\iris.xls')


# In[2]:


df


# In[3]:


df.describe()


# In[4]:


df.info()


# # CHECKING FOR NULL VALUES

# In[5]:


df.isna().sum()


# In[6]:


df.shape


# In[7]:


import matplotlib.pyplot as plt
freqgraph=df.select_dtypes(include=['float'])
freqgraph.hist(figsize=(20,15))
plt.show()


# In[8]:


import seaborn as sns
plt.figure(figsize=(7,4)) 
sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r') 
plt.show()


# petal width and length are highly correlated
# sepal length and awidth are not correlated

# In[9]:


df[df['SL'].isnull()].index.tolist()


# In[10]:


df[df['SW'].isnull()].index.tolist()


# In[11]:


df[df['PL'].isnull()].index.tolist()


# In[12]:


df.Classification.nunique()


# # FILLING THE NULL VALUES USING THE MEAN

# In[13]:


df['SL']=df['SL'].fillna(5.856)
df['SW']=df['SW'].fillna(3.05)
df['PL']=df['PL'].fillna(3.756)


# In[14]:


df.isna().sum()


# In[15]:


df['Classification'].value_counts


# # LABEL ENCODING THE CLASSIFICATION COLUMN

# In[16]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
df['Classification']=label_encoder.fit_transform(df['Classification'])
df


# # SPLITTING THE DATASET

# In[18]:



train_x = df[['SL','SW','PL','PW']]
train_y=df['Classification']
test_x= df[['SL','SW','PL','PW']]
test_y =df['Classification']


# In[20]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(train_x,train_y,test_size=0.3,random_state=2)


# # LOGISTIC REGRESSION

# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model=LogisticRegression()
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('The accuracy of the logistic regression is',metrics.accuracy_score(prediction,test_y))










# # KNN

# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('The accuracy of the KNN  is',metrics.accuracy_score(prediction,test_y))



# # DECISION TREE

# In[25]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('The accuracy of decision tree classifier  is',metrics.accuracy_score(prediction,test_y))



# # RANDOM FOREST

# In[27]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(train_x,train_y)
y_pred=rf.predict(test_x)
print('The accouracy of Random forets classifier is',metrics.accuracy_score(y_pred,test_y))


# In[ ]:




