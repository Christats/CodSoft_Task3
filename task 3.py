#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[9]:


iris = 'IRIS.csv'


# In[10]:


import pandas as pd

# Assuming iris is a variable containing the loaded DataFrame
iris = pd.read_csv('IRIS.csv')  # Adjust this line based on how you loaded the dataset

# Now you can use the describe method
iris_description = iris.describe()
print(iris_description)


# In[11]:


iris.shape


# In[12]:


iris


# In[13]:


iris.groupby('species').mean()


# In[14]:


sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.show()


# In[15]:


sns.lineplot(data=iris.drop(['species'], axis=1))
plt.show()


# In[16]:


iris.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
plt.show()


# In[17]:


sns.heatmap(iris.corr(), annot=True)
plt.show()


# In[18]:


g = sns.FacetGrid(iris, col='species')
g = g.map(sns.kdeplot, 'sepal_length')


# In[19]:


sns.pairplot(iris)


# In[20]:


iris.hist(color= 'mediumpurple' ,edgecolor='black',figsize=(10,10))
plt.show()


# In[21]:


iris.corr().style.background_gradient(cmap='coolwarm').set_precision(2)


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[23]:


x = iris.drop('species', axis=1)
y= iris.species

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(x_train, y_train)

knn.score(x_test, y_test)


# In[25]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x, y)
y_pred = logreg.predict(x)
print(metrics.accuracy_score(y, y_pred))


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

dtree.score(x_test, y_test)


# In[ ]:




