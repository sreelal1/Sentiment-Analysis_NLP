#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pickle 
from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv('cleaned_balanced_reviews_final.csv')
df


# In[5]:


features_train, features_test, labels_train, labels_test = train_test_split(df["review"], df["rating"],random_state = 23)


# In[6]:


print('Training data shape is {}'.format(features_train.shape))
print('Training labels shape is {}'.format(labels_train.shape))
print('Testing data  shape is {}'.format(features_test.shape))
print('Testing labels  shape is {}'.format(labels_test.shape))


# ### Creating a model using count vectorizer

# In[7]:


vect = CountVectorizer().fit(features_train)


# In[8]:


len(vect.get_feature_names())  #the length ie the no: of columns are around 65 thousand


# In[9]:


#just checkin the length of features ie the no: of columns we have after vectorization
len(vect.get_feature_names())

vect.get_feature_names()[15000:15010]


# In[10]:


features_train_vectorized = vect.transform(features_train)


# In[ ]:



#prepare the model
#KNN, SVM, Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Xgboost

#version 01
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)


predictions = model.predict(vect.transform(features_test))


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)


# ### Creating TFIDF model

# In[11]:


vect = TfidfVectorizer(min_df = 5).fit(features_train)


# In[12]:


#the length ie the no: of columns with count vectorized data are around 60K
features_train_vectorized = vect.transform(features_train)


# In[13]:


#no: of cols/features reduced to 16788 which was earlier 60K with count vectorizer
features_train_vectorized.shape


# In[ ]:


# again create model and predict using log regrssion using the TFiDf data
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)


# In[ ]:


predictions = model.predict(vect.transform(features_test))


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)


# ## as we see above we can see an imrovement from 89% to 90% with tfidf approach

# # importing the model as a pickle file

# In[ ]:


import pickle


# In[ ]:


# creating an empty pickle file to write the model
pkl_filename = "pickle_model.pkl"


# In[ ]:


with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)


# In[ ]:


# to use the file in the client machine, below code has to run in client machine

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


# In[ ]:


#then running the pickle model same as what we did.

#this is also called deserialization of the model
pred = pickle_model.predict(vect.transform(features_test))
roc_auc_score(labels_test, pred)

