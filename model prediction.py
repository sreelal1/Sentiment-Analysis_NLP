#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df1 = pd.read_csv('scrappedReviews.csv')


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


import pickle


file = open("pickle_model.pkl", 'rb') 
pickle_model = pickle.load(file)


file = open("feature.pkl", 'rb')
vocab = pickle.load(file)


def check_review(reviewText):
    
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)  
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(vectorised_review)


sentiment = []
predictedvalue = []
for i in range (len(df1['reviews'])):
    response = check_review(df1['reviews'][i])
    if (response[0]==1):
        predictedvalue.append(response[0])
        sentiment.append('Positive')
    elif (response[0] ==0 ):
        predictedvalue.append(response[0])
        sentiment.append('Negative')
    else:
        sentiment.append('Unknown')


# In[14]:


len(sentiment)


# In[15]:


df1['sentiment'] = sentiment
df1['predictedvalue'] = predictedvalue


# In[16]:


df1


# In[18]:


df1.drop(df1.columns[[0]], axis = 1, inplace = True)
df1.to_csv('etsy_predicted.csv', index = False)


# In[ ]:




