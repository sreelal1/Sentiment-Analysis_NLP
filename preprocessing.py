#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd 
import numpy as np
import re
import matplotlib.pyplot as plt
from textblob import Word
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pickle 
from sklearn.pipeline import Pipeline


# In[2]:


path = '/media/sreelal/CA24611524610633/FORSK Internship/Clothing_Shoes_and_Jewelry.json.gz'


# In[3]:


print(os.path.isfile(path))


# In[ ]:


df_reader = pd.read_json(path, lines=True, chunksize=1000000, compression='gzip')
counter = 1
for chunk in df_reader:
    new_df = pd.DataFrame(chunk[['overall','reviewText','summary']])
    new_df1 = new_df[new_df['overall'] == 5].sample(4000)
    new_df2 = new_df[new_df['overall'] == 4].sample(4000)
    new_df3 = new_df[new_df['overall'] == 3].sample(8000)
    new_df4 = new_df[new_df['overall'] == 2].sample(4000)
    new_df5 = new_df[new_df['overall'] == 1].sample(4000)
    
    new_df6 = pd.concat([new_df1,new_df2,new_df3,new_df4,new_df5], axis = 0, ignore_index = True)
    
    new_df6.to_csv(str(counter)+".csv", index = False)
    
    new_df = None
    counter = counter + 1


# In[ ]:


from glob import glob
filenames = glob('*.csv')
dataframes = [pd.read_csv(f) for f in filenames]
frame = pd.concat(dataframes, axis = 0, ignore_index = True)
frame.to_csv('/media/sreelal/CA24611524610633/FORSK Internship/balanced_review.csv', index = False)


# In[1]:


import pandas as pd

df = pd.read_csv('/media/sreelal/CA24611524610633/FORSK Internship/balanced_review.csv')


# In[6]:


df.head()


# In[2]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe(include='all')


# In[7]:


#dropping summary column
df=df.drop('summary',1)


# In[8]:


df.columns=['overall','review']


# In[9]:


df.isna().sum()


# In[10]:


df.dropna(subset=['review'],inplace=True)


# In[11]:


df.isna().sum()


# In[12]:


df.head()


# In[13]:


df['overall'].value_counts()


# In[14]:


ax = df['overall'].value_counts(sort=False).plot(kind='barh')
ax.set_xlabel('Number of Samples in training Set')
ax.set_ylabel('Label')


# In[15]:


# Adding new column rating, with review ratings > 3 are labelled 1 (positive review) and review ratings < 3 are labelled 0 (negative review)  
df['rating']=np.where(df['overall']>=3,1,0)
df


# In[16]:


#dropping overall column
df=df.drop('overall',1)
df


# In[17]:


df['rating'].value_counts()


# In[18]:


ax = df['rating'].value_counts(sort=False).plot(kind='bar')
ax.set_ylabel('Number of Samples in training Set')
ax.set_xlabel('Label')


# In[20]:


# Dealing with imbalance in the dataset
dfn=df[df['rating']==0]
dfp=df[df['rating']==1]
df=None
dfp=dfp.iloc[:263865,:]
df=pd.concat([dfp,dfn])
df['rating'].value_counts()


# In[21]:


ax = df['rating'].value_counts(sort=False).plot(kind='bar')
ax.set_ylabel('Number of Samples in training Set')
ax.set_xlabel('Label')


# In[22]:


df.to_csv('cleaned_balanced_reviews.csv',index=False)


# In[23]:


def clean_reviews(review):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",review.lower()).split())


# In[24]:


df['review']=df['review'].apply(clean_reviews)


# In[25]:


df


# In[26]:


stop = stopwords.words('english')
def remove_stopwords(df):
    df['review']=df['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[27]:


remove_stopwords(df)
df


# In[28]:


def lemmatization(df):
    df['review']=df['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))    


# In[29]:


lemmatization(df)
df


# In[30]:


def wordcloud(df,filename):
    ratings=''.join(df['review'])    
    wordcloud = WordCloud().generate(ratings)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    wordcloud.to_file('{}.png'.format(filename))


# In[35]:


plt.title("Popular review words!", fontsize=20)
wordcloud(df,"wholeword.png")


# In[33]:


df_pos=df[df['rating']==1]
plt.title("Positive review words!", fontsize=25)
wordcloud(df_pos,"posword.png")


# In[34]:


df_neg=df[df['rating']==0]
plt.title("Negative review words!", fontsize=25)
wordcloud(df_neg,"negword.png")


# In[41]:


df.isna().sum()


# In[42]:


df.dropna(subset=['review'],inplace=True)


# In[43]:


df.isna().sum()


# In[44]:


df.to_csv('cleaned_balanced_reviews_final.csv',index=False)

