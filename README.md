# Sentiment-analysis with insights using Natural Language Processing
### This is an AI based application which helps in understanding human sentiments from product reviews.
This project has been developed based on a requirement to build a dashboard where the user can analyze the reviews from e-commerce portal and perform sentiment analysis/prediction on it.

Prerequisites:
- Python3
- Amazon Dataset (3.6GB)
- Anaconda

# Solution Approach
-	Collect and Pre-Process the training data (Amazon dataset)
-	ML Life Cycle (data cleaning, EDA, modelling)
-	Web Scraping for testing/validation data
-	Building Dashboard UI and deploying the model

# Stage 1
-	Since the client didn’t have a big enough data set, we are Using Amazon Review Dataset (2018) of 14 GB size to train the model. It’s available in the public domain.
-	In the dataset, there is a column against each review and its values ranges from 1 to 5 which is our target variable
-	Loaded the dataset as chunks of data with 10 lakh record in each chunk from a total of 3 crore 33 lakhs records
-	Extracted 750K sample reviews from the population with an equal distribution of ratings ranging from 1 to 5
-	Then we considered reviews with rating less than 3 as –ve review and greater than 3 as positive review
-	Converted that into CSV using pandas

# Stage 2 - ML Life cycle (data cleaning, EDA, modelling)

Data cleaning
-	Working with pandas dataframes
-	Basic data cleaning steps like removing null values and dropping unwanted.
-	Further processing the dataset using various NLP techniques like:

Performed NLP techniques:
-	Using regular expression to clean the text
-	Removing the stop words
-	Lemmatization of words
-	Vectorization of words using (to convert words into numerical representations)			
  i.  CountVectorizer
  ### What is CountVectorizer?
  CountVectorizer is a great tool provided by the scikit-learn library in Python. It is used to transform a given text into a vector on the basis of the frequency (count) of each   word that occurs in the entire text. This is helpful when we have multiple such texts, and we wish to convert each word in each text into vectors (for using in further text       analysis). CountVectorizer creates a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix.     The value of each cell is nothing but the count of the word in that particular text sample.

  ii. TfidfTransformer
  ### What is TF-IDF Vectorizer?
  TF-IDF stands for Term Frequency - Inverse Document Frequency and is a statistic that aims to better define how important a word is for a document, while also taking into         account the relation to other documents from the same corpus.
  This is performed by looking at how many times a word appears into a document while also paying attention to how many times the same word appears in other documents in the         corpus.
  The rationale behind this is the following:
  * A word that frequently appears in a document has more relevancy for that document, meaning that there is higher probability that the document is about or in relation to that       specific word. 
  * A word that frequently appears in more documents may prevent us from finding the right document in a collection; the word is relevant either for all documents or   for none.       Either way, it will not help us filter out a single document or a small subset of documents from the whole set. 
  So then TF-IDF is a score which is applied to every word in every document in our dataset. And for every word, the TF-IDF value increases with every appearance of the word   in   a document, but is gradually decreased with every appearance in other documents.
  
-	After checking both countvectorizer and TFidf model. We found that TFIDF performs better with the final model.


EDA
-	At this point, I am plotting some basic graphs like  bar chart & pie chart to understand the distribution of the target variable (rating)
-	Ensuring equal distribution of the target values (0 & 1) and rectifying imbalance in the dataset
-	Developing word cloud model to visualize the more frequent positive and negative words. And saving the image files to embed in the final dashboard we create

Modelling
-	Building ML model using Sklearn library
-	I have tried out 4 algorithms in sklearn
  *	RandomForestClassifier
  *	LogisticRegression
  *	SVM
  *	MultinomialNB
-	Finalized the logistic regression model as it gives better score for accuracy metric (90.3% accuracy). And using the pickle library to save the model and use it for prediction.
-	Created sklearn pipeline by assembling the workflows described earlier to simplify/automate the machine learning life cycle.
-	Imported the final model as a pickle file so that we can deploy it in UI environment

# Stage 3 - Model validation
-	To check how the model works with new data, doing web scraping of reviews from etsy.com (which is another competitor of client)
-	Used selenium and beautiful soup library to develop scraping script
-	Then saved the data set to the database, we used sqlite as DB. will retrieve it in the UI to predict the sentiment

# Stage 4 - Building UI using dash framework

### What is Plotly Dash?
Dash is a productive Python framework for building web analytic applications.

Written on top of Flask, Plotly.js, and React.js, Dash is ideal for building data visualization apps with highly custom user interfaces in pure Python. It's particularly suited for anyone who works with data in Python.

Dash apps are rendered in the web browser. You can deploy your apps to servers and then share them through URLs. Since Dash apps are viewed in the web browser, Dash is inherently cross-platform and mobile ready.

Dash is an open source library, released under the permissive MIT license. Plotly develops Dash and offers a platform for managing Dash apps in an enterprise environment.

-	In my UI there are 4 components that are
  * Pie chart
  * Word Cloud
  * Etsy predict
  * Try it yourself!


<img src="https://github.com/sreelal1/Sentiment-Analysis_NLP/blob/main/assets/app.png"/>


