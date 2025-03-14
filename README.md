# movie-review-sentimental-analysis
sentimental analysis using Logistic Regression and Naive Bayes
#1.importing libraries

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

#printing the stopwords in english language
print(stopwords.words('english'))

#2.preprocessing

#loading the data from csv file to pandas data frame
movie_data = pd.read_csv("/content/IMDB Dataset.csv")

#printing the no of rows and columns
movie_data.shape

#printing the first five rows
movie_data.head()

# #counting the no of missing values
movie_data.isnull().sum()

#checking the distribution of sentiment column
movie_data['sentiment'].value_counts()

#stemming

port_stem = PorterStemmer()
def preprocess_and_tokenize(content):
    # Removing non-alphabet characters and converting to lowercase
    content = re.sub('[^a-zA-Z]', ' ', content).lower()
    # Splitting and removing stopwords
    tokens = [port_stem.stem(word) for word in content.split() if word not in stopwords.words('english')]
    return tokens
    # Tokenize the review
    movie_data['tokens'] = movie_data['review'].apply(preprocess_and_tokenize)
    
  # Separating the data and labels
  x = movie_data['tokens'].values
  y = movie_data['sentiment'].values

  movie_data.head()
  
  print(x)
  print(y)

  Splitting the data into training data and test data
  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
  print(x.shape,x_train.shape,x_test.shape)
  #vectorization
  import gensim
from gensim.models import Word2Vec
#5. Word2Vec Embeddings
# Training Word2Vec model on the training data
w2v_model = Word2Vec(sentences=x_train.tolist(), vector_size=100, window=5, min_count=1, sg=1)

# Function to get the average word vector for each review
def get_average_word2vec(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Creating average word vectors for each review in training and test sets
x_train_vec = np.array([get_average_word2vec(tokens, w2v_model, 100) for tokens in x_train])
x_test_vec = np.array([get_average_word2vec(tokens, w2v_model, 100) for tokens in x_test])

#Training the Machine learning model
#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train_vec,y_train)
x_train_prediction = lr.predict(x_train_vec)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
print('Accuracy score of the training data: ',training_data_accuracy*100)

#Naive Bayes
from sklearn.preprocessing import MinMaxScaler

# Assuming x_train_vec is your feature matrix with potential negative values
scaler = MinMaxScaler()
x_train_vec_scaled = scaler.fit_transform(x_train_vec)
x_test_vec_scaled = scaler.transform(x_test_vec)

# Now fit the MultinomialNB model
classifier = MultinomialNB()
classifier.fit(x_train_vec_scaled,y_train)

from sklearn.naive_bayes import MultinomialNB

x_train_prediction_nb = classifier.predict(x_train_vec_scaled)
training_datas_accuracy_nb = accuracy_score(x_train_prediction_nb,y_train)

print('Accuracy score of the training data: ',training_datas_accuracy_nb*100)
