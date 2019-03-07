import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import logging


def preprocess_data(titles):
    ps = PorterStemmer()
    data = []
    for item in titles:
        title = re.sub('[^a-zA-Z0-9]', ' ', item)
        title = title.lower()
        title = title.split()
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
       # title = ' '.join(title)
        data.append(title)
        
    return data


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.items())))
        #next(iter(graph.items()))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
            
            
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.items())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# Update stopwords database
nltk.download('stopwords')


 # Some declaration and initialization
X = []
attr_name = 'Camera'

dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
    
# Remove NaN entries from Benefits attribute
dataset_attr = dataset.dropna(subset=[attr_name])

titles = dataset_attr['title'].values
    
# Cleaning the titles
X = preprocess_data(dataset_attr['title'].values)
y = dataset_attr[attr_name].values
    
model = Word2Vec(
        X,
        size=500,
       # window=10,
        workers=10)
model.train(X, total_examples=model.corpus_count, epochs=10)

#words = list(model.wv.vocab)
#words.index("huangmi")

w2v = dict(zip(model.wv.index2word, model.wv.syn0))

vectorizer = TfidfEmbeddingVectorizer(w2v)
vectorizer.fit(X, y)
X = vectorizer.transform(X)





"""
#cv = CountVectorizer(max_features = 5000)
#X = cv.fit_transform(X).toarray()

#w1 = "huangmi"
#model.wv.most_similar(positive=w1)
"""
#X = [i for i in X if len(i) != 500]
xresult = []
yresult = []
for idx, i in enumerate(X):
    if len(i) == 500:
        xresult.append(i)
        yresult.append(y[idx])


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(xresult, yresult, test_size = 0.20, random_state = 0)
    
classifier = MLPClassifier(alpha=0.01, hidden_layer_sizes=500)

classifier.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred = classifier.predict(X_test)
    
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
    
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
    
# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')



