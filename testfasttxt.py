from gensim.models.wrappers import FastText
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def preprocess_data(titles):
    data = []
    for item in titles:
        title = item
        title = re.sub('[^a-zA-Z0-9\.]', ' ', title)
        title = title.lower()
        title = re.sub('[\S]*promo[\S]*', '', title) 
        title = re.sub('[\S]*beli[\S]*', '', title) 
        title = re.sub('[\S]*murah[\S]*', '', title) 
        title = re.sub('[\S]*hari[\S]*', '', title) 
        title = re.sub('[\S]*diskon[\S]*', '', title) 
        title = re.sub('[\S]*ini[\S]*', '', title) 
        title = re.sub('[\S]*sale[\S]*', '', title) 
        title = re.sub('[\S]*harga[\S]*', '', title) 
        title = title.split()
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


model = FastText.load_fasttext_format('cc.id.300.bin')



attr_name = 'Camera'

dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
    
# Remove NaN entries from Benefits attribute
dataset_attr = dataset.dropna(subset=[attr_name])

X = preprocess_data(dataset_attr['title'].values)
y = dataset_attr[attr_name].values

w2v = dict(zip(model.wv.index2word, model.wv.syn0))
vectorizer = TfidfEmbeddingVectorizer(w2v)
vectorizer.fit(X, y)
X = vectorizer.transform(X)


xresult = []
yresult = []
for idx, i in enumerate(X):
    if len(i) == 300:
        xresult.append(i)
        yresult.append(y[idx])



X_train, X_test, y_train, y_test = train_test_split(xresult, yresult, test_size = 0.20, random_state = 0)
    
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 0, min_samples_split = 6)
#RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 0, min_samples_split = 6)
#MLPClassifier(alpha=0.01, hidden_layer_sizes=500)

classifier.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred = classifier.predict(X_test)
    
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
    
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
    
# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')