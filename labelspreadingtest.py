# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

def preprocess_data(titles, regex):
    ps = PorterStemmer()
    data = []
    for item in titles:
        
        title = item

        title = re.sub(regex, ' ', title)
        title = title.lower()
        title = title.split()
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
        title = ' '.join(title)
        
        #mobile test filter
        title = re.sub('[\S]*promo[\S]*', '', title) 
        title = re.sub('[\S]*beli[\S]*', '', title) 
        title = re.sub('[\S]*murah[\S]*', '', title) 
        title = re.sub('[\S]*hari[\S]*', '', title) 
        title = re.sub('[\S]*diskon[\S]*', '', title) 
        title = re.sub('[\S]*ini[\S]*', '', title) 
        title = re.sub('[\S]*sale[\S]*', '', title) 
        title = re.sub('[\S]*harga[\S]*', '', title) 
        
        data.append(title)
        
    del titles
        
    return data

def vectorize_data(vectorizer, data):
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors


# Importing the dataset
dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)

attr_name = 'Camera'
keepcol = ['itemid', 'title', attr_name]
dataset = dataset[keepcol]

regex = '[^a-zA-Z0-9\.]'

indices = np.arange(len(dataset))
datasetWithAttr = dataset.dropna(subset=[attr_name])
datasetWithoutAttr = dataset[dataset[attr_name].isnull()]

X_title = np.vstack((datasetWithAttr['title'].values.reshape(-1, 1), datasetWithoutAttr['title'].values.reshape(-1, 1)))

# Cleaning the titles
X_title = preprocess_data(X_title, regex)
    
# Using Count Vectorizer as features
X = vectorize_data(CountVectorizer(max_features = 2000), X_title)

y = np.vstack((datasetWithAttr[attr_name].values.reshape(-1, 1), np.full((len(datasetWithoutAttr),1), -1)))

unlabeled_set = indices[len(datasetWithAttr):]

lp_model = LabelSpreading(max_iter=5, kernel='knn', n_neighbors = 10)
lp_model.fit(X, y)
predicted_labels = lp_model.transduction_[unlabeled_set]
