# Natural Language Processing

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
from sklearn.calibration import CalibratedClassifierCV


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

        title = re.sub('krem', 'cream', title) 
        title = re.sub('(?<=(natur)) (?=(republ))', '', title)
        title = re.sub('[\S]*promo[\S]*', '', title) 
        title = re.sub('[\S]*murah[\S]*', '', title) 
        title = re.sub('[\S]*new[\S]*', '', title) 
        title = re.sub('[\S]*diskon[\S]*', '', title) 
        title = re.sub('[\S]*best[\S]*', '', title) 
        title = re.sub('[\S]*sale[\S]*', '', title) 
        
        data.append(title)
        
    del titles
        
    return data

def vectorize_data(vectorizer, data):
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors



# Importing the dataset
dataset = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)

# Update stopwords database
nltk.download('stopwords')

classifier1 = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 7, min_samples_split = 6)
classifier2 = LogisticRegression(random_state = 0, multi_class = 'ovr', solver = 'lbfgs')
#, max_depth=130

attr_name = 'Benefits'
regex = '[^a-zA-Z0-9\.]'

# Some declaration and initialization
X = []
    
# Remove NaN entries from attribute
dataset_attr = dataset.dropna(subset=[attr_name])
    
# Cleaning the titles
X_title = preprocess_data(dataset_attr['title'].values, regex)
    
# Using Count Vectorizer as features
X = vectorize_data(CountVectorizer(max_features = 5000), X_title)

y = dataset_attr[attr_name].values
    
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)
X_title_train, X_title_test = train_test_split(X_title, test_size = 0.20, random_state = 5)
    
del X, y, X_title_train


calibrator1 = CalibratedClassifierCV(classifier1, cv=10)
calibrator2 = CalibratedClassifierCV(classifier2, cv=10)

calibrator1.fit(X_train, y_train)
calibrator2.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred_proba1 = calibrator1.predict_proba(X_test)
y_pred_proba2 = calibrator2.predict_proba(X_test)


"""
classifier1.fit(X_train, y_train)
classifier2.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred_proba1 = classifier1.predict_proba(X_test)
y_pred_proba2 = classifier2.predict_proba(X_test)
"""
y_classes1 = calibrator1.classes_
y_classes2 = calibrator2.classes_

y_pred = []

for i in range(len(y_pred_proba1)):
    probas = np.hstack((y_pred_proba1[i], y_pred_proba2[i]))
    classes = np.hstack((y_classes1, y_classes2))
    top1 = -1
    topproba = -1
    for i, prob in enumerate(probas):
        if prob > topproba:
            topproba = prob
            top1 = classes[i]
    y_pred.append(top1)
    
    
    
    
    
    
    
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
    
# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
    
