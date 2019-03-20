# Big Bird - NDSC 2019

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Update stopwords database
nltk.download('stopwords')
stopwords_factory = StopWordRemoverFactory()
stopwords_id = stopwords_factory.create_stop_word_remover()
stemmer_factory = StemmerFactory()
stemmer_id = stemmer_factory.create_stemmer()

# Functions definition --------------------------------------------------------------------------------

def preprocess_data(titles, regex):
    # Porter stemmer for word stemming
    ps = PorterStemmer()
    data = []
    for item in titles:
        # Replace regex term into space (non letters & non numbers)
        title = re.sub(regex, ' ', item)
        
        # Replace term to lowercase
        title = title.lower()
        
        # Split the term to list
        title = title.split()
        
        # Remove stopwords
        title = [stopwords_id.remove(word) for word in title]
        title = [stemmer_id.stem(word) for word in title]
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
        
        # Join the list of words back with string as seperator
        title = ' '.join(title)
        
        title = re.sub('krem', 'cream', title) 
        title = re.sub('spf', ' spf ', title) 
        title = re.sub('(?<=(natur)) (?=(republ))', '', title)
        title = re.sub('[\S]*promo[\S]*', '', title) 
        title = re.sub('[\S]*murah[\S]*', '', title) 
        title = re.sub('[\S]*new[\S]*', '', title) 
        title = re.sub('[\S]*diskon[\S]*', '', title) 
        title = re.sub('[\S]*best[\S]*', '', title) 
        title = re.sub('[\S]*sale[\S]*', '', title) 
        
        # Append the preprocessed text back to dataset
        data.append(title)
                
    return data



def vectorize_data(vectorizer, data):
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors

def calculate_top_preds(y_classes1, y_pred_proba1, y_classes2, y_pred_proba2):
    
    y_pred = []
    
    for i in range(len(y_pred_proba1)):
        probas = np.hstack((y_pred_proba1[i], y_pred_proba2[i]))
        classes = np.hstack((y_classes1, y_classes2))
        top1 = -1
        top2 = -1
        top1proba = -1
        top2proba = -1
        for i, prob in enumerate(probas):
            if prob > top1proba:
                top1proba = prob
                top1 = classes[i]

        for i, prob in enumerate(probas):
            if classes[i] == top1:
                continue
            if prob > top2proba:
                top2proba = prob
                top2 = classes[i]
        
        strPred = '{} {}'.format(int(top1), int(top2))
        y_pred.append(strPred)
    
    return y_pred



def train_predict_data(dataset_train, dataset_val, attr_name, classifier1, classifier2, regex, predNum, vectorizercount):
    
    # Some declaration and initialization
    X_train = []
    X_test = []
    
    # Remove NaN entries from Benefits attribute
    dataset_train_attr = dataset_train.dropna(subset=[attr_name])
    
    # Cleaning the titles
    X_train = preprocess_data(dataset_train_attr['title'].values, regex)
    X_test = preprocess_data(dataset_val['title'].values, regex)
    
    # Extract results for training
    y_train = dataset_train_attr[attr_name].values
    
    # Using Count Vectorizer as features
    trainCount = len(X_train)
    X = X_train + X_test
    X = vectorize_data(CountVectorizer(max_features = vectorizercount), X)
    X_train = X[0:trainCount]
    X_test = X[trainCount:len(X)]
    
    calibrator1 = CalibratedClassifierCV(classifier1, cv=8)
    calibrator2 = CalibratedClassifierCV(classifier2, cv=8)
    
    # Fitting Classifiers
    calibrator1.fit(X_train, y_train)
    calibrator2.fit(X_train, y_train)
    
    # Calculate the probabilities of predictions
    y_pred_proba1 = calibrator1.predict_proba(X_test)
    y_pred_proba2 = calibrator2.predict_proba(X_test)
    
    # Calculate top predictions acoording to probabilities
    y_pred = calculate_top_preds(calibrator1.classes_, y_pred_proba1, calibrator2.classes_, y_pred_proba2)
            
    return y_pred


#-------------------------------------------------------------------------------------------------------


# Some declaration & initialization
idlist = []
taglist = []
predictionNum = 2

# CHANGE THIS SECTION!!---------------------------------------------------------------------------------

# Change the base result file you want to use 
resultdf = pd.read_csv('submission8.csv')

# Change to the source dataset you want to rerun
dataset_train = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('beauty_data_info_val_competition.csv', quoting = 3)

# Change to the attribute name you want to rerun
attr_name = 'Benefits'

# Change to the classifier you want to use
classifier1 = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 0, min_samples_split = 6)
classifier2 = LogisticRegression(random_state = 0, multi_class = 'ovr', solver = 'lbfgs')

# Change the regex term
regex = '[^a-zA-Z0-9]'

# Change the vectorizer count
vecCount = 5000

#-------------------------------------------------------------------------------------------------------
resultdf = resultdf[~resultdf.id.str.contains(attr_name)]

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            attr_name,     \
                            classifier1,  \
                            classifier2,  \
                            regex, \
                            predictionNum, \
                            vecCount)

print ('Finish predicting')

for i, row in dataset_val.iterrows():
    itemid = "{}_{}".format(row['itemid'], attr_name)
    idlist.append(itemid)
    taglist.append(y_pred[i])

print ('Finish writing to list')

#-------------------------------------------------------------------------------------------------------
submission = pd.DataFrame(data = {'id': idlist, 'tagging': taglist})
resultdf = resultdf.append(submission)
resultdf.to_csv('submission9.csv', index=False)
