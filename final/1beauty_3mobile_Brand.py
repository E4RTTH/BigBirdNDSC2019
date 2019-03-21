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

# Update stopwords database
nltk.download('stopwords')

# Functions definition --------------------------------------------------------------------------------

def preprocess_data_beauty(titles, regex):
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
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
        
        # Join the list of words back with string as seperator
        title = ' '.join(title)
        
        title = re.sub('krem', 'cream', title) 
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

def preprocess_data_mobile(titles, regex):
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
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
        
        # Join the list of words back with string as seperator
        title = ' '.join(title)
        
        # Remove all the high frequency but unrelated terms
        title = re.sub('[\S]*promo[\S]*', '', title) 
        title = re.sub('[\S]*beli[\S]*', '', title) 
        title = re.sub('[\S]*murah[\S]*', '', title) 
        title = re.sub('[\S]*hari[\S]*', '', title) 
        title = re.sub('[\S]*diskon[\S]*', '', title) 
        title = re.sub('[\S]*ini[\S]*', '', title) 
        title = re.sub('[\S]*sale[\S]*', '', title) 
        title = re.sub('[\S]*harga[\S]*', '', title) 
        
        # Append the preprocessed text back to dataset
        data.append(title)
                
    return data


def vectorize_data(vectorizer, data):
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors

def calculate_top_preds(y_classes, y_pred_proba, topNum):
    
    y_pred = []
    
    # 1. Loop through all prediction probabilities datasets
    # 2. Sort through the probability list and extract the index of top probabilities based on input argument
    # 3. Use the indices and extract the class labels 
    # 4. Format it to space separated probability list
    for probas in y_pred_proba:
        clsidx = [probas.tolist().index(x) for x in sorted(probas, reverse=True)[:topNum]]
        pred = [int(y_classes[i]) for i in clsidx]
        strPred = ' '.join(str(x) for x in pred)
        y_pred.append(strPred)
    
    return y_pred



def train_predict_data_beauty(dataset_train, dataset_val, attr_name, classifier, regex, predNum, vectorizercount):
    
    # Some declaration and initialization
    X_train = []
    X_test = []
    
    # Remove NaN entries from Benefits attribute
    dataset_train_attr = dataset_train.dropna(subset=[attr_name])
    
    # Cleaning the titles
    X_train = preprocess_data_beauty(dataset_train_attr['title'].values, regex)
    X_test = preprocess_data_beauty(dataset_val['title'].values, regex)
    
    print("Finished preprocessing beauty")
    
    # Extract results for training
    y_train = dataset_train_attr[attr_name].values
    
    # Using Count Vectorizer as features
    trainCount = len(X_train)
    X = X_train + X_test
    X = vectorize_data(CountVectorizer(max_features = vectorizercount), X)
    X_train = X[0:trainCount]
    X_test = X[trainCount:len(X)]
    
    print("Finished vectorizing beauty")
    
    # Fitting Logistic Regression one vs all
    classifier.fit(X_train, y_train)
    
    print("Finished training beauty")
    
    # Calculate the probabilities of predictions
    y_pred_proba = classifier.predict_proba(X_test)
    
    print("Finished predicting beauty")
    
    # Calculate top predictions acoording to probabilities
    y_pred = calculate_top_preds(classifier.classes_, y_pred_proba, predNum)
    
    print("Finished calculating top probabilities for beauty")
            
    return y_pred


def train_predict_data_mobile(dataset_train, dataset_val, attr_name, classifier, regex, predNum, vectorizercount):
    
    # Some declaration and initialization
    X_train = []
    X_test = []
    
    # Remove NaN entries from Benefits attribute
    dataset_train_attr = dataset_train.dropna(subset=[attr_name])
    
    # Cleaning the titles
    X_train = preprocess_data_mobile(dataset_train_attr['title'].values, regex)
    X_test = preprocess_data_mobile(dataset_val['title'].values, regex)
    
    print("Finished preprocessing mobile")
    
    # Extract results for training
    y_train = dataset_train_attr[attr_name].values
    
    # Using Count Vectorizer as features
    trainCount = len(X_train)
    X = X_train + X_test
    X = vectorize_data(CountVectorizer(max_features = vectorizercount), X)
    X_train = X[0:trainCount]
    X_test = X[trainCount:len(X)]
    
    print("Finished vectorizing mobile")
    
    # Fitting Logistic Regression one vs all
    classifier.fit(X_train, y_train)
    
    print("Finished training mobile")
    
    # Calculate the probabilities of predictions
    y_pred_proba = classifier.predict_proba(X_test)
    
    print("Finished predicting mobile")
    
    # Calculate top predictions acoording to probabilities
    y_pred = calculate_top_preds(classifier.classes_, y_pred_proba, predNum)
    
    print("Finished calculating top probabilities for mobile")
            
    return y_pred

#-------------------------------------------------------------------------------------------------------


# BEAUTY BRAND-------------------------------------------------

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
attr_name = 'Brand'

# Change to the classifier you want to use
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = 0, min_samples_split = 8)

# Change the regex term
regex = '[^a-zA-Z0-9\.]'

# Change the vectorizer count
vecCount = 5000

#-------------------------------------------------------------------------------------------------------
resultdf = resultdf[~resultdf.id.str.contains(attr_name)]

y_pred = train_predict_data_beauty(dataset_train,  \
                                   dataset_val,    \
                                   attr_name,     \
                                   classifier,  \
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
resultdf.to_csv('submission8.csv', index=False)

del resultdf, submission, y_pred, idlist, taglist






# MOBILE BRAND-------------------------------------------------

# Some declaration & initialization
idlist = []
taglist = []
predictionNum = 2

# CHANGE THIS SECTION!!---------------------------------------------------------------------------------

# Change the base result file you want to use 
resultdf = pd.read_csv('submission8.csv')

# Change to the source dataset you want to rerun
dataset_train = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('mobile_data_info_val_competition.csv', quoting = 3)

# Change to the attribute name you want to rerun
attr_name = 'Brand'

# Change to the classifier you want to use
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = 0, min_samples_split = 6)

# Change the regex term
regex = '[^a-zA-Z0-9\.]'

# Change the vectorizer count
vecCount = 10000

#-------------------------------------------------------------------------------------------------------
#resultdf = resultdf[~resultdf.id.str.contains(attr_name)]

y_pred = train_predict_data_mobile(dataset_train,  \
                                   dataset_val,    \
                                   attr_name,     \
                                   classifier,  \
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
resultdf.to_csv('submission8.csv', index=False)
