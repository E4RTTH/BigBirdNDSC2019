# Big Bird - NDSC 2019

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


# Functions definition --------------------------------------------------------------------------------

def preprocess_data(titles):
    ps = PorterStemmer()
    data = []
    for i in range(0, len(titles)):
        title = re.sub('[^a-zA-Z]', ' ', titles[i])
        title = title.lower()
        title = title.split()
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
        title = ' '.join(title)
        data.append(title)
        
    del titles
        
    return data



def vectorize_data(vectorizer, data):
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors



def train_predict_data(dataset, attr_name, classifier):
    
    # Some declaration and initialization
    X = []
    
    # Remove NaN entries from Benefits attribute
    dataset_attr = dataset.dropna(subset=[attr_name])
    
    # Cleaning the titles
    X = preprocess_data(dataset_attr['title'].values)
    
    # Using Count Vectorizer as features
    X = vectorize_data(CountVectorizer(max_features = 10000), X)
    
    # Using TF-IDF Vectorizer (Word level) as features
    #X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000), X)
    
    # Using TF-IDF Vectorizer (NGram level) as features
    #X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000), X)
    
    y = dataset_attr[attr_name].values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    
    del X, y
    
    # Fitting Logistic Regression one vs all
    #classifier = LogisticRegression(random_state = 0, multi_class = 'ovr')
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    del X_train, X_test, y_train, y_test, y_pred
        
    return cm, accuracy, f1

#-------------------------------------------------------------------------------------------------------



# Importing the dataset
dataset = pd.read_csv('fashion_data_info_train_competition.csv', quoting = 3)

# Update stopwords database
nltk.download('stopwords')

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

cm_Pattern, acc_Pattern, f1_Pattern = train_predict_data(dataset, 'Pattern', classifier)
print("Pattern: acc=", acc_Pattern, ", f1=", f1_Pattern)

cm_Collar, acc_Collar, f1_Collar = train_predict_data(dataset, 'Collar Type', classifier)
print("Collar Type: acc=", acc_Collar, ", f1=", f1_Collar)

cm_Sleeves, acc_Sleeves, f1_Sleeves = train_predict_data(dataset, 'Sleeves', classifier)
print("Sleeves: acc=", acc_Sleeves, ", f1=", f1_Sleeves)

cm_Trend, acc_Trend, f1_Trend = train_predict_data(dataset, 'Fashion Trend', classifier)
print("Fashion Trend: acc=", acc_Trend, ", f1=", f1_Trend)

cm_Material, acc_Material, f1_Material = train_predict_data(dataset, 'Clothing Material', classifier)
print("Clothing Material: acc=", acc_Material, ", f1=", f1_Material)


