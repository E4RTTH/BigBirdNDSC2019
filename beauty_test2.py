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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Functions definition --------------------------------------------------------------------------------

def preprocess_data(titles):
    ps = PorterStemmer()
    data = []
    for item in titles:
        
        title = item
        title = re.sub('[^a-zA-Z0-9\.]', ' ', item)
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
    
    print("Finish preprocessing ", attr_name)
    
    # Using Count Vectorizer as features
    X = vectorize_data(CountVectorizer(max_features = 5000), X)
    
    print("Finish vectorizing ", attr_name)
    
    # Using TF-IDF Vectorizer (Word level) as features
    #X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000), X)
    
    # Using TF-IDF Vectorizer (NGram level) as features
    #X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000), X)
    
    y = dataset_attr[attr_name].values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    
    del X, y
    
    print("Finish spliting train and test sets for  ", attr_name)
    
    # Fitting Logistic Regression one vs all
    classifier.fit(X_train, y_train)
    
    print("Finish training for  ", attr_name)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    print("Finish predicting for  ", attr_name)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
        
    return accuracy, f1

#-------------------------------------------------------------------------------------------------------




# Models definition ------------------------------------------------------------------------------------
"""
classifierNames = ["Logistic regression OvR",
                   "Logistic regression Multinomial",
                   "Nearest Neighbors", 
                   "Linear SVM", 
                   "Linear SVM ovr", 
                   "Gaussian Process",
                   "Decision Tree", 
                   "Random Forest", 
                   "Neural Net", 
                   "AdaBoost",
                   "Naive Bayes", 
                   "QDA" ]

classifiers = [
        LogisticRegression(random_state = 0, multi_class = 'ovr', solver = 'lbfgs'),
        LogisticRegression(random_state = 0, multi_class = 'multinomial', solver = 'lbfgs'),
        KNeighborsClassifier(3),
        LinearSVC(random_state=0, tol=1e-5, multi_class='crammer_singer'),
        LinearSVC(random_state=0, tol=1e-5, multi_class='ovr'),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis() ]
"""
depths = [100,110,120,130,140,150,160]


#-------------------------------------------------------------------------------------------------------

# Attribute definition ---------------------------------------------------------------------------------

attributes = ["Benefits",
              "Colour_group",
              "Brand",
              "Product_texture",
              "Skin_type" ]

#-------------------------------------------------------------------------------------------------------

# Importing the dataset
dataset = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)

# Update stopwords database
nltk.download('stopwords')

val_list = []

for depth in depths:
    
    for attr in attributes:
        classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 0, min_samples_split = 6, max_depth = depth)
        print("Start running ", attr," with depth ", depth)
        acc, f1 = train_predict_data(dataset, attr, classifier)
        print(attr, ": depth=", depth," acc=", acc, ", f1=", f1)
        valitem = [depth, attr, acc, f1]
        val_list.append(valitem)


