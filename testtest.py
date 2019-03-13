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
from sklearn.linear_model.stochastic_gradient import SGDClassifier

def preprocess_data(titles, regex):
    ps = PorterStemmer()
    data = []
    for item in titles:
        
        title = item
        
        # Remove all the spaces between numbers and keyterm
        title = re.sub('(?<=(\d)) (?=(g ))', '', title) 
        title = re.sub('(?<=(\d)) (?=(gb|mb))', '', title)
        title = re.sub('(?<=(\d)) (?=(mp))', '', title)   
        title = re.sub('tahun|thn|yr', 'year', title)   
        title = re.sub('bulan|bln|mth', 'month', title)
        title = re.sub('(?<=(\d)) (?=(year|month|))', '', title)
        title = re.sub('(?<=(\d)) (?=(inch))', '', item) 
        title = re.sub('[\S]*(gb|mb|mp|year|month)', '', item) 
        
        title = re.sub(regex, ' ', item)
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





# Importing the dataset
dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)

# Update stopwords database
nltk.download('stopwords')

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0, min_samples_split = 6)
#, max_depth=130

attr_name = 'Phone Screen Size'
regex = '[^a-zA-Z0-9\.]'

# Some declaration and initialization
X = []
    
# Remove NaN entries from Benefits attribute
dataset_attr = dataset.dropna(subset=[attr_name])
    
# Cleaning the titles
X_title = preprocess_data(dataset_attr['title'].values, regex)
    
# Using Count Vectorizer as features
X = vectorize_data(CountVectorizer(max_features = 10000), X_title)

y = dataset_attr[attr_name].values
    
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_title_train, X_title_test = train_test_split(X_title, test_size = 0.20, random_state = 0)
    
del X, y
    
classifier.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred = classifier.predict(X_test)
    
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
    
# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
    
y_train_pred = classifier.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred, average='weighted')
    
df = pd.DataFrame(data={'title': X_title_test, 'actual': y_test, 'pred': y_pred})
df_wrong = df[df.actual != df.pred]

cm = confusion_matrix(y_test, y_pred)

