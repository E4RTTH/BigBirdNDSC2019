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
from xgboost import XGBClassifier 

def preprocess_data(titles, regex):
    ps = PorterStemmer()
    data = []
    for item in titles:
        
        title = item
        """
        # Remove all the spaces between numbers and keyterm
        title = re.sub('(?<=(\d)) (?=(g ))', '', title) 
        title = re.sub('(?<=(\d)) (?=(gb|mb))', '', title)
        title = re.sub('(?<=(\d)) (?=(mp))', '', title)   
        title = re.sub('tahun|thn|yr', 'year', title)   
        title = re.sub('bulan|bln|mth', 'month', title)
        title = re.sub('(?<=(\d)) (?=(year|month|))', '', title)
        title = re.sub('(?<=(\d)) (?=(inch))', '', title) 
        title = re.sub('[\S]*(gb|mb|mp|year|month)', '', title) 
        """
        #
        
        
        
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
        """
        title = re.sub(' o neck ', ' oneck ', title)
        title = re.sub(' high neck ', ' highneck ', title)
        title = re.sub(' v neck ', ' vneck ', title)
        title = re.sub(' scoop neck ', ' scoopneck ', title)
        title = re.sub(' boat neck ', ' boatneck ', title)
        title = re.sub(' square neck ', ' squareneck ', title)
        title = re.sub(' v ', ' vneck ', title) 
        title = re.sub(' 3 4 ', ' 34 ', title)
        title = re.sub('bunga', 'floral', title) 
        title = re.sub('untuk', '', title) 
        title = re.sub('tempat', '', title) 
        title = re.sub('di', '', title) 
        title = re.sub('[\S]*promo[\S]*', '', title) 
        title = re.sub('[\S]*murah[\S]*', '', title) 
        title = re.sub('[\S]*diskon[\S]*', '', title) 
        title = re.sub('sale', '', title) 
        title = re.sub('dengan', '', title) 
        title = re.sub('seller', '', title) 
        
        
        title = re.sub('krem', 'cream', title) 
        title = re.sub('(?<=(natur)) (?=(republ))', '', title)
        title = re.sub('[\S]*promo[\S]*', '', title) 
        title = re.sub('[\S]*murah[\S]*', '', title) 
        title = re.sub('[\S]*new[\S]*', '', title) 
        title = re.sub('[\S]*diskon[\S]*', '', title) 
        title = re.sub('[\S]*best[\S]*', '', title) 
        title = re.sub('[\S]*sale[\S]*', '', title) 
        """
        
        
        
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

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 7, min_samples_split = 6)
#, max_depth=130

attr_name = 'Camera'
regex = '[^a-zA-Z0-9\.]'

# Some declaration and initialization
X = []
    
# Remove NaN entries from attribute
dataset_attr = dataset.dropna(subset=[attr_name])
    
# Cleaning the titles
X_title = preprocess_data(dataset_attr['title'].values, regex)
    
# Using Count Vectorizer as features
X = vectorize_data(CountVectorizer(max_features = 3000), X_title)

y = dataset_attr[attr_name].values
    
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)
X_title_train, X_title_test = train_test_split(X_title, test_size = 0.20, random_state = 5)
    
del X, y, X_title_train
    
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

#dfcontaindot = dataset[dataset.title.str.contains(' 3 ')]
#dfcontainv = dataset_attr[dataset_attr.title.str.contains(' v ')]
#dfvneck = dfcontainv[dfcontainv['Collar Type'] == 8]

#containkrem = sum([title.count("krem") for title in X_title])
#containcream = sum([title.count("cream") for title in X_title])
#containnaturerepublic = sum([title.count("natur republ") for title in X_title])