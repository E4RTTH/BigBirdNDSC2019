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

def preprocess_data(titles):
    ps = PorterStemmer()
    data = []
    for i in range(0, len(titles)):
        title = re.sub('[^a-zA-Z0-9]', ' ', titles[i])
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



# Importing the dataset
dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)

# Update stopwords database
nltk.download('stopwords')



cm_OS, acc_OS, f1_OS = train_predict_data(dataset, 'Operating System', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("OS: acc=", acc_OS, ", f1=", f1_OS)
cm_Features, acc_Features, f1_Features = train_predict_data(dataset, 'Features', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Features: acc=", acc_Features, ", f1=", f1_Features)
cm_Network, acc_Network, f1_Network = train_predict_data(dataset, 'Network Connections', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Network Connections: acc=", acc_Network, ", f1=", f1_Network)
cm_RAM, acc_RAM, f1_RAM = train_predict_data(dataset, 'Memory RAM', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Memory RAM: acc=", acc_RAM, ", f1=", f1_RAM)
cm_Brand, acc_Brand, f1_Brand = train_predict_data(dataset, 'Brand', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Brand: acc=", acc_Brand, ", f1=", f1_Brand)
cm_Warranty, acc_Warranty, f1_Warranty = train_predict_data(dataset, 'Warranty Period', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Warranty Period: acc=", acc_Warranty, ", f1=", f1_Warranty)
cm_Storage, acc_Storage, f1_Storage = train_predict_data(dataset, 'Storage Capacity', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Storage Capacity: acc=", acc_Storage, ", f1=", f1_Storage)
cm_Color, acc_Color, f1_Color = train_predict_data(dataset, 'Color Family', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Color Family: acc=", acc_Color, ", f1=", f1_Color)
cm_Model, acc_Model, f1_Model = train_predict_data(dataset, 'Phone Model', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Phone Model: acc=", acc_Model, ", f1=", f1_Model)
cm_Camera, acc_Camera, f1_Camera = train_predict_data(dataset, 'Camera', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Camera: acc=", acc_Camera, ", f1=", f1_Camera)
cm_Size, acc_Size, f1_Size = train_predict_data(dataset, 'Phone Screen Size', LogisticRegression(random_state = 0, multi_class = 'ovr'))
print("Phone Screen Size: acc=", acc_Size, ", f1=", f1_Size)





"""
# Processing Benefits attribute tags
#----------------------------------------------------------------------------------------------

X = []

# Remove NaN entries from Benefits attribute
dataset_BeautyBenefits = dataset.dropna(subset=['Benefits'])

#titles =  dataset_BeautyBenefits['title'].values

# Cleaning the texts
X = preprocess_data(dataset_BeautyBenefits['title'].values)

# Using Count Vectorizer as features
X = vectorize_data(CountVectorizer(max_features = 10000), X)

# Using TF-IDF Vectorizer (Word level) as features
#X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000), X)

# Using TF-IDF Vectorizer (NGram level) as features
#X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000), X)


y = dataset_BeautyBenefits['Benefits'].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

del X, y

# Fitting Logistic Regression one vs all
classifier = LogisticRegression(random_state = 0, multi_class = 'ovr')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm_BeautyBenefits = confusion_matrix(y_test, y_pred)

# Calculate accuracy score
accuracy_BeautyBenefits = accuracy_score(y_test, y_pred)

# Calculate F1 score
f1_BeautyBenefits = f1_score(y_test, y_pred, average='weighted')

del X_train, X_test, y_train, y_test, y_pred

#----------------------------------------------------------------------------------------------

"""


