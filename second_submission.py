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

# Update stopwords database
nltk.download('stopwords')


# Functions definition --------------------------------------------------------------------------------

def preprocess_data(titles, regex):
    ps = PorterStemmer()
    data = []
    for i in range(0, len(titles)):
        title = re.sub(regex, ' ', titles[i])
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



def train_predict_data(dataset_train, dataset_val, attr_name, classifier, regex):
    
    # Some declaration and initialization
    X_train = []
    X_test = []
    
    # Remove NaN entries from Benefits attribute
    dataset_train_attr = dataset_train.dropna(subset=[attr_name])
    
    # Cleaning the titles
    X_train = preprocess_data(dataset_train_attr['title'].values, regex)
    X_test = preprocess_data(dataset_val['title'].values, regex)
    
    y_train = dataset_train_attr[attr_name].values
    
    # Using Count Vectorizer as features
    trainCount = len(X_train)
    X = X_train + X_test
    X = vectorize_data(CountVectorizer(max_features = 10000), X)
    X_train = X[0:trainCount]
    X_test = X[trainCount:len(X)]
    
    # Using TF-IDF Vectorizer (Word level) as features
    #X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000), X)
    
    # Using TF-IDF Vectorizer (NGram level) as features
    #X = vectorize_data(TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000), X)
    
    
    
    
    # Fitting Logistic Regression one vs all
    #classifier = LogisticRegression(random_state = 0, multi_class = 'ovr')
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    del X_train, X_test, y_train
        
    return y_pred


#-------------------------------------------------------------------------------------------------------




submission_df = pd.DataFrame(columns=['id', 'tagging'])

# Category: Beauty -------------------------------------------------------------------------------------

idlist = []
taglist = []

# Importing the dataset
dataset_train = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('beauty_data_info_val_competition.csv', quoting = 3)

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Benefits',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Benefits" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Beauty Benefits')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Brand',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Brand" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Beauty Brand')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Colour_group',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Colour_group" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Beauty Colour')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Product_texture',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Product_texture" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Beauty Product texture')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Skin_type',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Skin_type" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Beauty skin type')



df = pd.DataFrame(data = {'id': idlist, 'tagging': taglist})
df.sort_values(by=['id'])
submission_df = submission_df.append(df)

del y_pred, dataset_train, dataset_val, df, idlist, taglist, itemid, tag

print ('Finish writing Beauty to submission df')
    
#-------------------------------------------------------------------------------------------------------

# Category: Fashion -------------------------------------------------------------------------------------

idlist = []
taglist = []

# Importing the dataset
dataset_train = pd.read_csv('fashion_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('fashion_data_info_val_competition.csv', quoting = 3)

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Pattern',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Pattern" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Fashion Pattern')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Collar Type',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Collar Type" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Fashion Collar')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Fashion Trend',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Fashion Trend" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Fashion Trend')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Clothing Material',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Clothing Material" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Fashion Material')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Sleeves',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Sleeves" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Fashion Sleeves')


df = pd.DataFrame(data = {'id': idlist, 'tagging': taglist})
df.sort_values(by=['id'])
submission_df = submission_df.append(df)

del y_pred, dataset_train, dataset_val, df, idlist, taglist, itemid, tag

print ('Finish writing Fashion to submission df')
    
#-------------------------------------------------------------------------------------------------------



# Category: Mobile -------------------------------------------------------------------------------------

idlist = []
taglist = []

# Importing the dataset
dataset_train = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('mobile_data_info_val_competition.csv', quoting = 3)

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Operating System',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Operating System" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile OS')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Features',     \
                             LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                             '[^a-zA-Z]')

itemid = [str(i) + "_Features" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Features')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Network Connections',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z0-9]')

itemid = [str(i) + "_Network Connections" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Network Connections')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Memory RAM',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z0-9]')

itemid = [str(i) + "_Memory RAM" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile RAM')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Brand',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Brand" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Brand')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Warranty Period',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z0-9]')

itemid = [str(i) + "_Warranty Period" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Warranty')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Storage Capacity',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z0-9]')

itemid = [str(i) + "_Storage Capacity" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Storage')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Color Family',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z]')

itemid = [str(i) + "_Color Family" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Color')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Phone Model',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z0-9]')

itemid = [str(i) + "_Phone Model" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Model')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Camera',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z0-9]')

itemid = [str(i) + "_Camera" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Camera')

y_pred = train_predict_data(dataset_train,  \
                            dataset_val,    \
                            'Phone Screen Size',     \
                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                            '[^a-zA-Z0-9]')

itemid = [str(i) + "_Phone Screen Size" for i in dataset_val['itemid'].values] 
tag = [int(y) for y in y_pred] 
idlist.extend(itemid)
taglist.extend(tag)

print ('Finish predicting Mobile Size')


df = pd.DataFrame(data = {'id': idlist, 'tagging': taglist})
#df.sort_values(by=['id'])
submission_df = submission_df.append(df)

del y_pred, dataset_train, dataset_val, df, idlist, taglist, itemid, tag

print ('Finish writing Mobile to submission_df')
    
#-------------------------------------------------------------------------------------------------------

submission_df.to_csv('submission.csv', index=False)