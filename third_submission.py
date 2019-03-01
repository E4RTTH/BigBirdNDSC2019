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
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
        
        # Join the list of words back with string as seperator
        title = ' '.join(title)
        
        # Append the preprocessed text back to dataset
        data.append(title)
                
    return data



def vectorize_data(vectorizer, data):
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors

def calculate_top_preds(y_pred_proba, topNum):
    
    y_pred = []
    
    # 1. Loop through all prediction probabilities datasets
    # 2. Loop through the number of top predictions to be extracted
    # 3. Find the max value in probability list
    # 4. Copy out the index of max probability
    # 5. Replace the max probability with 0
    # 6. Repeat from Step 4 until number of top predictions achieved
    for probas in y_pred_proba:
        pred = []
        for i in range(topNum):
            m = max(probas)
            for i, j in enumerate(probas):
                if j == m:
                    pred.append(i)
                    probas[i] = 0
                    break
        strPred = ' '.join(str(x) for x in pred)
        y_pred.append(strPred)
    
    return y_pred



def train_predict_data(dataset_train, dataset_val, attr_name, classifier, regex):
    
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
    X = vectorize_data(CountVectorizer(max_features = 10000), X)
    X_train = X[0:trainCount]
    X_test = X[trainCount:len(X)]
    
    # Fitting Logistic Regression one vs all
    classifier.fit(X_train, y_train)
    
    # Calculate the probabilities of predictions
    y_pred_proba = classifier.predict_proba(X_test)
    
    # Calculate top predictions acoording to probabilities
    y_pred = calculate_top_preds(y_pred_proba, 2)
    
    del X_train, X_test, y_train
        
    return y_pred


#-------------------------------------------------------------------------------------------------------


# Some declaration
idlist = []
taglist = []

# Category: Beauty -------------------------------------------------------------------------------------

# Importing the dataset
dataset_train = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('beauty_data_info_val_competition.csv', quoting = 3)

y_pred_Beauty_Benefits = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Benefits',     \
                                            RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                            '[^a-zA-Z]')

print ('Finish predicting Beauty Benefits')

y_pred_Beauty_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                         '[^a-zA-Z]')

print ('Finish predicting Beauty Brand')

y_pred_Beauty_Colour = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Colour_group',     \
                                          RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                          '[^a-zA-Z]')

print ('Finish predicting Beauty Colour')

y_pred_Beauty_Texture = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Product_texture',     \
                                           RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                           '[^a-zA-Z]')

print ('Finish predicting Beauty Product texture')

y_pred_Beauty_Skin = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Skin_type',     \
                                        RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                        '[^a-zA-Z]')

print ('Finish predicting Beauty skin type')


for i, row in dataset_val.iterrows():
    itemid = "%d_Benefits" % row['itemid']
    tag = int(y_pred_Beauty_Benefits[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Brand" % row['itemid']
    tag = int(y_pred_Beauty_Brand[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Colour_group" % row['itemid']
    tag = int(y_pred_Beauty_Colour[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Product_texture" % row['itemid']
    tag = int(y_pred_Beauty_Texture[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Skin_type" % row['itemid']
    tag = int(y_pred_Beauty_Skin[i])
    idlist.append(itemid)
    taglist.append(tag)
    

del y_pred_Beauty_Benefits, y_pred_Beauty_Brand, y_pred_Beauty_Colour, y_pred_Beauty_Texture, y_pred_Beauty_Skin
del dataset_train, dataset_val

print ('Finish writing Beauty to submission df')
    
#-------------------------------------------------------------------------------------------------------

# Category: Fashion -------------------------------------------------------------------------------------

# Importing the dataset
dataset_train = pd.read_csv('fashion_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('fashion_data_info_val_competition.csv', quoting = 3)

y_pred_Fashion_Pattern = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Pattern',     \
                                            RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                            '[^a-zA-Z]')

print ('Finish predicting Fashion Pattern')

y_pred_Fashion_Collar = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Collar Type',     \
                                           RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                           '[^a-zA-Z]')

print ('Finish predicting Fashion Collar')

y_pred_Fashion_Trend = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Fashion Trend',     \
                                          RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                          '[^a-zA-Z]')

print ('Finish predicting Fashion Trend')

y_pred_Fashion_Material = train_predict_data(dataset_train,  \
                                             dataset_val,    \
                                             'Clothing Material',     \
                                             RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                             '[^a-zA-Z]')

print ('Finish predicting Fashion Material')

y_pred_Fashion_Sleeves = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Sleeves',     \
                                            RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                            '[^a-zA-Z]')

print ('Finish predicting Fashion Sleeves')


for i, row in dataset_val.iterrows():
    itemid = "%d_Pattern" % row['itemid']
    tag = int(y_pred_Fashion_Pattern[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Collar Type" % row['itemid']
    tag = int(y_pred_Fashion_Collar[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Fashion Trend" % row['itemid']
    tag = int(y_pred_Fashion_Trend[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Clothing Material" % row['itemid']
    tag = int(y_pred_Fashion_Material[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Sleeves" % row['itemid']
    tag = int(y_pred_Fashion_Sleeves[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    
del y_pred_Fashion_Pattern, y_pred_Fashion_Collar, y_pred_Fashion_Trend, y_pred_Fashion_Material, y_pred_Fashion_Sleeves
del dataset_train, dataset_val

print ('Finish writing Fashion to submission df')
    
#-------------------------------------------------------------------------------------------------------



# Category: Mobile -------------------------------------------------------------------------------------

# Importing the dataset
dataset_train = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('mobile_data_info_val_competition.csv', quoting = 3)

y_pred_Mobile_OS = train_predict_data(dataset_train,  \
                                      dataset_val,    \
                                      'Operating System',     \
                                      RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                      '[^a-zA-Z]')

print ('Finish predicting Mobile OS')

y_pred_Mobile_Features = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Features',     \
                                            RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                            '[^a-zA-Z]')

print ('Finish predicting Mobile Features')

y_pred_Mobile_Network = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Network Connections',     \
                                           RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                           '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Network Connections')

y_pred_Mobile_RAM = train_predict_data(dataset_train,  \
                                       dataset_val,    \
                                       'Memory RAM',     \
                                       RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                       '[^a-zA-Z0-9]')

print ('Finish predicting Mobile RAM')

y_pred_Mobile_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                         '[^a-zA-Z]')

print ('Finish predicting Mobile Brand')

y_pred_Mobile_Warranty = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Warranty Period',     \
                                            RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                            '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Warranty')

y_pred_Mobile_Storage = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Storage Capacity',     \
                                           RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                           '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Storage')

y_pred_Mobile_Color = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Color Family',     \
                                         RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                         '[^a-zA-Z]')

print ('Finish predicting Mobile Color')

y_pred_Mobile_Model = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Phone Model',     \
                                         RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                         '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Model')

y_pred_Mobile_Camera = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Camera',     \
                                          RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                          '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Camera')

y_pred_Mobile_Size = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Phone Screen Size',     \
                                        RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0),  \
                                        '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Size')


for i, row in dataset_val.iterrows():
    itemid = "%d_Operating System" % row['itemid']
    tag = int(y_pred_Mobile_OS[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Features" % row['itemid']
    tag = int(y_pred_Mobile_Features[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Network Connections" % row['itemid']
    tag = int(y_pred_Mobile_Network[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Memory RAM" % row['itemid']
    tag = int(y_pred_Mobile_RAM[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Brand" % row['itemid']
    tag = int(y_pred_Mobile_Brand[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Warranty Period" % row['itemid']
    tag = int(y_pred_Mobile_Warranty[i])
    idlist.append(itemid)
    taglist.append(tag)
     
    itemid = "%d_Storage Capacity" % row['itemid']
    tag = int(y_pred_Mobile_Storage[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Color Family" % row['itemid']
    tag = int(y_pred_Mobile_Color[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Phone Model" % row['itemid']
    tag = int(y_pred_Mobile_Model[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Camera" % row['itemid']
    tag = int(y_pred_Mobile_Camera[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    itemid = "%d_Phone Screen Size" % row['itemid']
    tag = int(y_pred_Mobile_Size[i])
    idlist.append(itemid)
    taglist.append(tag)
    
    
del y_pred_Mobile_OS, y_pred_Mobile_Features, y_pred_Mobile_Network, y_pred_Mobile_RAM, y_pred_Mobile_Brand
del y_pred_Mobile_Warranty, y_pred_Mobile_Storage, y_pred_Mobile_Color, y_pred_Mobile_Model, y_pred_Mobile_Camera, y_pred_Mobile_Size
del dataset_train, dataset_val

print ('Finish writing Mobile to submission_df')
    
#-------------------------------------------------------------------------------------------------------

submission_df = pd.DataFrame(data = {'id': idlist, 'tagging': taglist})
submission_df.to_csv('submission.csv', index=False)