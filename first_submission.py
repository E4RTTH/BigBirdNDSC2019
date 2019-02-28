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
    
    # Fitting Logistic Regression one vs all
    #classifier = LogisticRegression(random_state = 0, multi_class = 'ovr')
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    del X_train, X_test, y_train
        
    return y_pred


#-------------------------------------------------------------------------------------------------------



idlist = []
taglist = []

# Category: Beauty -------------------------------------------------------------------------------------

# Importing the dataset
dataset_train = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('beauty_data_info_val_competition.csv', quoting = 3)

y_pred_Beauty_Benefits = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Benefits',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z]')

print ('Finish predicting Beauty Benefits')

y_pred_Beauty_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z]')

print ('Finish predicting Beauty Brand')

y_pred_Beauty_Colour = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Colour_group',     \
                                          LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                          '[^a-zA-Z]')

print ('Finish predicting Beauty Colour')

y_pred_Beauty_Texture = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Product_texture',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z]')

print ('Finish predicting Beauty Product texture')

y_pred_Beauty_Skin = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Skin_type',     \
                                        LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
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
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z]')

print ('Finish predicting Fashion Pattern')

y_pred_Fashion_Collar = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Collar Type',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z]')

print ('Finish predicting Fashion Collar')

y_pred_Fashion_Trend = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Fashion Trend',     \
                                          LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                          '[^a-zA-Z]')

print ('Finish predicting Fashion Trend')

y_pred_Fashion_Material = train_predict_data(dataset_train,  \
                                             dataset_val,    \
                                             'Clothing Material',     \
                                             LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                             '[^a-zA-Z]')

print ('Finish predicting Fashion Material')

y_pred_Fashion_Sleeves = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Sleeves',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
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
                                      LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                      '[^a-zA-Z]')

print ('Finish predicting Mobile OS')

y_pred_Mobile_Features = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Features',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z]')

print ('Finish predicting Mobile Features')

y_pred_Mobile_Network = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Network Connections',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Network Connections')

y_pred_Mobile_RAM = train_predict_data(dataset_train,  \
                                       dataset_val,    \
                                       'Memory RAM',     \
                                       LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                       '[^a-zA-Z0-9]')

print ('Finish predicting Mobile RAM')

y_pred_Mobile_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z]')

print ('Finish predicting Mobile Brand')

y_pred_Mobile_Warranty = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Warranty Period',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Warranty')

y_pred_Mobile_Storage = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Storage Capacity',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Storage')

y_pred_Mobile_Color = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Color Family',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z]')

print ('Finish predicting Mobile Color')

y_pred_Mobile_Model = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Phone Model',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Model')

y_pred_Mobile_Camera = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Camera',     \
                                          LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                          '[^a-zA-Z0-9]')

print ('Finish predicting Mobile Camera')

y_pred_Mobile_Size = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Phone Screen Size',     \
                                        LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
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