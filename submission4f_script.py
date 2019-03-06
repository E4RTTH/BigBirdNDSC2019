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



def train_predict_data(dataset_train, dataset_val, attr_name, classifier, regex, predNum, vectorizercount):
    
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
    
    # Fitting Logistic Regression one vs all
    classifier.fit(X_train, y_train)
    
    # Calculate the probabilities of predictions
    y_pred_proba = classifier.predict_proba(X_test)
    
    # Calculate top predictions acoording to probabilities
    y_pred = calculate_top_preds(classifier.classes_, y_pred_proba, predNum)
            
    return y_pred


#-------------------------------------------------------------------------------------------------------


# Some declaration & initialization
idlist = []
taglist = []
classifier300 = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier150 = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 0)
predictionNum = 2

# Category: Beauty -------------------------------------------------------------------------------------

# Importing the dataset
dataset_train = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('beauty_data_info_val_competition.csv', quoting = 3)

y_pred_Beauty_Benefits = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Benefits',     \
                                            classifier150,  \
                                            '[^a-zA-Z]', \
                                            predictionNum, \
                                            5000)

print ('Finish predicting Beauty Benefits')

y_pred_Beauty_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         classifier150,  \
                                         '[^a-zA-Z]', \
                                         predictionNum,\
                                         5000)

print ('Finish predicting Beauty Brand')

y_pred_Beauty_Colour = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Colour_group',     \
                                          classifier150,  \
                                          '[^a-zA-Z]', \
                                          predictionNum,\
                                          5000)

print ('Finish predicting Beauty Colour')

y_pred_Beauty_Texture = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Product_texture',     \
                                           classifier150,  \
                                           '[^a-zA-Z]', \
                                           predictionNum,\
                                           5000)

print ('Finish predicting Beauty Product texture')

y_pred_Beauty_Skin = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Skin_type',     \
                                        classifier150,  \
                                        '[^a-zA-Z]', \
                                        predictionNum,\
                                        5000)

print ('Finish predicting Beauty skin type')


for i, row in dataset_val.iterrows():
    itemid = "%d_Benefits" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Beauty_Benefits[i])
    
    itemid = "%d_Brand" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Beauty_Brand[i])
    
    itemid = "%d_Colour_group" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Beauty_Colour[i])
    
    itemid = "%d_Product_texture" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Beauty_Texture[i])
    
    itemid = "%d_Skin_type" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Beauty_Skin[i])
    

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
                                            classifier150,  \
                                            '[^a-zA-Z]', \
                                            predictionNum,\
                                            5000)

print ('Finish predicting Fashion Pattern')

y_pred_Fashion_Collar = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Collar Type',     \
                                           classifier150,  \
                                           '[^a-zA-Z]', \
                                           predictionNum,\
                                           5000)

print ('Finish predicting Fashion Collar')

y_pred_Fashion_Trend = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Fashion Trend',     \
                                          classifier150,  \
                                          '[^a-zA-Z]', \
                                          predictionNum,\
                                          5000)
 
print ('Finish predicting Fashion Trend')

y_pred_Fashion_Material = train_predict_data(dataset_train,  \
                                             dataset_val,    \
                                             'Clothing Material',     \
                                             classifier150,  \
                                             '[^a-zA-Z]', \
                                             predictionNum,\
                                             5000)

print ('Finish predicting Fashion Material')

y_pred_Fashion_Sleeves = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Sleeves',     \
                                            classifier150,  \
                                            '[^a-zA-Z]', \
                                            predictionNum,\
                                            5000)

print ('Finish predicting Fashion Sleeves')


for i, row in dataset_val.iterrows():
    itemid = "%d_Pattern" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Fashion_Pattern[i])
    
    itemid = "%d_Collar Type" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Fashion_Collar[i])
    
    itemid = "%d_Fashion Trend" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Fashion_Trend[i])
    
    itemid = "%d_Clothing Material" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Fashion_Material[i])
    
    itemid = "%d_Sleeves" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Fashion_Sleeves[i])
    
    
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
                                      classifier300,  \
                                      '[^a-zA-Z]', \
                                      predictionNum,\
                                      10000)

print ('Finish predicting Mobile OS')

y_pred_Mobile_Features = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Features',     \
                                            classifier300,  \
                                            '[^a-zA-Z]', \
                                            predictionNum,\
                                            10000)

print ('Finish predicting Mobile Features')

y_pred_Mobile_Network = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Network Connections',     \
                                           classifier300,  \
                                           '[^a-zA-Z0-9]', \
                                           predictionNum,\
                                           10000)

print ('Finish predicting Mobile Network Connections')

y_pred_Mobile_RAM = train_predict_data(dataset_train,  \
                                       dataset_val,    \
                                       'Memory RAM',     \
                                       classifier300,  \
                                       '[^a-zA-Z0-9]', \
                                       predictionNum,\
                                       10000)

print ('Finish predicting Mobile RAM')

y_pred_Mobile_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         classifier300,  \
                                         '[^a-zA-Z]', \
                                         predictionNum,\
                                         10000)

print ('Finish predicting Mobile Brand')

y_pred_Mobile_Warranty = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Warranty Period',     \
                                            classifier300,  \
                                            '[^a-zA-Z0-9]', \
                                            predictionNum,\
                                            10000)

print ('Finish predicting Mobile Warranty')

y_pred_Mobile_Storage = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Storage Capacity',     \
                                           classifier300,  \
                                           '[^a-zA-Z0-9]', \
                                           predictionNum,\
                                           10000)

print ('Finish predicting Mobile Storage')

y_pred_Mobile_Color = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Color Family',     \
                                         classifier300,  \
                                         '[^a-zA-Z]', \
                                         predictionNum,\
                                         10000)

print ('Finish predicting Mobile Color')

y_pred_Mobile_Model = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Phone Model',     \
                                         classifier300,  \
                                         '[^a-zA-Z0-9]', \
                                         predictionNum,\
                                         10000)

print ('Finish predicting Mobile Model')

y_pred_Mobile_Camera = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Camera',     \
                                          classifier300,  \
                                          '[^a-zA-Z0-9]', \
                                          predictionNum,\
                                          10000)

print ('Finish predicting Mobile Camera')

y_pred_Mobile_Size = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Phone Screen Size',     \
                                        classifier300,  \
                                        '[^a-zA-Z0-9]', \
                                        predictionNum,\
                                        10000)

print ('Finish predicting Mobile Size')


for i, row in dataset_val.iterrows():
    itemid = "%d_Operating System" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_OS[i])
    
    itemid = "%d_Features" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Features[i])
    
    itemid = "%d_Network Connections" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Network[i])
    
    itemid = "%d_Memory RAM" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_RAM[i])
    
    itemid = "%d_Brand" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Brand[i])
    
    itemid = "%d_Warranty Period" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Warranty[i])
     
    itemid = "%d_Storage Capacity" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Storage[i])
    
    itemid = "%d_Color Family" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Color[i])
    
    itemid = "%d_Phone Model" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Model[i])
    
    itemid = "%d_Camera" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Camera[i])
    
    itemid = "%d_Phone Screen Size" % row['itemid']
    idlist.append(itemid)
    taglist.append(y_pred_Mobile_Size[i])
    
    
del y_pred_Mobile_OS, y_pred_Mobile_Features, y_pred_Mobile_Network, y_pred_Mobile_RAM, y_pred_Mobile_Brand
del y_pred_Mobile_Warranty, y_pred_Mobile_Storage, y_pred_Mobile_Color, y_pred_Mobile_Model, y_pred_Mobile_Camera, y_pred_Mobile_Size
del dataset_train, dataset_val

print ('Finish writing Mobile to submission_df')
    
#-------------------------------------------------------------------------------------------------------

submission_df = pd.DataFrame(data = {'id': idlist, 'tagging': taglist})
submission_df.to_csv('submission.csv', index=False)