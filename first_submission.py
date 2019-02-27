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

# Importing the dataset
dataset_train = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)
dataset_val = pd.read_csv('beauty_data_info_val_competition.csv', quoting = 3)

y_pred_Beauty_Benefits = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Benefits',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z]')

y_pred_Beauty_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z]')

y_pred_Beauty_Colour = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Colour_group',     \
                                          LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                          '[^a-zA-Z]')

y_pred_Beauty_Texture = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Product_texture',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z]')

y_pred_Beauty_Skin = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Skin_type',     \
                                        LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                        '[^a-zA-Z]')


for i in range(0, len(dataset_val)):
    itemid = "%d_Benefits" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Beauty_Benefits[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Brand" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Beauty_Brand[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Colour_group" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Beauty_Colour[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Product_texture" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Beauty_Texture[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Skin_type" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Beauty_Skin[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    

del y_pred_Beauty_Benefits, y_pred_Beauty_Brand, y_pred_Beauty_Colour, y_pred_Beauty_Texture, y_pred_Beauty_Skin
del dataset_train, dataset_val
    
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

y_pred_Fashion_Collar = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Collar Type',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z]')

y_pred_Fashion_Trend = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Fashion Trend',     \
                                          LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                          '[^a-zA-Z]')

y_pred_Fashion_Material = train_predict_data(dataset_train,  \
                                             dataset_val,    \
                                             'Clothing Material',     \
                                             LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                             '[^a-zA-Z]')

y_pred_Fashion_Sleeves = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Sleeves',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z]')


for i in range(0, len(dataset_val)):
    itemid = "%d_Pattern" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Fashion_Pattern[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Collar Type" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Fashion_Collar[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Fashion Trend" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Fashion_Trend[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Clothing Material" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Fashion_Material[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Sleeves" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Fashion_Sleeves[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    
del y_pred_Fashion_Pattern, y_pred_Fashion_Collar, y_pred_Fashion_Fashion, y_pred_Fashion_Material, y_pred_Fashion_Sleeves
del dataset_train, dataset_val
    
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

y_pred_Mobile_Features = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Features',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z]')

y_pred_Mobile_Network = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Network Connections',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z0-9]')

y_pred_Mobile_RAM = train_predict_data(dataset_train,  \
                                       dataset_val,    \
                                       'Memory RAM',     \
                                       LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                       '[^a-zA-Z0-9]')

y_pred_Mobile_Brand = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Brand',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z]')

y_pred_Mobile_Warranty = train_predict_data(dataset_train,  \
                                            dataset_val,    \
                                            'Warranty Period',     \
                                            LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                            '[^a-zA-Z0-9]')

y_pred_Mobile_Storage = train_predict_data(dataset_train,  \
                                           dataset_val,    \
                                           'Storage Capacity',     \
                                           LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                           '[^a-zA-Z0-9]')

y_pred_Mobile_Color = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Color Family',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z]')

y_pred_Mobile_Model = train_predict_data(dataset_train,  \
                                         dataset_val,    \
                                         'Phone Model',     \
                                         LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                         '[^a-zA-Z0-9]')

y_pred_Mobile_Camera = train_predict_data(dataset_train,  \
                                          dataset_val,    \
                                          'Camera',     \
                                          LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                          '[^a-zA-Z0-9]')

y_pred_Mobile_Size = train_predict_data(dataset_train,  \
                                        dataset_val,    \
                                        'Phone Screen Size',     \
                                        LogisticRegression(random_state = 0, multi_class = 'ovr'),  \
                                        '[^a-zA-Z0-9]')


for i in range(0, len(dataset_val)):
    itemid = "%d_Operating System" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_OS[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Features" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Features[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Network Connections" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Network[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Memory RAM" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_RAM[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Brand" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Brand[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Warranty Period" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Warranty[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
     
    itemid = "%d_Storage Capacity" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Storage[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Color Family" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Color[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Phone Model" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Model[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Camera" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Camera[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    itemid = "%d_Phone Screen Size" % dataset_val['itemid'].values[i]
    tag = int(y_pred_Mobile_Size[i])
    df = pd.DataFrame(data = {'id': [itemid], 'tagging': [tag]})
    submission_df = submission_df.append(df)
    
    
del y_pred_Mobile_OS, y_pred_Mobile_Features, y_pred_Mobile_Network, y_pred_Mobile_RAM, y_pred_Mobile_Brand
del y_pred_Mobile_Warranty, y_pred_Mobile_Storage, y_pred_Mobile_Color, y_pred_Mobile_Model, y_pred_Mobile_Camera, y_pred_Mobile_Size
del dataset_train, dataset_val
    
#-------------------------------------------------------------------------------------------------------

submission_df.to_csv('submission.csv', index=False)