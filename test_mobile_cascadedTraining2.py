# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

def calculate_top_preds(y_classes, y_pred_proba, topNum):
    
    y_pred = []
    
    # 1. Loop through all prediction probabilities datasets
    # 2. Sort through the probability list and extract the index of top probabilities based on input argument
    # 3. Use the indices and extract the class labels 
    # 4. Format it to space separated probability list
    for probas in y_pred_proba:
        clsidx = [probas.tolist().index(x) for x in sorted(probas, reverse=True)[:topNum]]
        pred = [int(y_classes[i]) for i in clsidx]
        y_pred.append(pred)
    
    return y_pred



# Importing the dataset
dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)

dataset = dataset.drop(columns=['image_path','Operating System','Features','Network Connections','Memory RAM','Brand','Warranty Period','Storage Capacity','Color Family','Phone Screen Size'])
dataset_M = dataset.dropna(subset=['Phone Model'])
dataset_MC = dataset_M.dropna(subset=['Camera'])
dataset_MwithoutC = dataset_M[dataset_M['Camera'].isnull()]

X_train = dataset_MC['Phone Model'].values
y_train = dataset_MC['Camera'].values
X_pred = dataset_MwithoutC['Phone Model'].values

onehotencoder = OneHotEncoder(n_values = 2280)
X_train = onehotencoder.fit_transform(X_train.reshape(-1, 1)).toarray()
X_pred = onehotencoder.fit_transform(X_pred.reshape(-1, 1)).toarray()

classifier = RandomForestClassifier(n_estimators=300, criterion = 'gini', random_state = 0, min_samples_split = 6)
#classifier = LogisticRegression(random_state = 0, multi_class = 'ovr', solver = 'lbfgs')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_pred)

dataset_built = pd.DataFrame(data = {'itemid': dataset_MwithoutC['itemid'].values, 
                                     'title': dataset_MwithoutC['title'].values,
                                     'Phone Model': dataset_MwithoutC['Phone Model'].values,
                                     'Camera': y_pred})

dataset_built = dataset_built.append(dataset_MC)

X_train = dataset_built['Phone Model'].values
y_train = dataset_built['Camera'].values

X_train = onehotencoder.fit_transform(X_train.reshape(-1, 1)).toarray()

classifier2 = RandomForestClassifier(n_estimators=300, criterion = 'gini', random_state = 0, min_samples_split = 6)
classifier2.fit(X_train, y_train)

dataset_sub = pd.read_csv('submission8.csv')
dataset_sub = dataset_sub[dataset_sub.id.str.contains("Phone Model")]

idlist = []
modellist1 = []
modellist2 = []

for index, row in dataset_sub.iterrows():
    a,b = row.id.split('_')
    idlist.append(a)
    model1, model2 = row.tagging.split()
    modellist1.append(model1)
    modellist2.append(model2)
    
modellist1 = onehotencoder.fit_transform(np.asarray(modellist1).reshape(-1, 1)).toarray()
modellist2 = onehotencoder.fit_transform(np.asarray(modellist2).reshape(-1, 1)).toarray()
    
y_pred_proba1 = classifier2.predict_proba(modellist1)
y_pred2 = classifier2.predict(modellist2)
y_pred = []

y_pred1 = calculate_top_preds(classifier2.classes_, y_pred_proba1, 2)

for i in range(len(y_pred1)):
    pred1 = int(y_pred1[i][0])
    if y_pred1[i][0] == y_pred2[i]:
        pred2 = int(y_pred1[i][1])
    else:
        pred2 = int(y_pred2[i])
    pred = '{} {}'.format(pred1, pred2)
    y_pred.append(pred)
    idlist[i] = '{}_{}'.format(idlist[i], "Camera")
    
resultdf = pd.read_csv('submission8.csv')
resultdf = resultdf[~resultdf.id.str.contains("Camera")]


submission = pd.DataFrame(data = {'id': idlist, 'tagging': y_pred})
resultdf = resultdf.append(submission)
resultdf.to_csv('submission8a.csv', index=False)



