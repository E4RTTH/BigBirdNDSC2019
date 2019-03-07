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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

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



# Importing the dataset
dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)

# Update stopwords database
nltk.download('stopwords')



# Some declaration and initialization
X = []
    
# Remove NaN entries from Benefits attribute
dataset_attr = dataset.dropna(subset=['Brand'])
dataset_attr = dataset_attr.dropna(subset=['Phone Model'])
dataset_attr = dataset_attr.dropna(subset=['Camera'])

y = dataset_attr['Camera'].values
Xb = [int(i) for i in dataset_attr['Brand'].values]
Xm = [int(i) for i in dataset_attr['Phone Model'].values]

data = list(zip(Xb, Xm))

enc = OneHotEncoder()
enc.fit(data)

data = enc.transform(data).toarray()

Xt = preprocess_data(dataset_attr['title'].values, '[^a-zA-Z0-9]')

cv = CountVectorizer(max_features = 10000)
Xt = cv.fit_transform(Xt).toarray()

X = np.hstack((data,Xt))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

classifier = RandomForestClassifier(n_estimators=300, criterion = 'entropy', random_state = 0)
# MLPClassifier(alpha=0.001, hidden_layer_sizes = (100,100,100))

classifier.fit(X_train, y_train)




Xtb = Xt
yb = dataset_attr['Brand'].values

X_trainb, X_testb, y_trainb, y_testb = train_test_split(Xtb, yb, test_size = 0.20, random_state = 0)

classifier2 = RandomForestClassifier(n_estimators=300, criterion = 'entropy', random_state = 0)
classifier2.fit(Xtb, yb)
y_predb = classifier2.predict(X_testb)


Xtm = Xt
ym = dataset_attr['Phone Model'].values

X_trainm, X_testm, y_trainm, y_testm = train_test_split(Xtm, ym, test_size = 0.20, random_state = 0)

classifier3 = RandomForestClassifier(n_estimators=300, criterion = 'entropy', random_state = 0)
classifier3.fit(Xtm, ym)
y_predm = classifier2.predict(X_testm)


y_predb = [int(i) for i in y_predb]
y_predm = [int(i) for i in y_predm]

data2 = list(zip(y_predb, y_predm))
data2 = enc.transform(data2).toarray()



Xt_train, Xy_test, y_train, y_test =  train_test_split(Xt, y, test_size = 0.20, random_state = 0)

X_test = np.hstack((data2,Xy_test))


y_pred = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
    
# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')




