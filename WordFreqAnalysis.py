# Importing the libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist
import csv

def preprocess_data(titles, regex):
    ps = PorterStemmer()
    data = []
    for item in titles:
        
        title = item
        title = re.sub(regex, ' ', title)
        title = title.lower()
        title = title.split()
        title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
        #title = ' '.join(title)
        data.append(title)
        
    del titles
        
    return data

def vectorize_data(vectorizer, data):
    vectors = vectorizer.fit_transform(data).toarray()
    return vectors




# Importing the dataset
dataset = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)

# Update stopwords database
nltk.download('stopwords')

regex = '[^a-zA-Z0-9\.]'

# Cleaning the titles
X_title = preprocess_data(dataset['title'].values, regex)

freq_dist = FreqDist([word for title in X_title for word in title])


with open('fdist_beauty.csv','w') as csvfile:
    fieldnames=['word','freq']
    writer=csv.writer(csvfile)
    writer.writerow(fieldnames)
    for key, value in freq_dist.items():
        writer.writerow([key] + [value]) 


