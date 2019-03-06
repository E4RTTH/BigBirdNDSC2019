import pandas as pd 
import numpy as np 
import matplotlib.image as img 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('mobile_data_info_train_competition.csv')
evl_dataset = pd.read_csv('mobile_data_info_val_competition.csv')

dataset = dataset.dropna(subset=['Color Family'])

# extract training data
X = []
y = []
for index, row in dataset.iterrows():
    image_path = '/mnt/wangpangpang/' + row['image_path']
    im = img.imread(image_path)
    # print(im.shape)
    num_pixel = im.size / 3
    hist_channel0 = np.histogram(im[:,:,0], bins=256)[0]
    hist_channel1 = np.histogram(im[:,:,1], bins=256)[0]
    hist_channel2 = np.histogram(im[:,:,2], bins=256)[0]
    hist_norm = np.concatenate([hist_channel0, hist_channel1, hist_channel2]) / num_pixel
    # print(hist_norm)
    # print(hist.shape)
    # plt.figure()
    # plt.plot(hist_norm)
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    X.append(hist_norm)
    y.append(row['Color Family'])

del dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

print("Colour Family: acc=", accuracy, ", f1=", f1)

# extract evaluation data
# for index, row in evl_dataset.iterrows():
#     image_path = '/mnt/wangpangpang/' + row['image_path']
#     im = img.imread(image_path)
#     row['image'] = im