import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

for i in range(26):
    os.system('mkdir /mnt/wangpangpang/mobile_train/mobile_color' + str(i))
    os.system('mkdir /mnt/wangpangpang/mobile_validate/mobile_color' + str(i))

dataset = pd.read_csv('mobile_data_info_train_competition.csv')
dataset = dataset.dropna(subset=['Color Family'])

x = dataset['image_path'].values
y = dataset['Color Family'].values

del dataset

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=0)
del x, y

for i in range(x_train.size):
    source_path = '/mnt/wangpangpang/' + x_train[i]
    target_path = '/mnt/wangpangpang/mobile_train/mobile_color' + str(int(y_train[i]))
    command = 'cp' + ' ' + source_path + ' ' + target_path
    os.system(command)

for i in range(x_val.size):
    source_path = '/mnt/wangpangpang/' + x_val[i]
    target_path = '/mnt/wangpangpang/mobile_validate/mobile_color' + str(int(y_val[i]))
    command = 'cp' + ' ' + source_path + ' ' + target_path
    os.system(command)
