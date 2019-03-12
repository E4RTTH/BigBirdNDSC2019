from translate_selenium import Translate, Translate_Concurrent
import pandas as pd

translator = Translate(from_lang = 'id', to_lang = 'en')

dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
titles = dataset['title'].values
titles_eng = [translator.translate(title) for title in titles]
dataset['title_eng'] = titles_eng
dataset.to_csv('mobile_data_info_train_competition_eng.csv', index=False)

del dataset, titles, titles_eng

dataset = pd.read_csv('mobile_data_info_val_competition.csv', quoting = 3)
titles = dataset['title'].values
titles_eng = [translator.translate(title) for title in titles]
dataset['title_eng'] = titles_eng
dataset.to_csv('mobile_data_info_val_competition_eng.csv', index=False)

del dataset, titles, titles_eng




dataset = pd.read_csv('beauty_data_info_train_competition.csv', quoting = 3)
titles = dataset['title'].values
titles_eng = [translator.translate(title) for title in titles]
dataset['title_eng'] = titles_eng
dataset.to_csv('beauty_data_info_train_competition_eng.csv', index=False)

del dataset, titles, titles_eng

dataset = pd.read_csv('beauty_data_info_val_competition.csv', quoting = 3)
titles = dataset['title'].values
titles_eng = [translator.translate(title) for title in titles]
dataset['title_eng'] = titles_eng
dataset.to_csv('beauty_data_info_val_competition_eng.csv', index=False)

del dataset, titles, titles_eng


dataset = pd.read_csv('fashion_data_info_train_competition.csv', quoting = 3)
titles = dataset['title'].values
titles_eng = [translator.translate(title) for title in titles]
dataset['title_eng'] = titles_eng
dataset.to_csv('fashion_data_info_train_competition_eng.csv', index=False)

del dataset, titles, titles_eng

dataset = pd.read_csv('fashion_data_info_val_competition.csv', quoting = 3)
titles = dataset['title'].values
titles_eng = [translator.translate(title) for title in titles]
dataset['title_eng'] = titles_eng
dataset.to_csv('fashion_data_info_val_competition_eng.csv', index=False)

del dataset, titles, titles_eng

print("ALL FINISHED!!")