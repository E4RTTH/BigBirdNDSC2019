from translate_selenium import Translate, Translate_Concurrent
import pandas as pd
import re

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

translator = Translate(from_lang = 'id', to_lang = 'en')

dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
titles = dataset['title'].values
titles_eng = []
chunks = list(chunks(titles, 100))
for i in range(39, len(chunks)):
    title_eng = [translator.translate(title) for title in chunks[i]]
    title_eng = [re.sub('[^a-zA-Z0-9]', ' ', title) for title in title_eng]
    titles_eng.append(title_eng)
engdf = pd.read_csv('mobile_data_info_train_competition_eng.csv', quoting = 3)
eng = engdf['eng'].values.tolist()
flat_list = [item for sublist in titles_eng for item in sublist]
eng.extend(flat_list)
df = pd.DataFrame(data={'eng': eng})
df.to_csv('mobile_data_info_train_competition_eng.csv', index=False)

del dataset, titles, titles_eng, flat_list, chunks





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