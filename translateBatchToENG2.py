
import pandas as pd
import textblob
import re

dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
titles = dataset['title'].values.tolist()

titles_eng = []
for i, title in enumerate(titles):
    try:
        test = str(textblob.TextBlob(title).translate(from_lang='id', to="en"))
    except Exception:
        test = title
    title = re.sub('[^a-zA-Z0-9\.]', ' ', title)
    titles_eng.append(test)

dataset['title_eng'] = titles_eng
dataset.to_csv('mobile_data_info_train_competition_eng2.csv', index=False)