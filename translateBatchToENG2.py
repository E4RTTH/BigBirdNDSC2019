
import pandas as pd
import textblob
import time


dataset = pd.read_csv('mobile_data_info_val_competition.csv', quoting = 3)
titles = dataset['title'].values.tolist()
#test = str(textblob.TextBlob(titles[1]).translate(from_lang='id', to="en"))
titles_eng = []
for i, title in enumerate(titles):
    try:
        test = str(textblob.TextBlob(title).translate(from_lang='id', to="en"))
    except Exception:
        test = title
    print(i)
    titles_eng.append(test)
    #time.sleep(5)

#titles_eng = [str(textblob.TextBlob(title).translate(from_lang='id', to="en")) for title in titles]
dataset['title_eng'] = titles_eng
dataset.to_csv('mobile_data_info_val_competition_eng.csv', index=False)