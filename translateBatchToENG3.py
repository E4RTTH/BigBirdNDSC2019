import pandas as pd
import textblob
import re

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]



dataset = pd.read_csv('mobile_data_info_train_competition.csv', quoting = 3)
titles = dataset['title'].values.tolist()
titles_eng = []
chunks = list(chunks(titles, 10000))
print("length chunk=", len(chunks))
for i in range(0, len(chunks)/3):
    eng_chunk = []
    print("processing chunk num ", i)
    for i, title in enumerate(chunks):
        try:
            test = str(textblob.TextBlob(title).translate(from_lang='id', to="en"))
        except Exception:
            test = title
        title = re.sub('[^a-zA-Z0-9\.]', ' ', title)
        eng_chunk.append(test)
        
    titles_eng.extend(eng_chunk)
    df = pd.DataFrame(data={'eng': titles_eng})
    df.to_csv('mobile_data_info_train_competition_eng2a.csv', index=False)




dataset['title_eng'] = titles_eng
dataset.to_csv('mobile_data_info_train_competition_eng2.csv', index=False)











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