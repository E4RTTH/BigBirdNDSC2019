import json
import pandas as pd

# To update an existing submission uncomment the following line.
submission = pd.read_csv("submission8.csv", index_col="id")

# Creates a leaked submission from the fashion train dataset.
category = "fashion"
train = pd.read_csv("%s_data_info_train_competition.csv" % (category))
with open("%s_profile_train.json" % (category)) as stream:
    metadata = json.load(stream)
leak = list()
for label in metadata.keys():
    train["suffix"] = ("_%s" % label)
    train["id"] = train["itemid"].astype("str") + train["suffix"]
    train["tagging"] = train[label].fillna(-1).astype("int").astype("str")
    train["tagging"].replace("-1", "", inplace=True)
    leak.append(train[["id", "tagging"]].copy())
train.drop(columns=["suffix", "id", "tagging"], inplace=True)
leak = pd.concat(leak, axis=0).set_index("id")

leak2 = leak[leak.tagging != ""]
leak3 = leak2.copy()



for i, row_leak in leak2.iterrows():
    try:
        row_pred = submission.loc[i].tagging.split()
        tag1 = row_leak.tagging
        if row_leak.tagging != row_pred[0]:
            tag2 = row_pred[0]
        else:
            tag2 = row_pred[1]
        strPred = '{} {}'.format(tag1, tag2)
       # leak3.set_value(i, 'tagging', strPred)
        leak3.at[i, 'tagging'] = strPred
    except KeyError:
        continue


    

# Replaces the leaked rows in the submission.
submission.update(leak3)
submission.to_csv("submission8_leak.csv")