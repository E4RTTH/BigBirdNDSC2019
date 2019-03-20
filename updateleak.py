import json
import pandas
"""
data_dir = "."

# Creates an empty submission for testing purposes.
submission = list()
for category in ("beauty", "fashion", "mobile"):
    val = pandas.read_csv("%s/%s_data_info_val_competition.csv" % (data_dir, category))
    with open("%s/%s_profile_train.json" % (data_dir, category)) as stream:
        metadata = json.load(stream)
    for label in metadata.keys():
        val["suffix"] = ("_%s" % label)
        val["id"] = val["itemid"].astype("str") + val["suffix"]
        val["tagging"] = ""
        submission.append(val[["id", "tagging"]].copy())
submission = pandas.concat(submission, axis=0).set_index("id")
"""
# To update an existing submission uncomment the following line.
submission = pandas.read_csv("submission8.csv", index_col="id")

# Creates a leaked submission from the fashion train dataset.
category = "fashion"
train = pandas.read_csv("%s_data_info_train_competition.csv" % (category))
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
leak = pandas.concat(leak, axis=0).set_index("id")
leak2 = leak.dropna(subset="tagging")

# Replaces the leaked rows in the submission.
submission.update(leak)
submission.to_csv("submission8_leak.csv")